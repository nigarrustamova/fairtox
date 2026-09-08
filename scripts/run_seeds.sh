#!/usr/bin/env bash
#
# Replicate the whole study under several seeds, then aggregate.
#
#   bash scripts/run_seeds.sh                 # 42 43 44 with configs/a100.yaml
#   bash scripts/run_seeds.sh 42 43 44 45
#   CONFIG=configs/smoke.yaml bash scripts/run_seeds.sh 1 2 3
#
# STAGES= replaces the MUST tier with a named set, dependencies still resolved:
#
#   CONFIG=configs/a100_cda.yaml BASE_NAME=cda \
#     STAGES="compare mechanism_analysis jigsaw_bias_metrics postproc_thresholds" \
#     bash scripts/run_seeds.sh 42 43 44
#
# It exists for a mitigation whose knobs the MUST tier does not sweep: `ablate`
# varies alpha, and the augmentation arm has no alpha, so running the full tier
# there would spend six GPU-hours per seed measuring a knob that is not connected
# to anything.
#
# `run.seed` drives the partition, the shuffling and the head initialisation, so
# each seed is a fresh replication of the whole experiment -- while inside a seed
# the two arms still differ only in the loss weighting.
#
# The first seed writes to `results/<base>`, the rest to `results/<base>_s<seed>`.
# The aggregation is handed the EXACT run names this invocation produced, never a
# `<base>*` glob: `dro*` also matches `drob*`, so aggregating Group DRO silently
# pulled the three DistilRoBERTa runs into the same mean. The stage caught it --
# it warned that the seeds disagreed on the outcome letter -- but a warning in a
# log is not a guard.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/a100.yaml}"
BASE_NAME="${BASE_NAME:-main}"
STAGES="${STAGES:-}"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(42 43 44)
fi

echo "=================================================================="
echo " FairTox -- seed replication"
echo " config: ${CONFIG}"
echo " seeds:  ${SEEDS[*]}"
echo " stages: ${STAGES:-<the whole MUST tier>}"
echo "=================================================================="

FIRST=1
for seed in "${SEEDS[@]}"; do
  if [[ ${FIRST} -eq 1 ]]; then
    run_name="${BASE_NAME}"
    FIRST=0
  else
    run_name="${BASE_NAME}_s${seed}"
  fi

  echo
  echo "------------------------------------------------------------------"
  echo " seed ${seed}  ->  results/${run_name}"
  echo "------------------------------------------------------------------"

  # Runs already on disk under identical settings are skipped, so an interrupted
  # sweep can be restarted with the same command and picks up where it stopped.
  if [[ -n "${STAGES}" ]]; then
    # Unquoted on purpose: STAGES is a list of stage names and has to split.
    # shellcheck disable=SC2086
    python scripts/run_all.py --config "${CONFIG}" --only ${STAGES} \
      --set "run.seed=${seed}" --set "run.name=${run_name}"
  else
    python scripts/run_all.py --config "${CONFIG}" --tier must \
      --set "run.seed=${seed}" --set "run.name=${run_name}"
  fi
done

echo
echo "------------------------------------------------------------------"
echo " Aggregating across seeds (no GPU)"
echo "------------------------------------------------------------------"

# --no-deps because this reads saved compare.json files and must never be able to
# queue a training stage. `analysis.seed_runs` is an explicit list of exactly the
# runs this invocation produced, so no sibling family can be swept in by a name
# that happens to share a prefix.
RUN_LIST="["
SEP=""
for seed in "${SEEDS[@]}"; do
  if [[ -z "${SEP}" ]]; then
    RUN_LIST+="'${BASE_NAME}'"
    SEP=","
  else
    RUN_LIST+="${SEP}'${BASE_NAME}_s${seed}'"
  fi
done
RUN_LIST+="]"

python scripts/run_all.py --config "${CONFIG}" \
  --only seed_robustness --no-deps \
  --set "run.name=${BASE_NAME}" \
  --set "analysis.seed_runs=${RUN_LIST}"

echo
echo "Done. Read these before writing anything up:"
echo "  results/${BASE_NAME}/tables/tab09_seed_robustness.md   per-seed deltas"
echo "  results/${BASE_NAME}/metrics/seed_robustness.json      mean, spread, sign agreement"
echo
echo "If the mean change is smaller than its spread, the honest write-up is"
echo "'inconclusive' -- the stage says so in its own log."
