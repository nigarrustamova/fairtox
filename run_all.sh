#!/usr/bin/env bash
#
# The single command that reproduces the headline result from a clean checkout.
#
#   bash run_all.sh                      # the booked-window configuration
#   bash run_all.sh configs/smoke.yaml   # synthetic data, CPU, ~10 minutes
#
# Every stage the paper's claim rests on comes out of ONE config file, so the two
# arms cannot drift apart.
set -euo pipefail

CONFIG="${1:-configs/a100.yaml}"
shift || true

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=================================================================="
echo " FairTox -- full MUST tier"
echo " config: ${CONFIG}"
echo "=================================================================="

python scripts/run_all.py --config "${CONFIG}" --tier must "$@"

echo
echo "Done. Headline artefacts:"
echo "  results/<run>/tables/tab02_headline_comparison.md"
echo "  results/<run>/figures/fig05_fpr_by_subgroup.png"
echo "  results/<run>/metrics/compare.json"
