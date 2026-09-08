#!/usr/bin/env bash
# Fetch the Jigsaw Unintended Bias in Toxicity Classification corpus.
#
#   bash scripts/download_data.sh [kaggle|huggingface]
#
# The route we actually used was manual: download all_data.csv.zip (~326 MB) from
# the competition's Data tab and unzip it into data/raw/. No token, no CLI.
# Licence: CC0 -- cite it in the report.
#
# Hugging Face `google/civil_comments` is not a substitute: its `identity_attack`
# column is a kind of toxicity, not a record that a comment mentions a group, so
# the subgroup audit would have nothing to slice on.
set -euo pipefail

SOURCE="${1:-kaggle}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

# The pipeline picks the first of train.csv / all_data.csv it finds, so a
# leftover synthetic train.csv would silently shadow the real corpus.
for existing in "${RAW_DIR}/train.csv" "${RAW_DIR}/all_data.csv"; do
  [[ -f "${existing}" ]] || continue
  if head -1 "${existing}" | grep -q "synthetic_marker"; then
    echo "$(basename "${existing}") is SYNTHETIC test data -- removing it."
    rm -f "${existing}"
  else
    echo "$(basename "${existing}") is already present in data/raw -- nothing to do."
    echo "Delete it first if you want to re-download."
    exit 0
  fi
done

case "${SOURCE}" in
  kaggle)
    echo "Downloading from Kaggle (jigsaw-unintended-bias-in-toxicity-classification)"
    if ! command -v kaggle >/dev/null 2>&1; then
      echo "ERROR: the 'kaggle' CLI is not on PATH. Install it with: pip install kaggle" >&2
      exit 1
    fi
    if ! kaggle competitions download \
        -c jigsaw-unintended-bias-in-toxicity-classification \
        -f all_data.csv -p "${RAW_DIR}"; then
      {
        echo
        echo "Download failed. The usual cause is that nobody has accepted the"
        echo "competition rules on this Kaggle account yet. Open this page while"
        echo "signed in, accept, then re-run:"
        echo "  https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules"
        echo
        echo "Newer Kaggle CLIs also want a token from"
        echo "  https://www.kaggle.com/settings/api"
        echo
        echo "Or start today without the gate:"
        echo "  bash scripts/download_data.sh huggingface"
      } >&2
      exit 1
    fi
    for archive in "${RAW_DIR}"/*.zip; do
      [[ -f "${archive}" ]] || continue
      unzip -o "${archive}" -d "${RAW_DIR}"
      rm -f "${archive}"
    done
    ;;

  huggingface|hf)
    # FALLBACK ONLY. This mirror carries ~125k rows, of which ~28k are
    # annotated -- enough to develop against, but only a handful of subgroups
    # clear the support floor on the test split, and the ones that drop out
    # (black, transgender, homosexual_gay_or_lesbian, jewish) are the groups
    # this project is about. Do not put its numbers in the paper.
    echo "Downloading from Hugging Face (james-burton/jigsaw_unintended_bias100K)"
    echo "WARNING: development mirror. Too few subgroups clear the floor for the paper."
    RAW_DIR="${RAW_DIR}" python - <<'PY'
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

raw_dir = Path(os.environ["RAW_DIR"])
raw_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("james-burton/jigsaw_unintended_bias100K")
frames = []
for split in dataset:
    part = dataset[split].to_pandas()
    part["orig_split"] = split
    frames.append(part)

frame = pd.concat(frames, ignore_index=True)
frame = frame.drop(columns=[c for c in frame.columns if c.startswith("__index")], errors="ignore")
# The error analysis joins predictions back to comment text by id, and this
# mirror ships without one.
if "id" not in frame.columns:
    frame.insert(0, "id", range(len(frame)))
frame.to_csv(raw_dir / "train.csv", index=False)
print(f"wrote {len(frame):,} rows to {raw_dir / 'train.csv'}")
PY
    ;;

  *)
    echo "ERROR: unknown source '${SOURCE}' (expected: kaggle | huggingface)" >&2
    exit 1
    ;;
esac

echo
echo "Next: confirm the identity annotations survived the load, and read the"
echo "subgroup support counts BEFORE booking any training time."
echo "  python scripts/run_all.py -c configs/a100.yaml --only eda"
echo
echo "Cite the licence and source in the report -- see README 'Getting the data'."
