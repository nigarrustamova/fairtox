# FairTox: demographic bias in toxicity detection

Measuring, and attempting to mitigate, unintended demographic bias in
Transformer-based toxicity classifiers on the **Jigsaw Unintended Bias in
Toxicity Classification** corpus.

**Track 2: Applied / domain research.** DLE-AI-202, AI Academy, National AI
Centre, Cohort I 2026.

---

## The research question

> Do toxicity classifiers produce disproportionately high **false-positive
> rates** on benign text that mentions particular demographic groups -- and can
> loss re-weighting reduce that disparity without costing too much overall
> performance, or too much safety?

A content-moderation classifier fails in two directions. A **false negative**
lets abuse through: a safety failure. A **false positive** silences a harmless
comment: a fairness failure. When benign sentences such as *"I am a proud Muslim
woman"* are flagged disproportionately often, the moderation system becomes a
tax on the communities it was meant to protect.

The mechanism is a shortcut. Identity terms co-occur with abuse in scraped
training text, so empirical risk minimisation learns to treat the identity token
itself as evidence of toxicity rather than learning hostility. The intervention
tested here upweights the cell that *breaks* that correlation -- non-toxic
comments which do mention an identity -- and measures what that costs.

> We do not claim the dataset proves human discrimination. We measure whether a
> **model** learns and amplifies subgroup error disparities from a data
> distribution. That is a narrower claim, and it is the one the evidence supports.

---

## Quick start

```bash
git clone https://github.com/nigarrustamova/fairtox.git
cd fairtox

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# No download, no GPU, ~3 minutes: proves the pipeline end to end.
python scripts/make_synthetic_data.py --rows 6000
bash run_all.sh configs/smoke.yaml
```

### One command reproduces the headline result

```bash
bash scripts/download_data.sh          # see "Getting the data"
bash run_all.sh configs/a100.yaml
```

That runs the eight MUST-tier stages in dependency order and writes every number
the report quotes to `results/main/`.

### On Windows

`run_all.sh` and `run_seeds.sh` are thin wrappers, so there is no PowerShell copy
to keep in sync. Either run them under the Git Bash that ships with Git for
Windows (`bash run_all.sh configs/smoke.yaml`), or call the driver directly --
this is the same command the wrapper issues:

```powershell
python scripts/make_synthetic_data.py --rows 6000
python scripts/run_all.py --config configs/smoke.yaml --tier must
```

The seed sweep is the loop in `scripts/run_seeds.sh`, one run per seed and then
the aggregation:

```powershell
foreach ($seed in 42, 43, 44) {
  $name = if ($seed -eq 42) { "main" } else { "main_s$seed" }
  python scripts/run_all.py --config configs/a100.yaml --tier must `
    --set "run.seed=$seed" --set "run.name=$name"
}
python scripts/run_all.py --config configs/a100.yaml `
  --only seed_robustness --no-deps `
  --set "run.name=main" --set "analysis.seed_run_glob=main*"
```

Training itself is booked on Linux (Kaggle/A100), so Windows is the smoke-test and
analysis path.

---

## How it is organised

The study is a set of **stages**. Each one reads the config, does a unit of work,
and writes artefacts. The driver topologically sorts whatever subset you ask for.

```bash
python scripts/run_all.py --list                  # what exists, and what trains
python scripts/run_all.py --tier must --dry-run   # the resolved plan, no work
```

| Tier | Stages |
| --- | --- |
| **MUST** (8) | `eda`, `tfidf_baseline`, `train_baseline`, `audit_baseline`, `train_mitigated`, `compare`, `ablate`, `error_analysis` |
| **OPTIONAL** (7) | `threshold_sensitivity`, `jigsaw_bias_metrics`, `attributions`, `mechanism_analysis`, `seed_robustness`, `postproc_thresholds`, `leakage_check` |
| **EXTRA** (2) | designed, then answered elsewhere -- see `src/fairtox/stages/_planned.py` |

Three stages train: `train_baseline`, `train_mitigated`, `ablate`. They are
marked `[GPU]` in `--list`, and the driver warns before it starts a plan that
includes one.

### Re-running analysis must never cost GPU time

Three independent guards:

```bash
# 1. --no-deps runs exactly what you name and pulls in nothing else.
python scripts/run_all.py -c configs/a100.yaml --only compare --no-deps

# 2. Even without it, a model whose checkpoint, predictions and metrics are all
#    on disk is skipped rather than retrained.
python scripts/run_all.py -c configs/a100.yaml --only compare   # seconds, not hours
```

```bash
# 3. But reuse is keyed on the settings, not the path. A rerun with a different
#    alpha, batch size or seed retrains instead of quietly keeping the old model.
python scripts/run_all.py -c configs/a100.yaml --only train_mitigated --set mitigation.alpha=2
#    -> "exists but was trained under different settings (...) -- retraining"
```

Guard 3 matters more than it looks: artefact paths are keyed on the model name,
and the headline pair'"'"'s names are fixed. Without it, training at alpha=4, watching the
model collapse, and re-running at alpha=2 would find the old files, skip, and report
the collapsed alpha=4 model as the alpha=2 result. The fingerprint deliberately covers
only what determines the weights -- overriding `data.raw_dir` (which the Kaggle
notebook does on every cell) or the log level does **not** trigger a retrain.

Override guard 2 and 3 with `--force` when you genuinely want to retrain.

### Seeds: the headline result is replicated, not run once

```bash
bash scripts/run_seeds.sh 42 43 44 45 46 47 48 49    # the headline family
```

Each seed re-partitions the corpus, re-shuffles and re-initialises the head, so
the runs are independent replications of the whole experiment -- closer to
repeated random subsampling than to re-running one fixed split. Within a seed the
two arms still share everything but the loss weighting, so each paired difference
stays a controlled comparison.

`seed_robustness` then aggregates the **per-seed differences** and reports mean,
spread, range, how many seeds agreed in sign, and whether every seed landed on
the same pre-registered outcome. It says so out loud when the mean is smaller
than its own spread.

**The stage itself never runs a significance test**, and that is deliberate: it
is written to be correct at three seeds, where a p-value would claim more than
the sample supports. The headline family reached **eight**, so the cross-family
analysis in `src/fairtox/evaluation/aggregate.py` does compute one -- a paired
Student-t interval and an exact binomial sign test, printed in
`analysis/study01_families.md`. That table prints `sign_p` for the three-seed
families too, where it can only ever be 0.25, precisely so that 3/3 is never
mistaken for evidence.

---

## Method

**Data.** Only the identity-annotated subset (~448k rows). A row with no identity
annotation is *unrated*, not *identity-free*; treating those two as the same
would push genuinely identity-bearing comments into the background weight and
corrupt the intervention.

**Splits.** 75 / 10 / 15, stratified on `label x any_identity`, fixed before any
statistic is fitted, and fingerprinted. The 15% test split is deliberate: the
reporting floor is counted on the test split, so a larger one buys more
subgroups the paper can make a claim about, at about 6% of training throughput.

**Model.** `distilbert-base-uncased`, `[CLS] -> dropout -> Linear(768, 1)`. The
loss is ours rather than `AutoModelForSequenceClassification`'s, because every
intervention in this study lives in the loss.

**The intervention.** Per-sample weights on BCE, normalised by the **sum of
weights** rather than the batch size -- otherwise a larger alpha would inflate the
gradient magnitude, act partly as a larger learning rate, and the ablation would
be measuring that confound instead of the weighting scheme.

| Scheme | Weight vector | Role |
| --- | --- | --- |
| `none` | all 1.0 | the baseline |
| `class` | inverse class frequency | control: *would ordinary imbalance handling have done this?* |
| `subgroup` | alpha on non-toxic identity-bearing rows | ours |

**The control is structural.** `train_baseline` and `train_mitigated` are the
same function reading the same config file. They cannot differ in architecture,
learning rate, seed or split, because there is nowhere for them to differ. A
reviewer checks the control by reading one function, not by diffing two configs.

**Identity labels never reach the model.** They are used to compute training
weights and to slice the evaluation. At inference the model sees text only.

---

## What is measured

| | Metric | Why |
| --- | --- | --- |
| **Utility** | macro-F1, ROC-AUC, PR-AUC | ~90% of comments are non-toxic, so accuracy is reported and never acted on |
| **Parity** | subgroup FPR, FPR gap, EOD, FPR variance | the wrongful-censorship rate, and how unevenly it falls |
| **Safety** | subgroup FNR, FNR gap | closing the FPR gap by missing more abuse is not a win |
| **Benchmark** | Subgroup AUC, BPSN, BNSP (Borkan et al., 2019) | threshold-free, and comparable to published work |

Three guards keep those numbers honest:

- **Separate support floors.** FPR is estimated on a subgroup's non-toxic rows
  and FNR on its toxic rows, and here those counts differ by an order of
  magnitude. Each rate carries its own floor and its own reportable flag; below
  it a rate is still measured and shown, but excluded from the aggregates.
- **Wilson intervals.** At zero observed false positives a percentile bootstrap
  reports `[0, 0]` whether the subgroup has 120 rows or 12,000. Wilson still
  widens as the sample shrinks, which is the honest behaviour for the case a
  fairness paper is most likely to overstate.
- **Collapse detection.** A model that flags nothing has an FPR of zero in every
  subgroup and therefore *perfect* parity. Read without this check, a dead model
  is the fairest model in the study. It is flagged, excluded from the verdict,
  and marked on the ablation figure.

### Why the gap moved, not just that it moved

`mechanism_analysis` answers the three questions a reviewer asks after the
headline table, from saved predictions and with no GPU:

- **Is the disparity a learned shortcut?** It fits subgroup FPR against the
  subgroup's toxic rate *on the test split*, and reports the slope, the
  intercept and the correlation for both arms. A steep, tight line is the
  shortcut measured rather than asserted; the change in slope is what the
  intervention actually did to it.
- **Did the gap narrow only because the model flags less?** Each of the
  mitigated model's operating points is compared against the baseline's gap **at
  the same global FPR**, interpolated from the threshold sweep. Points outside
  the baseline's measured range are reported with `comparable: false` and left
  out of the summary rather than silently clamped — `np.interp` holds its
  endpoint value, which would dress an extrapolation up as a measurement at
  exactly the end where a conservative mitigated model lands.
- **Did the gap close from the top or the bottom?** `max - min` falls both when
  the worst-off subgroup improves (*levelling up*) and when the best-off one
  deteriorates (*levelling down*). Only the first is a fairness gain, so the
  direction is named in the output instead of being left for the reader to
  reconstruct.

### Pre-registered outcomes

The verdict thresholds live in `configs/default.yaml`, in a file that predates
the results.

| | Outcome |
| --- | --- |
| **A** | FPR gap falls, macro-F1 cost within tolerance |
| **B** | FPR gap falls, aggregate performance falls with it |
| **C** | FPR gap falls, but subgroup FNR rises -- a safety regression |
| **D** | null or adverse: weighting did not reduce the gap |
| **INVALID** | a model collapsed; its parity numbers are an artefact |

All of A-D are publishable findings if the evaluation is controlled. Only
INVALID is a failure, and it is a failure to be fixed rather than reported.

---

## Getting the data

The project needs the **demographic identity annotations**, not just toxicity
scores.

Download `all_data.csv.zip` (~326 MB) from the competition's Data tab and unzip
it into `data/raw/`:

<https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data>

Or scripted:

```bash
bash scripts/download_data.sh kaggle          # needs the rules accepted once
bash scripts/download_data.sh huggingface     # smaller mirror, development only
```

**Licence: CC0 1.0 (public domain).** Cite the dataset and its licence in the
report.

> Hugging Face `google/civil_comments` is **not** a substitute. It ships toxicity
> subtypes only; its `identity_attack` column is a *kind of toxicity*, not a
> record that a comment mentions a group, so the subgroup audit would have
> nothing to slice on.

Data is git-ignored. Nothing under `data/` is ever committed.

**Before booking any training time**, read the subgroup support counts:

```bash
python scripts/run_all.py -c configs/a100.yaml --only eda
```

If too few subgroups clear the floor, fix it **then** -- not after seeing results,
which would be tuning on the outcome.

---

## Configuration

| File | Purpose |
| --- | --- |
| `configs/default.yaml` | the study: data, model, training, mitigation, evaluation |
| `configs/a100.yaml` | the booked window: batch 64, scaled LR, 20 GB allocation |
| `configs/smoke.yaml` | synthetic data, tiny backbone, CPU, ~3 minutes |

Each experiment is one more file that `extends: a100.yaml` and changes one thing.
The eight families reported in the paper are these, run over the seeds listed in
`scripts/summarise_study.py`:

| File | The one thing it changes | Result |
| --- | --- | --- |
| `configs/a100.yaml` | -- (the headline: `subgroup` weighting, alpha 4) | gap -0.0520, 8/8 seeds |
| `configs/a100_lastep.yaml` | `select_checkpoint: last` | the checkpoint control |
| `configs/a100_wtox.yaml` | `identity_toxic` raised to 2.0 | half the safety cost, most of the parity gain lost |
| `configs/a100_drob.yaml` | `backbone: distilroberta-base` | the effect is not backbone-specific |
| `configs/a100_cda.yaml` | counterfactual term swap, p = 0.5 | null |
| `configs/a100_cda_full.yaml` | the same at p = 1.0 | null, and no dose-response |
| `configs/a100_dro.yaml` | `objective: group_dro` (learned weights) | the gap gets 2.6x worse |
| `configs/a100_heldout.yaml` | weighting sees gender and religion only | transfers to unseen axes at 58% |

**Variants are FILES, never a per-arm `--set`:** an override applied to one arm
gives the two arms different training fingerprints, and `compare` refuses the
pair. Every one is run the same way as the headline study:

```bash
CONFIG=configs/a100_drob.yaml BASE_NAME=drob bash scripts/run_seeds.sh 42 43 44
```

Three of them (`cda`, `dro`, `heldout`) must be run with `--only`, not the full
MUST tier: `ablate` sweeps alpha, and alpha means nothing for an augmentation
probability or a learned weight vector. Each config's header says so.

Override anything from the command line; overrides are recorded in the config
fingerprint saved with every result, so they are not invisible after the fact.

```bash
python scripts/run_all.py -c configs/a100.yaml \
  --set training.batch_size=32 --set training.gradient_accumulation_steps=2
```

### Compute envelope

The brief allocates **~2 days** of scheduled training. It states ~10-12 GB of GPU
memory per team; the teaching assistants later raised the actual allocation to
**20 GB**, which is what `training.memory_budget_gb` records. Exceeding the
allocation is an automatic deduction.

Measured on the booked MIG 3g.20gb slice, DistilBERT at these settings peaked at
**3.99 GB reserved of 19.5 GB** -- a fifth of the allocation, and comfortably less
than the 5-7 GB estimated before the run. The trainer records allocated **and**
reserved peaks (reserved is what `nvidia-smi` shows and what other teams cannot
use), warns at 85% of the budget, and warns harder above it.

**The headroom is deliberately not spent on a larger batch.** At batch 64 this
model is already compute-bound on an A100, so 64 -> 128 is worth minutes on a MUST
tier of roughly two GPU-hours -- while halving the optimizer steps (5,250 per
epoch -> 2,625), which is a real risk to the very comparison the paper rests on,
and pushing the alpha sweep further toward collapse. Compute is not this
project's bottleneck. The spare budget goes on repeating the headline pair under
extra seeds; `configs/a100.yaml` documents how to measure a larger batch if you
want to revisit the decision.

The MUST tier is **six** training runs, not eight: the ablation reuses the
baseline for any arm whose weights come out uniform, which `subgroup` at alpha = 1.0
and the `none` scheme both do.

---

## Reproducibility

- Seeds fixed for Python, NumPy and Torch; the driver reseeds per stage, so a
  stage's result does not depend on which stages ran before it.
- Every run writes an `environment.json`: versions, GPU, CUDA, git commit,
  config fingerprint. `scripts/freeze_env.py` pins the exact package set.
- Splits **and training settings** are fingerprinted, and `compare` **refuses**
  to report two arms whose fingerprints differ. Driving both arms from one config
  makes them identical by construction -- but the runbook launches them as two
  commands, and a command line takes `--set`. A `--set training.batch_size=128`
  on one arm only would make the comparison measure batch size as much as it
  measures the intervention, and the output would look entirely normal. Differing
  GPUs warn rather than fail.
- Tables are written as Markdown *and* CSV, so the report reads numbers back from
  disk instead of having them retyped.
- CI runs lint, tests and the full MUST tier on generated data on every push.

```bash
python -m pytest tests/ -q          # 379 tests
python -m ruff check .              # the whole repo, notebook included
```

---

## Layout

```
configs/            default.yaml, smoke.yaml, a100.yaml + the seven experiment files
data/               raw/ and processed/ (git-ignored)
scripts/            run_all.py, run_seeds.sh, download_data.sh, make_synthetic_data.py,
                    freeze_env.py, summarise_study.py, rebuild_figures.py,
                    make_paper_figures.py, make_paper_tables.py
analysis/           committed outputs: the study tables, the run digest, the error coding
results/            every run's metrics, tables and figures (predictions excluded)
src/fairtox/
  config.py         dotted config with an `extends` chain
  registry.py       stage registration, tiers, dependency ordering
  pipeline.py       the driver and the run context
  provenance.py     comparability guards for the controlled comparison
  data/             load.py, split.py, dataset.py
  models/           transformer.py
  training/         losses.py, trainer.py
  evaluation/       metrics.py, predictions.py
  fairness/         auditor.py, jigsaw_metrics.py
  explain/          attributions.py
  stages/           one module per experiment
  utils/            io, logging, seed, device, plotting
tests/              379 tests
notebooks/          kaggle_runner.ipynb (backup compute path)
report/             report.tex + report.pdf (IEEE two-column), and the figures/
                    and tables/ it inputs -- both generated, never typed:
                      python scripts/make_paper_figures.py
                      python scripts/make_paper_tables.py
                      cd report && latexmk -pdf report.tex
presentation/       presentation.pdf -- written after the results exist
run_all.sh          the one-command reproduction
TRAINING_DAY.md     the booked-window runbook
```

`report/` and `presentation/` are intentionally empty. Both are written from real
numbers once training has finished; drafting them earlier would mean inventing
results.

---

## Ethics and limitations

- **Annotator subjectivity.** Toxicity labels are aggregated human judgements,
  not ground truth. Perceptions of slang, reclaimed slurs and political speech
  vary by annotator background.
- **Coarse identity categories.** The dataset's buckets are broad and binary;
  real identity is not, and intersectional groups are too small here to support
  separate claims.
- **No conversational context.** Comments arrive stripped of thread and
  community norms, and much sarcasm cannot be resolved without them.
- **Goodhart's law.** Optimising FPR parity alone does not make a moderation
  system fair or safe. That is exactly why the safety axis is in the headline
  table rather than an appendix.
- **Attribution is not causation.** Integrated Gradients measures local input
  sensitivity. High attribution on an identity token is not counterfactual proof
  and not a statement about the model's reasoning in general.
- **Responsible use.** Demographic annotations are used here strictly to audit
  and debias a safety model. They must never be used to profile users.

---

## Tools

AI assistance was used in developing this codebase, and is disclosed in the
report's *Tools & Acknowledgements* section as the brief requires. Every team
member is responsible for understanding and defending any part of the system.
