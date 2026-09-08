# Training-day runbook

The booked A100 window is the one resource that cannot be recovered. Everything
here exists so the window is spent training rather than debugging.

**Read this before the window opens, not during it.**

---

## The day before

### 1. Prove the pipeline on synthetic data -- no GPU, no download

```bash
python scripts/make_synthetic_data.py --rows 6000
bash run_all.sh configs/smoke.yaml
```

All eight MUST stages must finish green. Anything that fails here fails on the
A100 too, only far more expensively.

### 2. Run the checks

```bash
python -m pytest tests/ -q
python -m ruff check src/ scripts/ tests/
python scripts/run_all.py --list
```

### 3. Get the real data onto the machine and profile it

```bash
bash scripts/download_data.sh
python scripts/run_all.py -c configs/a100.yaml --only eda
```

Then **read `results/main/metrics/eda.json`**. Before training you need to know:

- how many rows survive the annotated-only filter;
- the toxic rate;
- **how many subgroups clear the support floor on the test split** -- that is the
  number of groups the paper can make a claim about.

If the stage warns that too few subgroups qualify, fix it **now**: use the full
corpus, raise `data.splits.test`, or lower `evaluation.min_support` and justify
it in the paper. Changing it after seeing results is tuning on the outcome.

### 4. Time one epoch, so the window is booked against a real number

```bash
python scripts/run_all.py -c configs/a100.yaml --only train_baseline \
  --set data.subsample=20000 --set training.num_epochs=1 --set run.name=timing
```

Multiply by `(full rows / 20000) x epochs`, then double it. That is the estimate
for the headline pair. Note `run.name=timing`, so this throwaway run cannot
overwrite anything real.

---

## In the window

### The whole study is one command

```bash
bash run_all.sh configs/a100.yaml
```

Stages are ordered so cheap work runs first: a data problem surfaces in the
seconds `eda` takes, not an hour into fine-tuning. The plan is printed before
anything starts, with every training stage marked `[GPU]`.

If you would rather drive it by hand, the order matters -- the headline pair
first, back to back, so a window that dies still leaves the paper's central
result:

```bash
python scripts/run_all.py -c configs/a100.yaml --only train_baseline
python scripts/run_all.py -c configs/a100.yaml --only train_mitigated
python scripts/run_all.py -c configs/a100.yaml --only compare --no-deps
python scripts/run_all.py -c configs/a100.yaml --only ablate --no-deps
```

> **`--no-deps` is not optional in those last two lines.** Without it the driver
> resolves `compare`'s dependencies and queues both training stages. They would
> be skipped as already-complete -- but rely on the flag, not on the safety net.

### Watch the first 100 steps

If the loss is not falling, kill it. A run that is not learning at step 100 will
not be learning at step 10,000, and you will have spent the window finding out.

### Check peak VRAM after the first run

```bash
python -c "import json; d=json.load(open('results/main/metrics/model_baseline.json')); print(d['training']['peak_allocated_gb'], 'allocated /', d['training']['peak_reserved_gb'], 'reserved GB')"
```

Against the **20 GB** allocation the TAs announced (the brief's own figure was
10-12 GB; `training.memory_budget_gb` records the real one). **Reserved** is the
number that matters: it is what `nvidia-smi` shows and what another team on the
same card cannot use. The trainer warns at 85% of the budget on its own.

Expect roughly 5-7 GB; the booked run came in lower still, at **3.99 GB reserved
of 19.5 GB** on the MIG 3g.20gb slice (the peak over all 115 models; the
headline run itself peaked at 3.90). That leaves four fifths of the allocation
unused, and that is intended -- see the long comment at the top of
`configs/a100.yaml` for why the headroom is not spent on a larger batch. Short
version: it buys minutes, halves the optimizer steps, and raises the collapse
risk in the alpha sweep.

Measured throughput on that slice: **0.064-0.071 s/step** at batch 64 and
sequence length 128 with AMP on, so one model is about 18 minutes (15,750 steps)
and a full MUST tier about 1 h 55 min for its six trained models.

If the peak does come out close to the ceiling, halve the batch and double
accumulation -- identical effective batch, smaller footprint. **Change it in the
config, not with a per-arm `--set`:**

```bash
--set training.batch_size=32 --set training.gradient_accumulation_steps=2
```

> Applying a training override to one arm and not the other breaks the controlled
> comparison. `compare` now refuses such a pair outright and names the setting
> that differs, so this cannot pass silently -- but it still costs you the runs.

### Watch for `DEGENERATE MODEL`

Heavy upweighting can push the model to predict "non-toxic" for everything. Such
a model has a false-positive rate of zero in *every* subgroup, and therefore a
**perfect** FPR gap, FPR variance and EOD. It will look like the best result in
the study.

The pipeline detects it, logs a warning, flags the row in the ablation table,
marks it on the trade-off figure, and makes `compare` return `INVALID` instead of
an outcome letter. If you see it: lower `mitigation.alpha`, train longer, re-run
that arm. **Never report a collapsed arm's parity numbers.**

The high end of the alpha sweep is where it is expected to appear. That is a
finding -- "the gain stops being free above alpha ~ N" -- not a failure.

### If the window ends mid-training

Do nothing special. `last.pt` carries model, optimizer, scheduler, scaler **and
the run history**, so re-running the same command resumes from the next epoch and
the training curve in the paper is the whole curve. `last.pt` is deleted once a
run completes cleanly.

---

## Guardrails

**Never split the headline pair across machines.** The paper's claim is a
controlled comparison in which only the loss weighting varies. Running the
baseline on a Colab T4 and the mitigated arm on the A100 puts a hardware
confound inside exactly that comparison, and there is no honest way to write
around it afterwards. The same applies to the ablation arms.

**Never tune on test.** Everything is selected on validation. The test split is
touched once, for the final numbers.

**Do not change the config to make results look better.** If a threshold or a
support floor needs changing, change it before seeing outcomes and say why in
the paper.

**Safe to run anywhere, any time** -- these read saved predictions, not models:
`eda`, `tfidf_baseline`, `audit_baseline`, `compare`, `error_analysis`,
`threshold_sensitivity`, `jigsaw_bias_metrics`, `seed_robustness`.

---

## Seeds: do this, it is the criticism from last time

```bash
bash scripts/run_seeds.sh 42 43 44
```

One command. It runs the full MUST tier under each seed (seed 42 into
`results/main`, the rest into `results/main_s<seed>`), then aggregates without a
GPU. Runs already on disk under identical settings are skipped, so an interrupted
sweep restarts with the same command.

**Cost:** roughly six training runs per seed, so about three times the single-seed
tier. On a two-day window with a ~2 GPU-hour tier that is comfortably affordable,
and it is the single highest-value use of the spare budget.

**Why it answers the criticism.** `run.seed` drives the split, the shuffling and
the head initialisation, so each seed is an independent replication of the whole
experiment on a fresh partition. Within a seed the two arms still differ only in
the loss weighting, so every paired difference is still controlled.

**What to read afterwards:**

```
results/main/tables/tab09_seed_robustness.md    per-seed deltas, one row per seed
results/main/metrics/seed_robustness.json       mean, sd, range, sign agreement
```

The stage reports the **per-seed difference** as the unit of analysis, plus how
many seeds agreed in sign and whether they all landed on the same outcome letter.
It does not run a significance test: with three or four seeds a p-value would
claim more than the sample size supports. Once a family reaches eight seeds a
test IS honest, and `scripts/summarise_study.py` computes it -- a paired t
interval and an exact sign test, side by side, because they can disagree.

It will tell you plainly when the effect is noise:

```
d_fpr_gap    mean -0.0198  sd 0.1086  min -0.0909  max 0.1053  same sign 2/3
d_macro_fpr  mean -0.0306  sd 0.0158  min -0.0486  max -0.0193 same sign 3/3
WARNING seeds disagree on the outcome (C, D). Report the disagreement.
WARNING the mean FPR-gap change is SMALLER than its spread. Report it as
        inconclusive, not as an effect.
```

That is a real run of this pipeline -- an **early one, on a 20k subsample**, kept
here because it shows what a null looks like, not because it is the study's
result. A single seed had reported the FPR gap falling by 0.074, and across three
seeds it was indistinguishable from noise. **Write up what survives replication,
not what one seed showed.** Reporting the disagreement is a finding; picking the
seed you prefer is not.

The full-corpus run behaved differently, and section "What actually happened" at
the end of this file records it. The lesson stands either way: the reason we know
the final effect is real is that eight seeds agreed, not that the first one did.

---

## After the window

No GPU needed:

```bash
python scripts/run_all.py -c configs/a100.yaml --tier optional --no-deps \
  --only threshold_sensitivity jigsaw_bias_metrics attributions --keep-going
python scripts/freeze_env.py
```

`threshold_sensitivity` asks whether the disparity survives a different threshold.
`jigsaw_bias_metrics` gives Subgroup/BPSN/BNSP AUCs, which make the numbers
comparable to published work. `attributions` needs a checkpoint but no training.

Three more stages cost no GPU and were added after the first window closed:
`postproc_thresholds` (per-subgroup decision thresholds, fitted on validation),
`leakage_check` (test rows that occur verbatim in train) and `mechanism_analysis`
(the shortcut regression). Then, once every family is finished:

```bash
python scripts/summarise_study.py --write-digest analysis/study_digest.json.gz
```

That writes the cross-family tables the report quotes into `analysis/`, and
archives every run's metrics into one 400 KB file so the analysis stays runnable
after the machine is gone.

### Record while it is fresh

`results/main/metrics/environment.json` and each `model_*.json` already hold the
GPU, driver, torch version, config fingerprint, peak VRAM and wall-clock. What
they cannot capture, and you should write down the same day:

- which runs happened in which session, on which hardware;
- anything that went wrong and what you did about it.

The second one matters most. *"We hit an OOM at batch 64, dropped to 32 with
accumulation 2, and re-ran both arms"* is a sentence examiners respect.
Reconstructing it from memory a week later is how honest reporting quietly
becomes fiction.

---

## If everything slips

There is always a complete result to write up:

```bash
python scripts/run_all.py -c configs/a100.yaml --tier must \
  --skip ablate --set data.subsample=100000 --set run.name=reduced
```

**Understand what that costs before using it.** Subsampling shrinks the test
split, and the reporting floor is counted there -- so the subgroups that drop out
first are the small ones the study is actually about. Report the subsample size,
the reduced subgroup coverage, and why. A smaller-N result with honest intervals
beats a missing one; a result that quietly dropped half its subgroups does not.


---

## What actually happened

The plan above was written before the window opened. This is the tally after it
closed, so the two can be compared honestly.

| | planned | actual |
| --- | --- | --- |
| seeds on the headline config | 3 | **8** |
| experiment families | 1 + ablation | **8** |
| trained runs | ~3 | **38** (115 models) |
| peak reserved VRAM | 5-7 GB expected | **3.99 GB of 19.5** |
| throughput | unknown | **0.064-0.071 s/step**, ~18 min per model |
| collapsed models | expected at high alpha | **zero**, at any alpha up to 10 |

Nothing in the recipe changed: batch 64, 3 epochs, lr 3e-5, AMP on, one MIG
slice, from the first run to the last. Every seed's runs share that seed's split
fingerprint, and every model in the study trained on the same GPU.

The headline result is a **-0.0520 FPR-gap change [-0.0742, -0.0297], 8/8 seeds,
sign test p = 0.0078**, at a cost of +0.1138 macro FNR. The three-seed estimate
had been -0.0644; it was inflated by one outlier seed, and that outlier turned out
to be a checkpoint-selection artefact rather than a property of the mitigation.
Both numbers are in `analysis/study01_families.md`, and the control that
diagnosed it is the `lastep` family.

**The single most useful thing the extra seeds bought was not a smaller error
bar.** It was finding out that the first number was wrong.
