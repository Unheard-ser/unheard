# Contributing

Five people are running experiments against one dataset and one results log.
Everything below exists so that five sets of numbers are comparable to each
other. Read `CLAUDE.md` for the full project rules; this file is the
day-to-day mechanics.

New to the repo? Run through `notebooks/00_project_walkthrough.ipynb` first.
It is the guided tour and takes about twenty minutes.

---

## 1. Clone and set up

```bash
git clone https://github.com/unheard-ser/unheard.git
cd unheard
```

Python **3.14** — the pins in `requirements.txt` are the versions actually
resolved and verified on 3.14, and feature values must match across all five
machines. Do not upgrade a pin without telling everyone.

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.14 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Check it worked:

```bash
python -c "from ser.splits import load_folds; print(len(load_folds()), 'fold rows')"
```

That should print `19440 fold rows` and needs no audio.

---

## 2. Get the data

**Audio is never committed.** `data/` is gitignored and stays that way
(`CLAUDE.md` rule 7). The 1,440 clips come from the team's shared copy — ask
in the team channel for the link, then unpack it so the tree looks exactly
like this:

```
data/
  data/                 <- note the doubled "data", this is not a typo
    Person01/
      01-01-01-01-01.wav
      ...               <- 60 clips
    Person02/
    ...
    Person24/
```

Two things about this copy that will bite you if you assume otherwise:

- Folders are `PersonNN`, **not** the canonical RAVDESS `Actor_NN`.
- Filenames have **5** hyphen-separated fields
  (`emotion-intensity-statement-repetition-actor`), not the canonical 7. The
  parser rejects 7-field names rather than mis-parsing them, so a fresh
  download from the canonical RAVDESS distribution will **not** drop in
  without renaming.

Verify your copy before you trust any number that comes out of it:

```bash
pytest tests/test_metadata.py tests/test_audit.py
```

Those encode the dataset facts — 1,440 files, 96 neutral, 24 actors, 48 kHz,
the five known dual-channel clips. If one fails, your copy differs from
everyone else's and nothing downstream is comparable.

`features/*.parquet` is also gitignored. It is regenerable and large; let
each machine build its own cache.

---

## 3. Branch per person

`main` is protected by CI and is never committed to directly.

```bash
git switch main
git pull
git switch -c <yourname>/<what-youre-doing>
```

Examples: `sahitya/cnn-mel-spectrogram`, `priya/feature-ablation`,
`arjun/wav2vec-probe`.

Work on your branch, commit as you go, then:

```bash
git push -u origin <yourname>/<what-youre-doing>
```

Open a pull request. The template asks three questions — what changed, which
protocol, and whether `results.csv` was updated. Answer all three; they are
the questions the graded report will need answered anyway.

One teammate reviews. CI must be green. Then squash-merge and delete the
branch.

**Rebase rather than merge** when `main` moves under you
(`git pull --rebase origin main`), so the history stays readable for the
write-up.

### results.csv will conflict, and that is fine

Everyone appends to `results/results.csv`, so two branches touching it will
conflict on merge. The resolution is always the same: **keep both sets of
rows.** Never drop someone else's row to make a conflict go away. Each row
carries its own `run_id` and `owner`, so order does not matter.

---

## 4. The frozen-fold contract

This is the one rule that makes five people's results mean anything.

**`splits/folds.csv` is frozen.** It was generated once, on 2026-08-22, and
committed. It is never regenerated, never hand-edited, never reordered.
`write_folds()` raises on an existing file, and a test guards it. Every
experiment by every team member uses those exact fold assignments — that is
what makes your macro-F1 and mine comparable at all.

**`ser/evaluate.py` is the single scoring function.** Everyone calls it;
nobody writes their own accuracy loop. If the metric definition changes
mid-project, every number logged before the change becomes incomparable to
every number after it, and the results log is worthless.

So:

- Do not edit `splits/folds.csv`. Ever.
- Do not edit `ser/evaluate.py` to change what a metric means.

If you genuinely believe a split is wrong, **stop and raise it with the team**
before touching anything. A bad split we all share is recoverable; a split
that changed halfway through is not.

Consume the folds, do not rebuild them:

```python
from ser.splits import get_split
train_files, test_files = get_split("speaker_independent", fold=0)
```

### The rules that go with it

- **Never fit anything on test data.** Scalers, PCA, feature selection, class
  weights, per-speaker statistics — fit on the training fold, apply to the
  held-out fold.
- **Augmentation applies to training folds only.**
- **`speaker_independent` is the primary protocol.** `random_stratified` is
  run only as a deliberate comparison to quantify leakage. Never report the
  random-split number on its own.
- **Every run appends a row to `results/results.csv`.** A run that is not
  logged did not happen.
- **Seeds are fixed at 42** — numpy, sklearn, torch, and `PYTHONHASHSEED`.
- **Report accuracy and macro-F1 together**, always with the confusion matrix
  and the gender / intensity / emotion slices. Accuracy alone hides failure
  on neutral, which has half the clips of every other class.

---

## 5. Where code goes

- All logic lives in `ser/` as importable modules. **Notebooks import from
  `ser/`; they never define functions.**
- Type hints on public functions. Docstrings state what is fit on train only.
- Plotting functions return a `Figure`. The caller saves it — that keeps them
  testable.
- Feature extraction is parameterised, not hardcoded. Aggregation,
  normalisation and MFCC count are arguments with defaults, because comparing
  them is part of the experiment.
- **Never call `librosa.load` directly.** Use `ser.preprocess.load_audio`. It
  is the one entry point applying the mono downmix and the 16 kHz target
  rate. `librosa.load` silently resamples to 22 050 Hz.
- Anything with an invariant — counts, fold disjointness, no-leakage — gets a
  pytest test.

---

## 6. Tests

```bash
pytest
```

The full suite needs the audio present. CI does not have the audio, so it
rebuilds a filename-only stand-in from `splits/folds.csv` and runs everything
except the two files that decode real waveforms
(`tests/test_preprocess.py`, `tests/test_audit.py`). That still covers all
the fold-contract, no-leakage and metric invariants. Run the full suite
locally before you open a PR.

Slow tests that decode all 1,440 clips are marked:

```bash
pytest -m "not slow"
```

---

## 7. What good work looks like

The deliverable is a graded report, not a leaderboard entry. A negative
result that is correctly measured is worth more than a high number that is
not reproducible. When your model underperforms, say so in the PR and
investigate why — do not tune until it looks good.
