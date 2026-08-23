# Unheard — Speech Emotion Recognition on RAVDESS

ISB AMPBA foundation project. Multi-class speech emotion recognition over
1,440 RAVDESS clips, 24 actors, 8 emotion classes.

**Business framing:** contact-centre and telecaller QA — scoring calls for
customer sentiment and predicting CSAT from the audio rather than from a
post-call survey nobody fills in.

**Deliverable:** a graded report. A negative result that is correctly
measured is worth more than a high number that is not reproducible.

| | |
| --- | --- |
| Mid review | 6 Sep 2026 |
| Final review | 4 Oct 2026 |
| Team | 5 |

---

## Where things stand

Classical baselines on MFCC-40 + Δ + ΔΔ, mean/std aggregated, defaults only,
no tuning — pooled out-of-fold:

| protocol | model | accuracy | macro-F1 |
| --- | --- | --- | --- |
| **speaker_independent** | svm-rbf | **0.515** | **0.498** |
| speaker_independent | random-forest | 0.467 | 0.423 |
| random_stratified | svm-rbf | 0.652 | 0.636 |
| random_stratified | random-forest | 0.585 | 0.554 |

The gap between the two protocols is the point, not a nuisance. Random
stratified splits let the same 24 voices appear on both sides, and it buys
~14 accuracy points that would not survive contact with an unseen speaker.
**Speaker-independent is the primary protocol.** The random-split number is
never reported on its own.

Full log: `results/results.csv`. Split definitions: `docs/splits.md`.

---

## Start here

`notebooks/00_project_walkthrough.ipynb` — the guided tour. Twenty minutes,
end to end: the data, the splits, the features, the baseline, the scoring.

Setting up your machine, getting the audio, the branch workflow and the
frozen-fold contract are all in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read
it before your first PR.

Quick version:

```bash
python3.14 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The audio is **not** in this repo and never will be — `data/` is gitignored.
Get the team's copy and unpack it to `data/data/PersonNN/`; CONTRIBUTING has
the details and the verification command.

---

## Run the baseline

```bash
python -m ser.run_baselines
```

Drives the default feature config through every (model, protocol, fold),
scores each through `ser.evaluate`, and appends every run to
`results/results.csv`. First run extracts features for all 1,440 clips and
caches them to `features/`; later runs reuse the cache.

---

## Add your own experiment

Swap the model, keep everything else. That is the whole contract — same
folds, same scorer, so your number is comparable to everyone else's.

```python
from ser.features import FeatureConfig, extract, feature_columns
from ser.metadata import load
from ser.splits import get_split
from ser.evaluate import evaluate_pooled

meta = load()
config = FeatureConfig()                      # or vary it -- that IS the experiment
features = extract(config, meta).set_index("filename")
cols = feature_columns(features.reset_index(), meta)
X, y = features[cols], features["emotion_label"]

per_fold = []
for fold in range(5):
    train_files, test_files = get_split("speaker_independent", fold)
    model = YourModel()                        # <- the only line that is yours
    model.fit(X.loc[train_files].to_numpy(), y.loc[train_files].to_numpy())
    predicted = model.predict(X.loc[test_files].to_numpy())
    per_fold.append(
        (y.loc[test_files].to_numpy(), predicted, features.loc[test_files].reset_index())
    )

result = evaluate_pooled(
    per_fold, "speaker_independent",
    model="your-model-name", feature_config=config.name,
    notes="what you were testing",
)
print(result["accuracy"], result["macro_f1"])
print(result["confusion_matrix"])
```

`evaluate_pooled` appends the row to `results/results.csv` for you and hands
back the confusion matrix and the gender / intensity / emotion slices. That
append is the point — **a run that is not logged did not happen.**

Pass `append=False` when you are reproducing something rather than claiming a
new result.

Anything fit on data — scalers, PCA, selection, class weights — goes inside
the fold loop, fit on `train_files` only. Never on the held-out fold.

---

## Layout

```
ser/
  metadata.py      the clip inventory -- parses filenames, never opens audio
  preprocess.py    load_audio() -- THE single audio entry point: mono, 16 kHz
  audit.py         integrity checks on the raw audio
  splits.py        4 protocols, FROZEN in splits/folds.csv
  features.py      parameterised feature extraction
  evaluate.py      the single scoring function everyone calls
  models/          one module per model family
  run_baselines.py the classical floor
notebooks/         exploration and figures only -- no logic lives here
splits/folds.csv   committed, frozen 2026-08-22, never regenerated
results/results.csv appended by evaluate.py
docs/              data_audit.md, splits.md
figures/           saved plots for the deck
```

`CLAUDE.md` holds the full project rules and the verified dataset facts.
`PLAN.md` is the phase checklist.

---

## Tests

```bash
pytest
```

CI runs on every push and every pull request. It has no audio, so it rebuilds
a filename-only stand-in from `splits/folds.csv` and runs everything except
the two files that decode real waveforms — 70 tests, including every
fold-disjointness, no-leakage and metric invariant. The full suite needs the
clips present; run it locally before opening a PR.
