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

Pooled out-of-fold on the frozen `speaker_independent` folds — the primary
protocol, and the only one reported on its own:

| approach | accuracy | macro-F1 |
| --- | --- | --- |
| **wav2vec2 L6 + per-speaker norm + linear head** | **0.799** | **0.794** |
| handcrafted MFCC-20 + extras, svm-rbf (the floor) | 0.639 | 0.631 |
| handcrafted, + all 3 external corpora pooled in | 0.524 | 0.520 |
| first MFCC-40 baseline | 0.515 | 0.498 |

Three results the project rests on:

**Frozen pretrained embeddings beat hand-designed features by 16.3 pp**, using a
*linear* head and no feature engineering — more than the entire Phase 4 feature
programme bought (+13.3 pp). Emotion lives in the **middle** of wav2vec2: layer
6 beats the final layer by 17.6 pp, and layers 11–12 fall below the handcrafted
floor entirely. See `docs/wav2vec2_findings.md`.

**Merging three more corpora does not help, and that is measured, not assumed.**
No subset of CREMA-D / TESS / SAVEE significantly improves on RAVDESS alone
under any feature/head combination tried — 8 subsets on handcrafted features,
10 variants on wav2vec2 embeddings, plus four covariate-shift remedies. Inner
cross-validation, given the choice and no sight of the test fold, discards the
external data in all five folds. *How much* it hurts turns out to depend more on
the classifier than on the features: the same handcrafted features lose 11.1 pp
under an RBF kernel and 4.8 pp under a linear one. See
`docs/merged_baseline_findings.md` and `docs/wav2vec2_transfer_findings.md`.

**The features that recognise emotion best also identify the speaker best.**
wav2vec2 recovers which of 24 actors is talking with 94.9% accuracy — above
MFCCs' 92.7%, 22.8× chance — and per-speaker normalisation leaves *more*
residual identity than it does on MFCCs. The leakage problem does not go away
with better representations.

Every number carries a cluster-bootstrap interval that resamples **speakers,
not clips** (`ser/stats.py`); the headline 0.794 is [0.748, 0.836].

Full log: `results/results.csv`. Split definitions: `docs/splits.md`.
Environment gotchas (Windows Smart App Control, OneDrive): `docs/environment.md`.

---

## Start here

`notebooks/00_project_walkthrough.ipynb` — the guided tour. Twenty minutes,
end to end: the data, the splits, the features, the baseline, the scoring.

`notebooks/02_full_pipeline.ipynb` — **the graded submission
notebook.** Self-contained: every function it uses is defined inside it, so it
imports nothing from `ser/` and reproduces every result in the report starting
from the raw audio. Read this one if you want the whole project in one file.

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
  splits_merged.py the two multi-corpus architectures; own frozen file
  corpora.py       RAVDESS + CREMA-D + TESS + SAVEE as one index
  features.py      parameterised feature extraction
  embeddings.py    frozen wav2vec2 / WavLM / HuBERT; caches all 13 layers
  evaluate.py      the single scoring function everyone calls
  stats.py         cluster bootstrap + McNemar -- uncertainty on every claim
  models/          one module per model family
  run_baselines.py the classical floor
notebooks/         exploration and figures only -- no logic lives here,
                   EXCEPT 02_full_pipeline.ipynb, which inlines
                   everything deliberately: it is the standalone deliverable
splits/folds.csv   committed, frozen 2026-08-22, never regenerated
results/results.csv appended by evaluate.py
docs/              data_audit.md, splits.md, wav2vec2_findings.md,
                   merged_baseline_findings.md, dataset_merge.md,
                   merged_corpus_design.md, environment.md
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
the two files that decode real waveforms — including every fold-disjointness,
no-leakage and metric invariant, the guard that `splits/folds.csv` is
byte-identical after any merged-corpus work, and the cluster-bootstrap
invariants. 259 tests at last count. The full suite needs the clips present;
run it locally before opening a PR.
