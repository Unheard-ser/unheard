# CLAUDE.md — Speech Emotion Recognition (RAVDESS)

## Project

ISB AMPBA foundation project. Multi-class speech emotion recognition on RAVDESS:
1,440 mono .wav clips, 24 actors, 8 emotion classes. Team of 5. Graded submission.

Deadlines: Mid review 6 Sep 2026. Final review 4 Oct 2026.
Business framing: contact-centre / telecaller QA and CSAT prediction.

## Non-negotiable rules

1. **Never modify `splits/folds.csv` once committed.** Every experiment by every
   team member uses those exact fold assignments. This is the contract that makes
   five people's results comparable. If you believe a split is wrong, stop and ask.
2. **Never fit anything on test data.** Scalers, PCA, feature selection, class
   weights, and per-speaker statistics are fit on the training fold only and applied
   to the held-out fold. No exceptions.
3. **Augmentation applies to training folds only.** Never to held-out data.
4. **Speaker-independent is the primary protocol.** Random-stratified is run only
   as a deliberate comparison to quantify leakage. Never report the random-split
   number alone.
5. **Every run appends a row to `results/results.csv`.** No result exists unless
   it is logged there.
6. **Fix all random seeds.** `SEED = 42`, set for numpy, sklearn, torch, and the
   Python hash seed.
7. **Never commit audio files.** `data/` is gitignored. Never commit
   `features/*.parquet` either — they are regenerable.
8. If execution diverges from the approved plan, stop and re-enter plan mode.

## Repo layout

```
ser/
  metadata.py      parse the 7-field RAVDESS filename into a DataFrame
  audit.py         integrity checks on the raw audio
  eda.py           plotting functions (return figures, never call plt.show)
  splits.py        fold definitions — frozen, written to splits/folds.csv
  features.py      feature extraction, parameterised (see below)
  evaluate.py      the single scoring function everyone calls
  models/          one module per model family
notebooks/         exploration and figures only — no logic lives here
splits/folds.csv   committed, frozen
results/results.csv appended by evaluate.py
figures/           saved plots for the deck
PLAN.md            phase checklist — update as phases complete
```

## Conventions

- Python 3.11, `uv` or venv, pinned `requirements.txt`.
- All logic in `ser/` as importable modules. Notebooks import from `ser/`, never
  define functions.
- Type hints on all public functions. Docstrings state what is fit on train only.
- Plotting functions return a `Figure`; the caller saves it. Keeps them testable.
- Feature extraction is **parameterised, not hardcoded**: aggregation strategy,
  normalisation strategy, and MFCC count are arguments with defaults, because
  comparing them is part of the experiment.
- pytest for anything with an invariant (counts, fold disjointness, no-leakage).

## Data facts (assert these, don't assume them)

- 1,440 files total. 96 neutral, 192 each of the other seven emotions.
- Neutral has half because there is no strong-intensity neutral recording.
- Filename is 7 hyphen-separated fields:
  modality-vocalchannel-emotion-intensity-statement-repetition-actor
  e.g. `03-01-06-01-02-01-12.wav`
- Emotion codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
  07 disgust, 08 surprised.
- Intensity: 01 normal, 02 strong. Statement: 01 "kids", 02 "dogs".
- Actor 01–24. **Odd = male, even = female.**
- Actors are the grouping variable for all speaker-independent splits.

## Metrics

Report accuracy AND macro-F1 always. Accuracy alone hides failure on the
under-represented neutral class. Always produce the confusion matrix. Always
produce results sliced by gender, by intensity, and by emotion.

## What good output looks like

The deliverable is a graded report, not a leaderboard entry. A negative result
that is correctly measured is worth more than a high number that is not
reproducible. When a model underperforms, say so and investigate why rather than
tuning until it looks good.
