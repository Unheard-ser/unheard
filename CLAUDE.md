# CLAUDE.md — Speech Emotion Recognition (RAVDESS)

## Project

ISB AMPBA foundation project. Multi-class speech emotion recognition on RAVDESS:
1,440 .wav clips, 24 actors, 8 emotion classes. Team of 5. Graded submission.

Deadlines: Mid review 6 Sep 2026. Final review 4 Oct 2026.
Business framing: contact-centre / telecaller QA and CSAT prediction.

## Non-negotiable rules

1. **Never modify `splits/folds.csv` once committed.** Frozen 2026-08-22;
   `write_folds()` raises on an existing file and a test guards it.
   Four protocols — see `docs/splits.md`. Headline metric is POOLED
   out-of-fold, not the mean of per-fold metrics (folds are unequal).
   Every experiment by every
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
  metadata.py      parse the 5-field filename into a DataFrame (see Data facts)
  preprocess.py    load_audio() — THE single audio entry point: mono + 16 kHz
  audit.py         integrity checks on the raw audio
  eda.py           plotting functions (return figures, never call plt.show)
  splits.py        4 protocols — FROZEN in splits/folds.csv, never regenerate
  features.py      feature extraction, parameterised (see below)
  evaluate.py      the single scoring function everyone calls
  models/          one module per model family
notebooks/         exploration and figures only — no logic lives here
  00_project_walkthrough.ipynb  the guided tour: START HERE to onboard
splits/folds.csv   committed, frozen
results/results.csv appended by evaluate.py
figures/           saved plots for the deck
PLAN.md            phase checklist — update as phases complete
```

## Conventions

- Python 3.14 (`.venv/`), pinned `requirements.txt`. Pins are the versions
  actually resolved and verified locally — match them exactly, since feature
  values must be comparable across all five machines.
- All logic in `ser/` as importable modules. Notebooks import from `ser/`, never
  define functions.
- Type hints on all public functions. Docstrings state what is fit on train only.
- Plotting functions return a `Figure`; the caller saves it. Keeps them testable.
- Feature extraction is **parameterised, not hardcoded**: aggregation strategy,
  normalisation strategy, and MFCC count are arguments with defaults, because
  comparing them is part of the experiment.
- pytest for anything with an invariant (counts, fold disjointness, no-leakage).

## Data facts (assert these, don't assume them)

Verified against the actual files on 2026-08-22. Where this copy of RAVDESS
differs from the canonical distribution, the divergence is marked ⚠.

- 1,440 files total. 96 neutral, 192 each of the other seven emotions.
- Neutral has half because there is no strong-intensity neutral recording.
- ⚠ Layout is `data/data/PersonNN/` (note the doubled `data`), **not**
  `Actor_NN/`. 24 folders, exactly 60 clips each.
- ⚠ Filename is **5** hyphen-separated fields, not the canonical 7:
  emotion-intensity-statement-repetition-actor
  e.g. `06-02-01-02-12.wav`
  The constant `modality` (03) and `vocalchannel` (01) fields were stripped
  when this copy was repackaged. `ser/metadata.py` re-adds them as constant
  columns, so the parsed DataFrame matches the canonical RAVDESS shape.
  The parser **rejects** 7-field names rather than mis-parsing them.
- Emotion codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
  07 disgust, 08 surprised.
- Intensity: 01 normal, 02 strong. Statement: 01 "kids", 02 "dogs".
- Actor 01–24. **Odd = male, even = female.** The actor is encoded twice —
  field 5 and the `PersonNN` folder — and they agree in all 1,440 files.
- Actors are the grouping variable for all speaker-independent splits.

Audio properties (measured, see `docs/data_audit.md`):

- WAV PCM_16 at **48 000 Hz**, uniform across all 1,440 files. Note that
  `librosa.load()` silently resamples to 22 050 Hz — **always pass `sr=`.**
- ⚠ 5 files are dual-channel, not mono as originally assumed. Both channels
  are bit-identical, so averaging is lossless. The raw files are left untouched.
- **Never call `librosa.load` directly. Use `ser.preprocess.load_audio`.**
  It is the one entry point that applies the mono downmix and the 16 kHz
  target rate, so those decisions hold everywhere instead of being re-decided
  per call site. Pass `sr=None` only when you deliberately need native rate
  (the audit does).
- Duration 2.94–5.27 s (median 3.67). No zero-length or corrupt files.

## Metrics

Report accuracy AND macro-F1 always. Accuracy alone hides failure on the
under-represented neutral class. Always produce the confusion matrix. Always
produce results sliced by gender, by intensity, and by emotion.

## What good output looks like

The deliverable is a graded report, not a leaderboard entry. A negative result
that is correctly measured is worth more than a high number that is not
reproducible. When a model underperforms, say so and investigate why rather than
tuning until it looks good.
