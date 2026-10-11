# Speech Emotion Recognition — code submission

Multi-class emotion recognition from speech. The primary corpus is RAVDESS
(1,440 clips, 24 actors, 8 emotions); three further corpora — CREMA-D, TESS and
SAVEE — are used to test whether more data and more speakers help.

Business framing: contact-centre call QA. A model that scores emotion on a
recorded call lets a QA team triage which calls a human should listen to, and
gives a leading indicator for CSAT before the survey comes back.

---

## Headline result

| | Accuracy | Macro-F1 |
|---|---|---|
| Handcrafted MFCC features + tuned SVM-RBF | 0.6389 | 0.6333 |
| **wav2vec2 embeddings + linear head (nested CV)** | **0.8243** | **0.8220** |
| 1-D CNN over wav2vec2 frames | 0.8188 | 0.8126 |

All three are **pooled out-of-fold** scores on the 8-class RAVDESS problem under
the speaker-independent protocol: no actor appears in both training and test.

The 0.8220 is the number we stand behind, and it is deliberately not the largest
number in `results.csv`. Single configurations reach 0.8296, but that figure was
chosen *after* seeing the test folds, so it is optimistic by construction.
Nested cross-validation re-runs the selection inside each training fold and
measures the honest value — 0.61 percentage points lower. See notebook 04.

### Why the protocol matters more than the number

The same features and the same model, scored under a random split instead of a
speaker-independent one:

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| Random stratified split | 0.7486 | 0.7429 |
| Speaker-independent split | 0.6389 | 0.6333 |

A random split inflates the result by **11 percentage points**, because clips
from the same actor land on both sides and the model recognises the voice rather
than the emotion. A diagnostic confirmed this directly: the raw features predict
*which actor is speaking* with 92.7% accuracy, 22× chance. Every headline number
in this submission uses the speaker-independent protocol.

---

## Files

| File | What it is |
|---|---|
| `README.md` | this file |
| `requirements.txt` | pinned environment |
| `ser_lib.py` | the whole library: metadata, audio loading, splits, features, embeddings, scoring, statistics, models, multi-corpus |
| `01_project_walkthrough.ipynb` | guided tour — start here |
| `02_ravdess_pipeline.ipynb` | RAVDESS end to end: audit, EDA, features, the seven classical models |
| `03_multicorpus_and_wav2vec2.ipynb` | wav2vec2, the four-corpus merge, transfer, nested CV |
| `04_results_and_findings.ipynb` | every result read off the log, with the diagnostics |
| `05_cnn_on_wav2vec2_colab.ipynb` | the CNN, written for a Colab GPU |
| `run_cnn.py` | the same CNN as a script, for a local CPU run |
| `folds.csv` | the frozen fold assignments — 19,440 rows, 4 protocols |
| `results.csv` | every run ever logged — 1,061 rows, 13 models, 91 feature configurations |

Eleven files. The library is one module rather than a package of twenty because
the submission allows twenty files in total; `ser_lib.install()` re-creates the
original `ser.*` import paths so the notebooks read as they were written.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

wav2vec2 and the CNN additionally need:

```bash
pip install torch==2.13.0 transformers==5.15.1
```

### Audio

The audio is not in this submission — 1,440 RAVDESS clips plus ~10,700 external
clips are several gigabytes, and the brief asks for code.

**Download:** `<PASTE YOUR GOOGLE DRIVE LINK HERE>`

Unzip it beside these files so the tree reads:

```
submission/
  ser_lib.py
  folds.csv
  data/data/Person01/ ... Person24/        RAVDESS, 60 clips each
  data_external/CREMA-D/  TESS/  SAVEE/    the other three corpora
```

RAVDESS is also available from Zenodo (record 1188976) and the external corpora
are fetched automatically by `ser_lib.build_index()` if you would rather not use
the archive.

---

## Running it

Notebooks 01–04 are shipped **with their outputs saved**, so they can be read
end to end without running anything. To re-execute:

```bash
jupyter lab
```

Run them in order. Notebook 01 needs only the RAVDESS audio. Notebook 02
regenerates its own feature caches on first run (about 12 minutes on 16 cores).
Notebook 03 needs the wav2vec2 embeddings, which take roughly 50 minutes to
extract on CPU and are then cached. Notebook 04 reads `results.csv` and needs no
audio at all.

For the CNN, either open notebook 05 in Colab and select a T4 runtime, or run it
locally:

```bash
python run_cnn.py
```

Local CPU training is about 25 minutes for all five folds. The frame-level
features are cached after the first pass.

Every run appends a row to `results.csv`, including the protocol, the exact
feature configuration, the hyperparameters and the seed. No result exists in
this project unless it is in that file.

---

## How the evaluation is kept honest

These are the rules the code enforces, not aspirations:

**The folds are frozen.** `folds.csv` was written once and never regenerated.
Every experiment by all five team members resolves against those exact
assignments, which is what makes five people's numbers comparable. The writer
raises on an existing file and a test guards it.

**The headline is pooled, not averaged.** The folds are unequal (300, 300, 300,
300, 240 clips) because 24 actors do not divide into 5 groups evenly. Averaging
five per-fold F1 scores over-weights the small fold, so predictions from all
five folds are concatenated and scored once.

**Nothing is fit on test data.** Scalers, PCA, feature selection and class
weights are fit on the training fold and applied to the held-out fold. There is
one documented exception: per-speaker normalisation computes a held-out
speaker's statistics from that speaker's own clips, because under the
speaker-independent protocol that speaker has no training clips at all. It uses
no labels, only audio that is equally available at deployment, and it is
standard practice (CMVN). It is transductive, and `ser_lib.per_speaker_normalise`
says so in its docstring. It is worth +16 points, so hiding it would be
misleading.

**Augmentation is training-only.** Never applied to held-out data.

**Seeds are fixed.** `SEED = 42`, set for Python, numpy, scikit-learn and torch
by `ser_lib.set_seeds()`, called at the top of every run.

**Accuracy is never reported alone.** Macro-F1, the confusion matrix, and slices
by gender, intensity and emotion accompany every result. Neutral has 96 clips
against 192 for the other seven classes, and accuracy hides failure there.

---

## What the experiments found

**Learned features beat handcrafted ones by a wide margin, and the margin is
real.** 0.8220 against 0.6333 macro-F1. A speaker-clustered bootstrap — resampling
*speakers*, not clips, since clips from one actor are not independent — puts the
wav2vec2 interval clear of the handcrafted one, and an exact McNemar test on the
paired predictions agrees.

**Mid-depth layers carry emotion; the last layer does not.** Layer 6 of 12 wins.
The final layers specialise toward phonetic content, which is what the
pre-training objective asks of them, so emotion is strongest in the middle.

**Temporal structure adds nothing at this scale.** The CNN keeps the time axis
instead of averaging frames, and ties the pooled linear model (0.8126 against
0.8220). Averaging was not discarding usable signal. Consistent with this, a
*linear* head beat XGBoost on the pooled embeddings by 13 points — the
representation is already close to linearly separable, and extra model capacity
has nothing left to find.

**More data does not help.** Adding CREMA-D, TESS and SAVEE to the training side
never improved the RAVDESS result, across four training arms and a six-way
corpus-selection sweep. Nine augmentation variants — Gaussian noise at three
scales, within-class cross-speaker mixup — moved nothing. The model is neither
data-starved nor capacity-limited; it is limited by what the representation
contains. A frozen-backbone two-stage transfer was the only variant that cost
nothing measurable, and it gained nothing either.

**Cross-corpus generalisation is poor, and that is the deployment number.**
Leave-one-corpus-out scores 0.3576 accuracy / 0.3049 macro-F1 — four different
recording setups, four different microphone chains. A model trained on acted
studio speech from one corpus does not transfer to another. For a contact-centre
deployment this is the honest expectation, and it argues for collecting
in-domain audio rather than for a better architecture.

**The merged-corpus numbers are not comparable to the RAVDESS ones.** Scoring
across all 12,162 clips gives 0.5744 / 0.5274, but that is a different problem
with a different denominator and a different class balance — `calm` is
RAVDESS-only and falls to 1.6% of the merged set. The two tables are reported
side by side with the denominators named, never as a single ranking.

---

## Data

| Corpus | Clips | Speakers | Emotions |
|---|---|---|---|
| RAVDESS | 1,440 | 24 | 8 |
| CREMA-D | 7,442 | 91 | 6 |
| TESS | 2,800 | 2 | 7 |
| SAVEE | 480 | 4 | 7 |
| **merged** | **12,162** | **121** | 8 |

RAVDESS specifics this code asserts rather than assumes, because this copy
diverges from the canonical distribution: the directory layout is
`data/data/PersonNN/`, and filenames carry five hyphen-separated fields
(`emotion-intensity-statement-repetition-actor`) rather than the canonical
seven — the constant modality and vocal-channel fields were stripped when the
copy was repackaged. `ser_lib.parse_filename` re-adds them as constant columns
and rejects seven-field names rather than mis-parsing them.

All files are WAV PCM_16 at 48 kHz, durations 2.94–5.27 s (median 3.67). Five
files are dual-channel; both channels are bit-identical, so the downmix is
lossless. Note that `librosa.load` silently resamples to 22,050 Hz — all audio
in this project goes through `ser_lib.load_audio`, the single entry point that
applies the mono downmix and the 16 kHz target rate that wav2vec2 requires.
