# What the winning model actually does

CLAUDE.md requires a confusion matrix and slices by gender, intensity and
emotion for every result. `ser.evaluate` has computed all of them on every one
of ~100 Phase 13 runs and discarded them — only accuracy and macro-F1 reach
`results/results.csv`. This is the first look at them for the headline model.

Model: `wav2vec2-base` layer 6, mean pooled, per-speaker normalised, logistic
regression. **0.7993 accuracy / 0.7944 macro-F1**, pooled out-of-fold on the
frozen `speaker_independent` folds.

## 1. Gender: the gap narrowed but did not close

| Gender | n | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|---|
| female | 720 | 0.8431 | **0.8386** | [0.795, 0.882] |
| male | 720 | 0.7556 | **0.7501** | [0.673, 0.810] |

**Gap: 8.84 pp, worse on male.** Phase 7 measured 12.8 pp on handcrafted
features, also worse on male.

Two things follow for the deck's demographic-risk slide, which currently quotes
the 12.8 pp figure:

- **The number is stale and should be updated to 8.84 pp.**
- **The risk has not gone away.** A better representation shrank the gap by
  about a third and left the direction unchanged. The intervals overlap
  slightly, so the gap is not sharply established on 24 speakers — but it has
  now appeared in the same direction on two independent feature sets, which is
  more persuasive than either measurement alone.

The slice is balanced by construction (720 clips each, 12 actors each), so this
is not a sample-size artefact.

## 2. Intensity: the model is better at performed emotion

This is the finding with the most direct consequence for the business case.

| Intensity | n | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|---|
| normal | 672 | 0.7679 | 0.6814 | [0.638, 0.725] |
| strong | 672 | 0.8423 | **0.7385** | [0.695, 0.874] |

**Strong delivery is +5.7 pp macro-F1 easier than normal delivery**, on
identical class composition and identical sample size.

A note on how this is computed, because the naive version is wrong. RAVDESS has
no strong-intensity neutral, so the strong slice contains 7 classes. Macro-F1
over the project's fixed 8 labels scores the absent class 0 and divides by 8,
understating strong by a factor of 7/8 — it appears as 0.7385 vs normal's
0.7617, i.e. *worse*, which is backwards. The table above restricts both slices
to the 7 shared classes. `ser/run_slices.py` reports both and says which is
which.

**Why this matters.** The business framing is contact-centre QA. Nobody in a
support call is performing. RAVDESS's "strong" intensity is theatrical
delivery by professional actors, and it is the easier half of an already
acted corpus. The honest deployment estimate is the **normal-intensity number,
0.681**, not the headline 0.794 — and even that is measured on acted speech.
This belongs in the limitations slide, not buried.

## 3. Per class: the weak class changed

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.867 | 0.880 | **0.873** | 192 |
| disgust | 0.872 | 0.854 | 0.863 | 192 |
| surprised | 0.866 | 0.839 | 0.852 | 192 |
| fearful | 0.820 | 0.828 | 0.824 | 192 |
| calm | 0.769 | 0.849 | 0.807 | 192 |
| happy | 0.746 | 0.750 | 0.748 | 192 |
| neutral | 0.742 | 0.719 | 0.730 | 96 |
| sad | 0.682 | 0.635 | **0.658** | 192 |

**`sad` is now the weakest class, not `neutral`.** Through the handcrafted
phases `neutral` was the problem — it has half the clips of every other class
and Phase 5 reported it at 0.490 recall. Learned features lift neutral to 0.719
recall despite the same class imbalance, and expose `sad` as the genuine
difficulty at 0.635.

That is a real change in where the model's weakness lies, and it means the
class-imbalance framing carried through Phases 4–7 no longer describes the
headline model. Phase 14a's balanced class weights made no difference
(−0.10 pp), which is consistent: the remaining problem is confusability, not
imbalance.

## 4. Errors cluster by arousal, but not cleanly enough to exploit

289 errors. **68.9% stay within the same arousal group** (high: angry, happy,
fearful, surprised, disgust; low: neutral, calm, sad). Random prediction would
put roughly 46% within-group given the class supports, so the structure is real.

The dominant confusions, from the matrix:

| Confusion | Count |
|---|---|
| sad → calm | 31 |
| calm → sad | 17 |
| sad → fearful | 16 |
| surprised → happy | 15 |
| happy → surprised | 12 |
| neutral → calm | 11 |

Two clusters account for most of it: **a low-arousal cluster (sad / calm /
neutral)** and **a high-arousal positive pair (happy / surprised)**. `sad ↔
calm` alone is 48 of 289 errors — 17% — and explains why `sad` is the weakest
class: it is not that sad is unrecognisable, it is that sad and calm are
acoustically adjacent, both low-energy and low-pitch-variance.

**This does not rescue the two-stage arousal model.** Phase 7 built exactly that
on handcrafted features and it lost 1.62 pp, because the arousal gate was only
85.1% accurate and its errors were unrecoverable. 68.9% within-group means 31%
of errors *cross* the arousal boundary, so a gate would still face the same
unrecoverable-error problem. The structure is worth reporting; it is not worth
building on.

## 5. Deployment numbers

Measured in passing during Phase 13 and worth stating, since the deck proposes a
deployable QA tool:

- **295 ms per clip** end to end on 16 CPU threads, no GPU — wav2vec2 forward
  pass plus pooling, for ~3.7 s of audio. Roughly 12× faster than real time.
- **971 MB** for the cached 13-layer representation of 12,162 clips; the
  deployed model needs one layer, so ~75 MB.
- The classifier is a logistic regression over 768 features. Inference cost is
  negligible beside the encoder.

The encoder is the whole cost, it runs on commodity hardware, and it needs no
GPU. That is a genuinely favourable deployment story and it has not been stated
anywhere in the project so far.

## 6. What should change in the deck

1. Gender gap: **12.8 pp → 8.84 pp**, still worse on male, now confirmed on two
   feature sets.
2. Add the intensity caveat: **0.681 on normal delivery** against the 0.794
   headline. This is the number a contact-centre pilot would see.
3. Replace the "neutral is the weak class" framing — it is **sad** now, and the
   mechanism is `sad ↔ calm` confusability rather than class imbalance.
4. Add the latency and footprint figures.

Figures: `21_confusion_wav2vec2.png`, `22_gender_fairness_wav2vec2.png`.
All runs in `results/results.csv` under `phase-14c`.
