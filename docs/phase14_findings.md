# Phase 14a: six ways to improve the model, none of which does

**The Phase 13 configuration is on a plateau.** Six standard improvement levers
were applied to wav2vec2 layer-6 + per-speaker normalisation + logistic
regression (0.7944 macro-F1). None produces a statistically distinguishable
gain. Five variants significantly *hurt*.

This is a negative result, and it is the useful kind: it says the next gain has
to come from a **different representation**, not from tuning around this one.

## Why these six

The project had spent four phases on whether external corpora help (they do
not). These are the levers on the *model* that had never been tried, and each
had a specific reason to expect something:

- **Regularisation** — 768 dimensions against 1,140 training clips is badly
  over-parameterised, and `C` had sat at its library default throughout.
- **Dimensionality reduction** — the textbook response to that ratio.
- **Multi-layer combination** — layers 5, 6 and 7 score within 1.5 pp of each
  other, so using only one discards two nearly-as-good views.
- **Prosody fusion** — wav2vec2 normalises its input waveform, so absolute
  loudness is gone by construction, and Phase 1 found loudness *carries*
  emotion. The handcrafted features should be complementary, not redundant.
- **Class weighting** — `neutral` is 6.7% of the corpus and was the weakest
  class throughout the handcrafted phases.

## Results

Frozen RAVDESS `speaker_independent` folds throughout. Paired cluster bootstrap
against the Phase 13 baseline, resampling speakers.

| Lever | Macro-F1 | Δ vs baseline | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| L6 + handcrafted prosody | 0.8030 | +0.86 pp | [−0.45, +2.15] | 0.204 | not distinguishable |
| L6, C = 0.01 | 0.8028 | +0.84 pp | [−0.60, +2.31] | 0.250 | not distinguishable |
| L6, C = 0.1 | 0.8024 | +0.80 pp | [−0.36, +1.97] | 0.201 | not distinguishable |
| L4–L8 concat | 0.8010 | +0.66 pp | [−1.30, +2.58] | 0.531 | not distinguishable |
| L6, C by inner CV | 0.8007 | +0.63 pp | [−0.80, +2.06] | 0.399 | not distinguishable |
| **L6, C = 1 (baseline)** | **0.7944** | — | — | — | — |
| L6, balanced weights | 0.7934 | −0.10 pp | [−0.81, +0.63] | 0.769 | not distinguishable |
| L5+L6+L7 concat | 0.7926 | −0.18 pp | [−2.22, +1.84] | 0.821 | not distinguishable |
| L6, PCA-256 | 0.7869 | −0.75 pp | [−1.89, +0.24] | 0.148 | not distinguishable |
| L6, C = 0.001 | 0.7759 | −1.85 pp | [−4.40, +0.71] | 0.161 | not distinguishable |
| L6, C = 10 | 0.7644 | −3.00 pp | [−4.58, −1.31] | <0.001 | **significantly worse** |
| L6, PCA-128 | 0.7626 | −3.18 pp | [−5.19, −1.33] | 0.001 | **significantly worse** |
| L6, PCA-32 | 0.7388 | −5.56 pp | [−8.44, −2.49] | 0.002 | **significantly worse** |
| L6, PCA-64 | 0.7384 | −5.59 pp | [−8.20, −3.07] | <0.001 | **significantly worse** |
| L6, LDA-7 | 0.5473 | −24.71 pp | [−28.05, −21.02] | <0.001 | **significantly worse** |

## 1. Regularisation: a real curve, an unreal gain

`C` traces a clean inverted U — 0.7759 at C=0.001, peaking at 0.8028 at C=0.01,
falling to 0.7644 at C=10. The default C=1 sits on the descending side, so the
project *was* under-regularised.

But the peak is only +0.84 pp above the default, and its interval spans zero.
Selecting `C` honestly, by inner cross-validation inside each training fold,
gives 0.8007 (+0.63 pp, p = 0.399) — and the inner CV picks C=0.01 in some
folds and C=0.1 in others, which is itself a sign the optimum is flat.

Worth noting for the report: **only the over-regularised end is significantly
bad.** C=10 costs 3.00 pp with p < 0.001. Getting regularisation wrong in the
permissive direction is punished; getting it "right" is not rewarded.

## 2. Dimensionality reduction fails, and that is informative

This is the most interesting cell in the table. With 768 dimensions and 1,140
training clips, compressing the feature space is the obvious move. It does not
work at any setting tried:

| Components | Macro-F1 | Δ |
|---|---|---|
| 768 (none) | 0.7944 | — |
| 256 | 0.7869 | −0.75 pp |
| 128 | 0.7626 | −3.18 pp |
| 64 | 0.7384 | −5.59 pp |
| 32 | 0.7388 | −5.56 pp |
| LDA, 7 | 0.5473 | **−24.71 pp** |

Performance degrades monotonically with compression. **The emotion signal is
distributed across many low-variance directions**, which is precisely what PCA
discards — it ranks directions by variance, and variance in a wav2vec2
embedding is dominated by speaker and channel, not by emotion. LDA projecting
to 7 dimensions is catastrophic: it overfits the projection on 1,140 samples
and loses two-thirds of the performance.

The practical lesson: this representation is *not* redundant despite its
dimensionality. Do not compress it.

## 3. Prosody fusion: the best lever, and still not significant

Concatenating the 162 handcrafted features onto the 768 embedding dimensions
gives the highest score in the phase, 0.8030. The hypothesis was sound —
wav2vec2 normalises its input, so absolute loudness genuinely is absent from it,
and Phase 1 established that loudness carries emotion in this corpus.

But +0.86 pp [−0.45, +2.15] is not a result. It is a direction, and on a test
set of 24 speakers it cannot be told apart from noise.

## 4. What this means

**The plateau is real.** Five of the six levers moved the number by less than a
percentage point in either direction, and the interval on every positive move
includes zero. Phase 13 did not leave easy gains behind.

**Where the remaining headroom is not.** Not in the head (Phase 13 axis 4: seven
estimators, the linear one won). Not in regularisation, layer combination,
class weighting, or dimensionality. Those are now measured rather than assumed.

**Where it might be.** A different encoder — WavLM-base-plus adds denoising and
speaker simulation to the pretraining objective and leads wav2vec2 on SUPERB's
paralinguistic tasks. Or more effective training data via augmentation, which
`PLAN.md` has listed since Phase 6 and which no one on the team has run. Or
fine-tuning the encoder, which is not feasible on 16 CPU threads.

## 5. A caveat in the table itself

The `dim` column in `features/phase14a_levers.parquet` reports 768 for the PCA
rows. That is the *input* dimensionality: PCA runs inside the pipeline, so the
effective dimensionality is the component count in the row label. Cosmetic, but
worth knowing before quoting the table.

All runs are in `results/results.csv` under notes beginning `phase-14a`.
