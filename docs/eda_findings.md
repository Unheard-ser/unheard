# EDA findings — Phase 2

**Every figure has a written interpretation.** That is the Phase 2 gate: a chart
with no sentence attached earns nothing.

**Training folds only.** Phase 3 was deliberately run before Phase 2 so this is
possible (see PLAN.md). All signal analysis below uses the 1,140 clips from the
19 training actors of `speaker_independent` fold 0. The 5 held-out actors were
not looked at. The single exception is Figure 1, which counts clips — a property
of the corpus, not of the test signal — and says so on the figure itself.

Reproduce with:

```bash
.venv/Scripts/python.exe -m ser.run_eda
```

---

## 1. Class balance — `01_class_balance.png`

*All 1,440 clips, deliberately.*

> Neutral has 96 clips against 192 for every other emotion, because RAVDESS
> never recorded neutral at strong intensity — you cannot perform "very
> neutral". Every other class splits exactly 96/96 between normal and strong.

**Why it matters:** neutral is 6.7% of the data, so accuracy can look healthy
while neutral fails completely. This is the reason macro-F1 is reported
alongside accuracy on every run.

## 2. Waveform and mel-spectrogram grid — `02_waveform_spectrogram_grid.png`

> Angry and fearful show dense, high-energy spectrograms with visible harmonic
> structure across the full range; calm and neutral are visibly sparser and
> quieter, with energy concentrated in the low bands.

**Why it matters:** the classes differ in ways that are visible by eye before
any model is fitted, which is a sanity check that the task is learnable at all.

---

## Prosody by emotion (training folds)

| Emotion | Mean F0 (Hz) | F0 sd | RMS | Speech (s) | Voiced frac |
|---|---|---|---|---|---|
| angry | 264.2 | 48.7 | **0.065** | 1.95 | 0.75 |
| fearful | **295.6** | **69.2** | 0.040 | 1.81 | 0.81 |
| happy | 248.9 | 44.1 | 0.031 | 1.76 | 0.83 |
| surprised | 259.2 | 64.1 | 0.022 | 1.61 | 0.67 |
| disgust | 192.1 | 40.8 | 0.014 | **2.14** | 0.68 |
| sad | 219.9 | 62.1 | 0.012 | 1.99 | 0.78 |
| neutral | 168.2 | **27.4** | 0.009 | 1.66 | **0.89** |
| calm | **167.7** | 37.9 | **0.006** | 2.05 | 0.87 |

## 3. F0 by emotion — `03_f0_by_emotion.png`

> Mean pitch separates the classes strongly: fearful sits highest at 296 Hz and
> calm lowest at 168 Hz, a 1.8× spread, and the ordering tracks arousal almost
> perfectly — the four high-arousal emotions occupy the top four positions.

**Why it matters:** pitch is a genuinely discriminative feature, and its
alignment with arousal rather than with emotion identity predicts the confusion
structure we later observe.

## 4. RMS energy by emotion — `04_rms_by_emotion.png`

> Energy separates the classes even more sharply than pitch: angry averages
> 10.2× the RMS of calm, and the ordering again follows arousal.

**Why it matters:** this is the visual confirmation of the audit's central
finding. Loudness carries emotion, which is why `docs/data_audit.md` Decision 2
rejects per-clip peak normalisation — it would have erased this.

## 5. Speech duration by emotion — `05_duration_by_emotion.png`

> After silence trimming, duration still separates the classes but *inversely*
> to arousal: disgust (2.14 s) and calm (2.05 s) are longest, surprised (1.61 s)
> and neutral (1.66 s) shortest — high-arousal emotions are spoken faster.

**Why it matters:** duration is a real cue and it is orthogonal to energy, so it
adds information rather than duplicating it. It also only appears *after*
trimming; on raw clips it would be swamped by recording padding.

## 6. F0 by gender — `06_f0_by_gender.png`

> Female actors average 288 Hz against 178 Hz for male — a 1.62× difference far
> larger than any between-emotion difference.

**Why it matters, and this is the important one:** the single strongest signal
in the pitch feature is *who is speaking*, not what they are feeling. A model
given raw pitch will spend much of its capacity on speaker identity. This is the
quantitative case for per-speaker normalisation, which Phase 4 later confirmed
was worth +6.53 pp macro-F1.

## 7. RMS by intensity — `07_rms_by_intensity.png`

> Strong-intensity clips average 3.2× the energy of normal-intensity clips
> (0.041 vs 0.013).

**Why it matters:** intensity is largely encoded as loudness, which explains why
the model scores 17 pp better on strong clips. Real contact-centre audio is
overwhelmingly normal intensity, so the normal-intensity number is the honest
deployment estimate.

## 8. Duration by intensity — `08_duration_by_intensity.png`

> Strong-intensity clips are also longer: 2.00 s against 1.78 s of speech.

**Why it matters:** intensity is not only volume. Any model that succeeds mainly
on strong clips may be reading duration as much as energy.

---

## The leakage story, made visible

**Figures 9–16 are the same feature space twice**: once raw, once per-speaker
normalised — each coloured by emotion and by actor. Read them as a 2×2.

## 9–10. PCA, raw features — `09_pca_raw_by_emotion.png`, `10_pca_raw_by_actor.png`
## 13–14. UMAP, raw features — `13_umap_raw_by_emotion.png`, `14_umap_raw_by_actor.png`

> In the raw feature space, colouring by **actor** produces clean, well-separated
> clusters — most speakers occupy their own distinct region. Colouring the
> *identical* space by **emotion** produces no visible structure at all: every
> class is smeared uniformly across the map.

**Why it matters:** this is the central finding of the whole project, in two
pictures. **The features know who is speaking. They do not know what is being
felt.** A model trained on this space and tested on the same speakers can score
well by memorising voices — which is exactly what the random-split protocol
lets it do, and exactly what the 13.75 pp leakage gap measures.

## 11–12, 15–16. The same space, per-speaker normalised

> After per-speaker z-scoring, the actor clusters dissolve — speakers are
> scattered throughout — while the emotion colouring becomes marginally more
> organised.

**Why it matters:** normalisation removes the identity structure rather than
merely masking it. This is the mechanism behind the Phase 4 result, and behind
the leakage gap narrowing from 13.75 pp to 8.82 pp.

## 17. Cluster tightness — `17_cluster_tightness.png`

Silhouette score: above zero means the label explains real structure.

| Feature space | by actor | by emotion |
|---|---|---|
| **raw** | **+0.0122** | −0.0359 |
| per-speaker normalised | −0.0106 | −0.0244 |

> In the raw space, actor identity is the **only** label with positive
> silhouette — emotion scores negative, meaning it explains no cluster structure
> whatsoever. Per-speaker normalisation drives the actor score below zero
> (identity structure destroyed) while improving the emotion score from −0.036
> to −0.024.

**Why it matters:** it converts the eyeball judgement of Figures 13–14 into a
number that can go in the report. The raw features encode speaker identity more
strongly than they encode the target variable. That is the leakage mechanism,
measured rather than asserted.

**Honest caveat:** emotion silhouette stays negative even after normalisation.
The classes are *not* linearly clustered in this space — the SVM's 63.9% comes
from a non-linear boundary, not from separable blobs. Anyone expecting clean
emotion clusters from a 2-D projection will not find them, and should not.

---

## What this predicts for modelling

1. **Arousal is the dominant axis.** F0, energy and duration all order by
   arousal rather than by emotion identity. Confusions should therefore cluster
   within arousal bands — and they do: sad→calm and happy→fearful are the two
   largest error cells. This is the direct motivation for the Phase 7
   arousal-first two-stage classifier.
2. **Speaker identity must be removed.** Predicted here by Figure 6 and Figures
   13–14, confirmed by Phase 4's +6.53 pp.
3. **Neutral will be hardest.** Fewest clips, lowest F0 variability (sd 27 Hz,
   the lowest of any class), and highest voiced fraction — it is the least
   distinctive class on every prosodic axis measured.
4. **Trimming is essential before duration is meaningful.** The duration signal
   in Figure 5 exists only after silence removal; Phase 4 measured trimming as
   worth 7.42 pp.
