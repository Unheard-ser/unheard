# Presentation deck — content and speaker notes

Follows the Session 1 structure: abstract → motivation → problem → data → EDA →
approach → results → deployment → conclusion → references.

Figures referenced by filename from `figures/`. Every number is reproducible
from `results/results.csv` under the frozen folds in `splits/folds.csv`.

**Status:** content is complete and final. Slide *design* and the recorded demo
video are still to do — see the checklist at the end.

---

## Slide 1 — Title

**Speech Emotion Recognition on RAVDESS**
*What a correctly-measured result looks like*

Group-nn · ISB AMPBA · [names + PGIDs]

> **Speaker note:** open with the punchline, not the method. "We built a model
> that reads emotion from speech. The interesting part is not the accuracy —
> it's that we can show most published numbers on this dataset are inflated,
> and by how much."

---

## Slide 2 — Abstract

Multi-class speech emotion recognition on RAVDESS: 1,440 clips, 24 actors, 8
emotions. We built a frozen evaluation harness first, then measured.

**Headline: 63.9% accuracy / 63.3% macro-F1**, speaker-independent, pooled
out-of-fold over all 1,440 clips.

**The finding: a random train/test split inflates that to 74.9% — an 11 pp
overstatement.** We can show why, precisely: the features identify *which of
24 actors is speaking* with 92.7% accuracy.

> **Speaker note:** 63.9% against a 12.5% chance rate on 8 classes. Say the
> leakage number early — it is what distinguishes this from a tutorial.

---

## Slide 3 — Motivation: the business case

Contact-centre / telecaller QA and CSAT prediction.

- Manual call review covers ~1–2% of calls; the rest is unmonitored.
- Detecting frustration or anger automatically flags calls for review and
  predicts CSAT before the survey comes back.
- The commercial question is not "can we classify acted emotion" — it is
  **"will this work on a caller the system has never heard?"**

> **Speaker note:** this framing is what makes speaker-independent evaluation a
> business requirement rather than a methodological nicety. Every new caller is
> an unseen speaker, by definition.

---

## Slide 4 — Problem statement

Given ~3 s of speech, predict one of 8 emotions: neutral, calm, happy, sad,
angry, fearful, disgust, surprised.

**Constraints we imposed on ourselves:**

| Rule | Why |
|---|---|
| Speaker-independent evaluation is primary | every real caller is an unseen speaker |
| Nothing fit on test data | scalers, selection, tuning — all train-fold only |
| Folds frozen before any modelling | five people, comparable numbers |
| Accuracy **and** macro-F1 always | neutral is 6.7% of the data and accuracy hides it |

> **Speaker note:** these were fixed before we saw a single result. That
> sequencing is the reason the numbers are trustworthy.

---

## Slide 5 — The data

RAVDESS: 1,440 `.wav`, 24 actors (12 M / 12 F), 8 emotions, fixed script.

**What auditing found — `docs/data_audit.md`:**

| Finding | Consequence |
|---|---|
| **50.6% of the average clip is silence** | trimming is worth **7.4 pp** |
| Peak amplitude spans 44 dB | do **not** peak-normalise (below) |
| 5 files secretly dual-channel | one shared loader, `load_audio` |
| Neutral has half the clips | no strong-intensity neutral exists |
| 48 kHz uniform, 0 corrupt | clean corpus |

**Figure:** `01_class_balance.png`

> **Speaker note:** the audit is 10 marks. Lead with the silence number — it is
> the most actionable and it later proved to be the single largest preprocessing
> effect.

---

## Slide 6 — The measurement that reversed a decision

**Textbook default:** normalise every clip to the same peak volume.

**What we measured:**

| | Spread |
|---|---|
| Loudness across **emotions** | **19× (~26 dB)** — angry-strong vs calm-strong |
| Loudness across **speakers** | 2.9× (~9 dB) |

**Loudness carries emotion, not recording gain.** Peak-normalising would have
deleted the single most discriminative cue in the dataset.

**Figure:** `04_rms_by_emotion.png`

> **Speaker note:** this is the "we measured instead of assuming" slide. We were
> about to apply the default and the data said no. A test now guards it.

---

## Slide 7 — EDA: prosody by emotion

Training folds only — 1,140 clips, 19 actors.

| | Range | Direction |
|---|---|---|
| Mean F0 | 168 Hz (calm) → 296 Hz (fearful) | tracks arousal |
| RMS energy | 10.2× angry over calm | tracks arousal |
| Speech duration | 1.61 s → 2.14 s | **inverse** to arousal |

**Figures:** `03_f0_by_emotion.png`, `04_rms_by_emotion.png`,
`05_duration_by_emotion.png`

> **Speaker note:** every prosodic axis orders by arousal, not by emotion
> identity. That single observation predicts the confusion structure we find
> later. Note the EDA is on training folds only — we did not look at held-out
> data before designing the experiment.

---

## Slide 8 — EDA: the leakage story, in two pictures

**Same feature space. Two colourings.**

| Coloured by | What you see | Silhouette |
|---|---|---|
| **actor** | clean, separated clusters | **+0.0122** |
| **emotion** | no structure at all | **−0.0359** |

**The features know who is speaking. They do not know what is being felt.**

**Figures:** `14_umap_raw_by_actor.png` and `13_umap_raw_by_emotion.png`,
side by side. Then `17_cluster_tightness.png`.

> **Speaker note:** put the two UMAPs next to each other and say nothing for a
> beat. This is the strongest visual in the deck. Actor identity is the *only*
> label with positive silhouette.

---

## Slide 9 — Approach: the frozen harness

Built **before** any modelling, and before EDA.

```
splits/folds.csv    4 protocols, 19,440 rows, FROZEN
results/results.csv every run, one schema, 173 rows
```

| Protocol | Purpose |
|---|---|
| `speaker_independent` | **primary** — 5-fold GroupKFold by actor |
| `random_stratified` | leakage comparison **only** |
| `statement_holdout` | lexical invariance |
| `statement_holdout_si` | lexical invariance, speakers held out too |

Headline metric is **pooled out-of-fold**, not the mean of per-fold scores —
folds are unequal (300/300/300/300/240) because 24 actors do not divide by 5.

> **Speaker note:** `write_folds()` raises if the file exists, and a test asserts
> the committed CSV still matches a fresh build. The contract is enforced in
> code, not by discipline.

---

## Slide 10 — Approach: feature engineering as experiment

19 configurations, one axis at a time, one fixed model.

| Axis | Effect on macro-F1 |
|---|---|
| **per-speaker normalisation** | **+6.53 pp** |
| 20 MFCCs instead of 40 | +3.71 pp |
| percentile aggregation | +3.34 pp |
| extra spectral descriptors | +2.58 pp |
| **shorter 25 ms frames** | **−1.25 pp** ← *expected to win, lost* |
| no deltas | −6.36 pp |
| **no silence trimming** | **−7.42 pp** |

**Winner: `mfcc20 + extras + per-speaker` — +13.31 pp over the floor, using 162
features instead of 240.** Better *and* smaller.

> **Speaker note:** the 25 ms framing result is worth dwelling on. A teammate
> asked what frame length we used; it turned out to be an inherited librosa
> default of 128 ms, five times the speech convention. We tested the "correct"
> value and it lost — because we aggregate over the whole clip, so a longer
> window gives a more stable estimate. Good question, negative result, kept.

---

## Slide 11 — Results: model comparison

Seven classical models, identical features, both protocols.

| Model | SI macro-F1 | SI fold sd |
|---|---|---|
| **SVM-RBF** | **0.6311** | 0.0509 |
| XGBoost | 0.5956 | 0.0620 |
| Logistic regression | 0.5925 | 0.0498 |
| Random forest | 0.5488 | 0.0665 |
| GaussianNB | 0.5071 | 0.0433 |
| KNN | 0.4602 | 0.0312 |
| AdaBoost | 0.4504 | 0.0785 |

**Tuning the winner gained +0.22 pp** — against a fold sd of 5.09 pp.

> **Speaker note:** feature design moved the number 13.31 pp; estimator tuning
> moved it 0.22 pp. Representation beats configuration by roughly sixty to one
> here. Always quote the fold sd — with 24 actors in 5 folds, which speakers
> land in test matters.

---

## Slide 12 — Results: the leakage gap

| | Speaker-independent | Random split | Gap |
|---|---|---|---|
| Untuned | 0.6389 | 0.7271 | **8.82 pp** |
| **Tuned** | 0.6389 | **0.7486** | **10.97 pp** |

**Tuning widened the gap.** It improved the random-split score by 2.15 pp and
left the speaker-independent score untouched.

Under a random split the inner CV can select hyperparameters that exploit
speaker structure present on both sides. Under speaker-independent there is
nothing there to exploit.

**Tuning against a leaky protocol manufactures improvement that does not exist.**

> **Speaker note:** a tuned random-split RAVDESS number inflates twice over —
> once from the split, once from tuning into it. This is the most transferable
> lesson in the project.

---

## Slide 13 — Results: where the model fails

**Confusion matrix** — SVM-RBF, speaker-independent, pooled over 1,440.

| Class | F1 |
|---|---|
| disgust | 0.744 |
| angry | 0.730 |
| surprised | 0.721 |
| calm | 0.699 |
| happy | 0.636 |
| fearful | 0.544 |
| **neutral** | **0.540** |
| **sad** | **0.435** |

Dominant confusions: **sad→calm**, **happy→fearful**, **neutral→calm**.

**62.1% of errors stay within an arousal band** (42.9% expected by chance —
a 1.45× lift).

> **Speaker note:** the model works out *how activated* the speaker is, then
> struggles to say which emotion at that level. This is why accuracy alone would
> mislead — macro-F1 exposes neutral and sad.

---

## Slide 14 — Novelty: we proved the leakage mechanism

**Experiment:** train a classifier to predict *which actor* is speaking.

| Features | Actor-ID accuracy | vs chance (4.2%) |
|---|---|---|
| **raw** | **92.7%** | **22.3×** |
| per-speaker normalised | 21.2% | 5.1× |

**The features are a speaker-recognition system that we are asking to do emotion
recognition.**

Normalisation removes most of the identity — but not all. 21.2% is still 5.1×
chance, which is why the leakage gap narrowed rather than closed.

> **Speaker note:** this is the experiment that turns "there is leakage" from an
> assertion into a measurement. 92.7% out of 24 classes.

---

## Slide 15 — Novelty: what else we tested

| Experiment | Result |
|---|---|
| **Lexical invariance** | Only **−2.6 pp** across a different sentence — prosody generalises |
| ...with speakers held out too | −11.4 pp — so **8.8 pp** was speaker familiarity |
| **Gender fairness** | **12.8 pp** macro-F1 gap, **worse on male** |
| Intensity | Accuracy gap is a class-composition artefact, not difficulty |
| **Arousal two-stage** | **Failed, −1.62 pp** |

**Why the two-stage model failed:** the arousal gate is 85.1% accurate and its
errors are unrecoverable — a clip routed to the wrong specialist cannot receive
the right label. Hard routing discards the flat model's ability to hedge.

> **Speaker note:** the diagnosis was right (errors do cluster by arousal) and
> the remedy still failed. Report both. Soft routing is the natural follow-up.

---

## Slide 16 — Deployment considerations

| Risk | Evidence | Mitigation |
|---|---|---|
| **Unseen speakers** | 11 pp gap | report speaker-independent only |
| **Gender bias** | 12.8 pp worse on male | disclose; monitor per-segment |
| **Acted ≠ spontaneous** | RAVDESS is performed | validate on real call audio before launch |
| **Neutral is weakest** | F1 0.540 | neutral is most of real audio — the binding constraint |
| Latency | 0.8 s to train, ms to predict | not a concern |

**Honest deployment estimate: expect materially below 63.9% on real
contact-centre audio.** RAVDESS is acted, scripted, clean and studio-recorded.

> **Speaker note:** do not let the 63.9% travel unqualified. The gap between
> acted and spontaneous emotion is the largest untested risk in the project.

---

## Slide 17 — Conclusion

**What we built:** a frozen, tested evaluation harness, then a model.
163 tests, 173 logged runs, 4 split protocols.

**What we found:**
1. **63.9% / 63.3% macro-F1** speaker-independent — ~5× the majority baseline
2. **A random split inflates this by 11 pp**, and we proved the mechanism: 92.7% actor identification
3. **Feature design beat model tuning 60 to 1** (+13.31 pp vs +0.22 pp)
4. **Two hypotheses failed** and are reported in full

**What we would do next:** soft-routed arousal model, spontaneous-speech
validation, and the pretrained-embedding track.

> **Speaker note:** close on the negative results. A project that reports only
> its wins has not told you whether to trust it.

---

## Slide 18 — References

- Livingstone & Russo (2018), *The Ryerson Audio-Visual Database of Emotional
  Speech and Song (RAVDESS)*, PLoS ONE 13(5).
- McFee et al., *librosa: Audio and music signal analysis in Python*.
- Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*, JMLR 12.
- Project documentation: `docs/data_audit.md`, `docs/splits.md`,
  `docs/feature_findings.md`, `docs/baseline_findings.md`,
  `docs/novelty_findings.md`.

---

## Remaining checklist — owner: the team

- [ ] Build actual slides from this content
- [ ] Insert figures from `figures/` where referenced
- [ ] **Record the code demo video** (10 marks)
- [ ] **Assignment Submission Form** — all names + PGIDs
- [ ] Rename final files to `Group-nn`
- [ ] Confirm Phase 6 (deep tracks) status from its owner — must beat **0.6333
      macro-F1** to earn a slide
