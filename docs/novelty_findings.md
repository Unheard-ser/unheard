# Novelty experiments — Phase 7

Five experiments. Each states a **hypothesis**, a **result**, and an
**interpretation** — including the two that failed. A negative result correctly
measured is worth more than a high number that is not reproducible.

Reproduce with:

```bash
.venv/Scripts/python.exe -m ser.run_phase7
```

Model: SVM-RBF on `mfcc20-...-per_speaker`, the Phase 5 winner. Reference
point: **0.6389 accuracy / 0.6311 macro-F1** pooled speaker-independent.

---

## 1. Leakage diagnosis — can the features identify the speaker?

**Hypothesis.** If the leakage gap is caused by the features encoding speaker
identity, then a classifier trained to predict *the actor* should succeed far
above the 1/24 = 4.2% chance rate. And if per-speaker normalisation removes
identity, that accuracy should collapse.

**Result.**

| Features | Actor-ID accuracy | vs chance |
|---|---|---|
| **raw** | **0.9271** | **22.3×** |
| per-speaker normalised | 0.2118 | 5.1× |

**Interpretation.** This is the most direct evidence in the project. **The
features identify which of 24 actors is speaking with 92.7% accuracy.** They
are, to a first approximation, a speaker-recognition system that we are asking
to do emotion recognition. Every result under a random split is contaminated by
this, and the 8.82 pp leakage gap is the measurable consequence.

Per-speaker normalisation cuts identifiability from 92.7% to 21.2% — it removes
most of the identity signal, which is exactly why it was worth +6.53 pp in
Phase 4.

**But it does not remove all of it.** 21.2% is still 5.1× chance. Residual
speaker information survives normalisation, which sets a floor on how much of
the leakage gap any feature-level fix can close, and explains why the gap
narrowed to 8.82 pp rather than to zero.

*Note on design:* actor identity cannot be tested speaker-independently — a
held-out actor has no training examples of themselves — so this uses a
stratified split over actors. That is the correct design for this question, not
a lapse from the primary protocol.

---

## 2. Lexical invariance — does prosody survive a change of words?

**Hypothesis.** If the model reads *how* something is said rather than *what* is
said, it should transfer from statement 01 ("Kids are talking by the door") to
statement 02 ("Dogs are sitting by the door").

**Result.**

| Protocol | What changes at test time | Accuracy | Macro-F1 |
|---|---|---|---|
| `speaker_independent` | speaker | 0.6389 | 0.6311 |
| `statement_holdout` | sentence | 0.6194 | 0.6053 |
| `statement_holdout_si` | **both** | 0.5431 | 0.5175 |

**Interpretation.** Changing the *sentence* alone costs **−2.58 pp** macro-F1.
Changing the *speaker* alone costs **nothing by definition** (that is the
reference). Changing **both** costs **−11.36 pp**.

**Prosody generalises across content almost for free.** A 2.6 pp drop for an
entirely different sentence is a genuinely good result — it says the model is
reading delivery, not memorising the acoustics of particular words.

The interesting quantity is the difference between the two statement protocols:
**8.78 pp**. That is the portion of `statement_holdout` performance attributable
to *speaker familiarity* rather than to lexical generalisation. Anyone reporting
`statement_holdout` alone would overstate lexical invariance by that much, which
is precisely why the speaker-disjoint variant was added to `folds.csv`.

---

## 3. Intensity conditioning

**Hypothesis.** Strongly-performed emotion is easier than normal-intensity
emotion, so accuracy should be higher on strong clips.

**Result.**

| Intensity | n | Accuracy | Macro-F1 |
|---|---|---|---|
| normal | 768 | 0.5990 | 0.6003 |
| strong | 672 | **0.6845** | 0.5984 |

**Interpretation — and this one nearly produced a wrong conclusion.**

Accuracy is 8.6 pp higher on strong clips, which looks like clean confirmation.
But **macro-F1 is essentially identical (0.6003 vs 0.5984)**, and that
discrepancy has a cause worth catching:

**The strong subset contains no neutral clips at all.** RAVDESS never recorded
strong-intensity neutral, so `strong` is a **7-class** problem that excludes our
single hardest class, while `normal` is the full **8-class** problem. Some of
the apparent advantage is not difficulty at all — it is a easier label set.

Macro-F1, which weights every class equally, sees through this and shows the two
conditions are close to equally hard once class composition is accounted for.

**Consequence for the business case:** the earlier claim that normal-intensity
performance (~60%) is the honest deployment estimate still stands, but the
*reason* is not that strong clips are dramatically easier. It is that real
contact-centre audio contains neutral speech, and neutral is the class we fail
on. That is a sharper and more defensible framing.

---

## 4. Gender fairness

**Hypothesis.** Per-gender performance differs enough to matter for the
demographic-bias risk in the business case.

**Result.**

| Gender | n | Accuracy | Macro-F1 |
|---|---|---|---|
| female | 720 | 0.6958 | 0.6923 |
| male | 720 | 0.5819 | 0.5646 |
| **Gap** | | **11.4 pp** | **12.8 pp** |

**Interpretation.** **The model is substantially worse on male speakers**, and
the gap is larger on macro-F1 (12.8 pp) than on accuracy — meaning male
performance degrades most on the minority classes.

The dataset is perfectly balanced by gender (12 actors each, 720 clips each), so
this is **not** a data-quantity effect. Phase 2 offers the likely mechanism:
female F0 averages 288 Hz against 178 Hz for male, a 1.62× difference larger
than any between-emotion difference. The emotional pitch excursions that carry
the signal are compressed into a narrower band for male speakers.

**This must be disclosed, not buried.** A contact-centre QA system that is
12.8 pp worse at reading male callers is a deployment risk with a fairness
dimension, and it is the kind of thing that surfaces after launch if it is not
stated before.

---

## 5. Arousal-first two-stage — a failed hypothesis

**Hypothesis.** Phase 2 showed prosody orders by arousal, and the largest
confusions (sad→calm, happy→fearful) sit *within* arousal bands. If the hard
part is discriminating within a band, then splitting the problem — predict
high/low arousal first, then emotion within that branch — should let each
specialist focus and beat the flat 8-way model.

**Result.**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| flat 8-way | **0.6389** | **0.6311** |
| two-stage (arousal first) | 0.6194 | 0.6149 |
| Difference | −1.95 pp | **−1.62 pp** |

Stage-1 arousal accuracy: **0.8507**.

**Interpretation — the hypothesis was right and the remedy still failed.**

The diagnosis holds up (see §6 below): errors really do cluster by arousal. But
the two-stage architecture *loses* 1.62 pp, and the reason is visible in the
stage-1 number.

**The gate is only 85.1% accurate, and its mistakes are unrecoverable.** When
the arousal classifier sends a clip to the wrong branch, the specialist there
cannot possibly produce the right label — the correct answer is not in its
output set. Roughly 15% of clips are condemned before the second stage begins.
The flat model, by contrast, can still recover a cross-band case.

For the two-stage design to win, stage 1 would need to be *more* accurate than
the flat model is on the same distinction. It is not. **Hard routing throws away
the flat model's ability to hedge**, and on this data that costs more than
specialisation gains.

**What would be worth trying instead:** soft routing — multiply each
specialist's probabilities by the gate's arousal probability rather than
committing to a branch. That keeps the specialisation while restoring the
ability to recover from a wrong gate. Not run here; recorded as the natural
follow-up.

---

## 6. Confusion structure — do errors cluster by arousal?

**Hypothesis.** If arousal is the dominant perceptual axis, errors should fall
disproportionately *within* an arousal band rather than across it.

**Result.**

| | Count | Share |
|---|---|---|
| within-arousal errors | 323 | **62.1%** |
| across-arousal errors | 197 | 37.9% |
| chance expectation | | 42.9% |
| **lift over chance** | | **1.45×** |

**Interpretation.** **62.1% of all errors stay inside the arousal band, against
42.9% expected by chance — a 1.45× lift.** The model reliably works out roughly
*how activated* the speaker is, and then struggles to say which emotion at that
activation level.

This is the cleanest statement of what the model has and has not learned. It
confirms the Phase 2 prosody analysis, explains the sad→calm and happy→fearful
confusions, and it is what motivated experiment 5 — which then failed for
reasons unrelated to the diagnosis being wrong.

---

## Summary

| # | Experiment | Outcome |
|---|---|---|
| 1 | Leakage diagnosis | **Confirmed, strongly** — 92.7% actor ID, 22.3× chance |
| 2 | Lexical invariance | **Confirmed** — only −2.6 pp across sentences |
| 3 | Intensity conditioning | **Partly refuted** — accuracy gap is a class-composition artefact |
| 4 | Gender fairness | **Confirmed** — 12.8 pp macro-F1 gap, worse on male |
| 5 | Arousal two-stage | **Refuted** — −1.62 pp; gate errors are unrecoverable |
| 6 | Confusion structure | **Confirmed** — 1.45× lift toward within-arousal errors |

Three confirmed, one partly refuted, one refuted, one confirmed. The two that
did not work are reported here in the same detail as the ones that did.
