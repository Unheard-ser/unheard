# Merged corpus: does more data help?

**No.** Adding CREMA-D, TESS and SAVEE to the training set costs **11.11 pp of
macro-F1** on the RAVDESS test set, and *every* subset of them costs something.

One model (SVM-RBF, default hyperparameters) on the Phase 4 winning feature
config, throughout. The question here is directional; Phase 13-14 settle
whether learned representations behave differently.

## The setup

Architecture A (CLAUDE.md rule 9): the test side is the frozen RAVDESS
`speaker_independent` fold, untouched. Only the training side varies. So every
row below is measured on the identical 1,440 clips, and the only difference
between any two rows is which external clips were added to training.

That matters because it makes this a single-variable experiment. It also means
the `calm` corpus shortcut is unavailable by construction: every test clip is
RAVDESS, so corpus identity is constant on the test set.

## 1. The four cells

| Cell | Accuracy | Macro-F1 | Test set |
|---|---|---|---|
| C1 — Phase 5 reproduction | 0.6389 | 0.6311 | 1,440 RAVDESS |
| C2 — RAVDESS only, merged pipeline | 0.6389 | 0.6311 | 1,440 RAVDESS |
| C3 — + all three external corpora | **0.5236** | **0.5200** | 1,440 RAVDESS |
| C4 — Architecture B *(different denominator)* | 0.5744 | 0.5274 | 12,162 merged |

Two controls before any conclusion:

**C1 reproduces the logged Phase 5 run to four decimals** (0.638889 / 0.631106).
Nothing drifted when the project moved off OneDrive and the environment was
rebuilt.

**C2 equals C1 exactly.** The merged feature pipeline, per-speaker normalised
across 121 speakers instead of 24, returns identical numbers on RAVDESS. This
confirms `per_speaker_normalise` is genuinely per-speaker — adding 97 speakers
does not perturb the 24 that were already there. C2, not C1, is the control the
rest of this document compares against.

**C4 is not comparable** to C1-C3 and is never printed beside them without this
sentence. Different clips, different denominator, 8 classes spread very
differently.

## 2. Which corpora help? None.

Every subset of the external corpora, same frozen test folds:

| Added to training | External clips | Accuracy | Macro-F1 | vs control |
|---|---|---|---|---|
| *(RAVDESS only)* | 0 | 0.6389 | **0.6311** | — |
| SAVEE | 480 | 0.6215 | 0.6131 | −1.80 |
| TESS + SAVEE | 3,280 | 0.5854 | 0.5804 | −5.07 |
| TESS | 2,800 | 0.5826 | 0.5794 | −5.17 |
| CREMA-D + SAVEE | 7,922 | 0.5410 | 0.5361 | −9.51 |
| CREMA-D | 7,442 | 0.5347 | 0.5275 | −10.36 |
| CREMA-D + TESS + SAVEE | 10,722 | 0.5236 | 0.5200 | −11.11 |
| CREMA-D + TESS | 10,242 | 0.5174 | 0.5122 | −11.89 |

**The damage scales with the volume of external data: Pearson r = −0.980
(p = 0.0001), Spearman ρ = −0.893 (p = 0.007).**

Marginal harm diminishes — the first 480 clips cost 3.75 pp per thousand, the
full 10,722 cost 1.04 pp per thousand — which is what saturation looks like, not
a threshold effect.

## 3. Where the points go

Per-class recall, control vs all-three, identical test clips:

| Class | Control | + external | Δ | In CREMA-D? | Share of merge |
|---|---|---|---|---|---|
| calm | 0.745 | 0.380 | **−36.5 pp** | **no** | 1.58% |
| surprised | 0.693 | 0.406 | **−28.7 pp** | **no** | 5.36% |
| happy | 0.677 | 0.516 | −16.2 pp | yes | 15.81% |
| disgust | 0.771 | 0.646 | −12.5 pp | yes | 15.81% |
| sad | 0.417 | 0.339 | −7.8 pp | yes | 15.81% |
| angry | 0.698 | 0.693 | −0.5 pp | yes | 15.81% |
| fearful | 0.547 | 0.630 | **+8.3 pp** | yes | 15.81% |
| neutral | 0.490 | 0.635 | **+14.6 pp** | yes | 14.00% |

The two classes CREMA-D does not contain are the two that collapse. But note
what the sweep adds: **TESS alone, which *does* contain `surprised`, still costs
5.17 pp.** So class coverage is not the mechanism — it decides *which* classes
suffer worst, layered on top of a volume effect that is present regardless.

The honest reading is that both are real and volume dominates:

- **Volume / domain shift** sets the size of the loss (r = −0.98).
- **Class coverage** sets its distribution across classes.

`neutral` gaining 14.6 pp is the one genuinely encouraging number here. It was
RAVDESS's weakest class at 6.7% of the corpus; the merge takes it to 14% and the
model finally has enough of it. The merge is not useless — it is being applied
the wrong way.

## 4. Are the corpora separable at all?

A logistic-regression probe predicting `source`, speaker-grouped so it cannot
win by recognising a voice:

| Features | Corpus identified | Majority baseline | Ratio |
|---|---|---|---|
| raw | **91.8%** | 61.2% | 1.50× |
| per-speaker normalised | 61.2% | 61.2% | **1.00×** |

Raw features identify the corpus 91.8% of the time. Per-speaker normalisation
takes that to *exactly* the majority-class rate — the probe has been reduced to
guessing.

This is the same shape as the Phase 7 actor-identity finding (92.7% raw → 21.2%
normalised), except here the signal is removed completely rather than reduced.
It makes sense: speakers nest inside corpora, so z-scoring within a speaker also
removes that corpus's recording signature.

Consequence for `docs/merged_corpus_design.md` criterion (2): with per-speaker
normalisation in place, corpus identity is not available as a shortcut. The
confound is real in the raw features and neutralised by the pipeline we already
use.

The first version of this probe reported only the normalised number and was
briefly mistaken for "the corpora are indistinguishable". They are extremely
distinguishable; the normalisation is doing the work. Both numbers are reported
here because only the pair is interpretable.

## 5. What this does not say

- It does not say the external data is bad. `docs/dataset_merge.md` shows the
  merge is correctly built: exact counts, zero cross-corpus duplicates, zero
  unreadable files.
- It does not say external data cannot help. It says *naive pooling* does not
  help a shallow classifier on handcrafted features. Reweighting external
  samples, training on merged and fine-tuning on RAVDESS, or using the merge for
  representation learning rather than classification are all untested here.
- It is one estimator at default settings. Phases 13-14 repeat this on learned
  representations, where the picture may differ — a wav2vec2 encoder is already
  pretrained on far more out-of-domain speech than this merge contains, so the
  volume effect that dominates here may not apply.

## 6. Applying it correctly: four transfer strategies

"Naive pooling fails" is a diagnosis, not a conclusion. Four standard remedies
for covariate shift were implemented in `ser/run_transfer.py` and run on the
same frozen test folds.

| Strategy | Accuracy | Macro-F1 | vs control |
|---|---|---|---|
| **RAVDESS only (control)** | 0.6389 | **0.6311** | — |
| scaler-only | 0.6389 | 0.6311 | **0.00** |
| uniform, weight chosen by inner CV | 0.6389 | 0.6311 | **0.00** |
| uniform w = 0.01 | 0.6333 | 0.6269 | −0.42 |
| uniform w = 0.02 | 0.6292 | 0.6249 | −0.63 |
| uniform w = 0.05 | 0.6118 | 0.6068 | −2.43 |
| uniform w = 0.1 | 0.5875 | 0.5837 | −4.74 |
| importance weighting (density ratio) | 0.5854 | 0.5810 | −5.01 |
| uniform w = 0.2 | 0.5611 | 0.5564 | −7.47 |
| uniform w = 0.5 | 0.5451 | 0.5399 | −9.12 |
| uniform w = 1 (naive pooling) | 0.5236 | 0.5200 | −11.11 |
| class-selective (classes from inner CV) | 0.4715 | 0.4586 | **−17.26** |

**Nothing beats the control. The ceiling is a tie.**

### The result that settles it

The `uniform` strategy's weight was chosen by inner cross-validation inside
each training fold, with no access to the test fold. **It chose weight = 0.0 in
all five folds** — the training data, judging for itself, concluded that the
right amount of external data is none. The outer score is then exactly the
control, because that is what weight 0.0 means.

This matters more than the sweep. The sweep shows external data never helps on
*this* test set; the inner-CV result shows the decision to exclude it is
reachable from training data alone, so it is a property of the data rather than
of one particular held-out split.

### Why the down-weighting curve has no sweet spot

Macro-F1 rises monotonically as the external weight falls — 0.5200 → 0.5399 →
0.5564 → 0.5837 → 0.6068 → 0.6249 → 0.6269 — and converges on the control from
below. A curve with a maximum in the interior would mean external data carried
complementary signal worth some non-zero amount. It does not. Every additional
unit of external weight costs accuracy.

### Class-selective was worse than doing nothing

Adding external clips only for the classes inner CV said they helped (`neutral`
and `fearful` in every fold, `angry` in three) produced **0.4586 — six points
below naive pooling and the worst result in the study.**

The mechanism is instructive rather than embarrassing: classes were selected
for individual benefit while ignoring what selection does to the *priors*.
Adding thousands of external clips to three classes and none to the other five
skews the training distribution far harder than adding them everywhere.
`neutral` genuinely does improve when external data is added uniformly (+14.6
pp, §3) — but harvesting that gain in isolation costs more elsewhere than it
returns.

This is worth stating plainly because the §3 per-class table invites exactly
this idea, and it is wrong.

### Importance weighting helped, but not enough

Weighting each external clip by `p(RAVDESS|x) / (1 - p(RAVDESS|x))` — fit on
training data only — recovered 6.1 pp against naive pooling (−5.01 vs −11.11).
The covariate-shift correction is doing real work. It is still beaten by simply
multiplying every external clip by 0.02, and by using no external data at all.

### External data does not even help the representation

`scaler-only` — fit the feature scaler on all 11,862 training clips, fit the
classifier on the 1,140 RAVDESS ones — returns the control to four decimals.
The extra 10,722 clips contribute nothing to the feature standardisation.
Unsurprising in hindsight: the features are already per-speaker normalised, so
a global scaler has little left to learn.

## 7. How much of this is noise?

Every number above is a point estimate, and the project has been reporting
differences of 11.11 pp and 0.42 pp in the same voice. `ser/run_significance.py`
attaches intervals.

The resampling unit is the **speaker**, not the clip. Sixty clips from one actor
succeed or fail together, so treating them as 1,440 independent observations
understates the variance — in the direction that makes weak results look strong.
With 24 test speakers the honest intervals are wide. A test in
`tests/test_stats.py` shows the clip-level bootstrap produces intervals more
than 3× too narrow on clustered data.

### Absolute, with 95% cluster-bootstrap intervals

| Strategy | Macro-F1 | 95% CI |
|---|---|---|
| control (RAVDESS only) | 0.6311 | [0.5865, 0.6759] |
| scaler-only | 0.6311 | [0.5865, 0.6759] |
| uniform w = 0.01 | 0.6269 | [0.5840, 0.6691] |
| uniform w = 0.05 | 0.6068 | [0.5652, 0.6496] |
| importance weighting | 0.5810 | [0.5393, 0.6220] |
| naive pooling | 0.5200 | [0.4720, 0.5677] |

**The project's headline 0.6311 carries a ±4.5 pp interval.** That is not a flaw
in the result; it is the width 24 speakers buy. It does mean any claim resting on
a 1-2 pp difference is unsupported.

### Paired differences against the control

Both models see identical test clips, so the bootstrap resamples the same
speakers for both and bootstraps the *difference* — much sharper than comparing
two overlapping intervals.

| Strategy | Δ macro-F1 | 95% CI | bootstrap p | McNemar p | Verdict |
|---|---|---|---|---|---|
| scaler-only | +0.00 pp | [+0.00, +0.00] | 1.000 | 1.00 | identical predictions |
| uniform w = 0.01 | −0.42 pp | [−1.68, +0.85] | 0.528 | 0.43 | **not distinguishable** |
| uniform w = 0.05 | −2.43 pp | [−4.74, −0.04] | 0.047 | 0.008 | significant, barely |
| importance weighting | −5.01 pp | [−7.50, −2.42] | 0.002 | 5×10⁻⁶ | significant |
| naive pooling | −11.11 pp | [−13.96, −7.99] | <0.001 | 2×10⁻¹⁵ | significant |

So the 11.11 pp headline loss is solid, and so is the 5.01 pp for importance
weighting. But **uniform down-weighting at w = 0.01 is statistically
indistinguishable from using no external data at all** — which strengthens
rather than weakens §6's conclusion: the best any transfer strategy achieves is
a tie, and the tie is a genuine tie rather than a small loss.

### Two retroactive consequences

If −0.42 pp is inside the noise on this test set, then so are two numbers this
project has already reported as results:

- **Phase 5's tuning gain of +0.22 pp** cannot be distinguished from zero. The
  write-up already called it "a fifth of the fold standard deviation"; it is
  now properly unsupported, which if anything sharpens Phase 5's actual point —
  feature design moved the number 13.31 pp and estimator tuning moved it
  nothing.
- **Phase 4's 25 ms framing loss of −1.25 pp** is likewise within the interval.
  It should be reported as "no detectable effect" rather than as a loss.

Neither changes a conclusion. Both should change the wording.

## 8. What follows

1. **Report the merge as a measured negative result.** Four principled transfer
   strategies, one corpus-subset sweep, and an inner-CV selection that
   independently chose to discard the data. That is a stronger finding than a
   marginal gain would have been.
2. **Architecture A's headline stays the RAVDESS-only 0.6389 / 0.6311.**
3. **This result is bounded to shallow classifiers on handcrafted features.**
   The remaining hypothesis is that the external corpora help *representation
   learning* rather than classification — which is Phase 13-14, and where the
   project was already going. A wav2vec2 encoder is pretrained on far more
   out-of-domain speech than this merge contains, so the volume effect that
   dominates here need not apply. Domain-adversarial training (Phase 14) is the
   direct attack on the confound measured in §4.

All runs behind this document are in `results/results.csv` under notes
beginning `phase-9.5`, `phase-12` and `phase-12b`.
