# Does the merged corpus help once the features are learned?

**No — and the reason the first answer looked different is a confound worth
reporting.**

`docs/merged_baseline_findings.md` closed on a bounded claim: pooling the
external corpora costs 11.11 pp macro-F1, the damage tracks the *volume* of
out-of-domain data at r = −0.98, and four covariate-shift remedies all fail. It
explicitly left one hypothesis open — that the external corpora might help
*representation learning* rather than classification, which wav2vec2 could test.

This phase tested it, got an encouraging answer, and then a control run
dismantled the explanation. Both halves are below, because the second is the
more useful finding.

## 1. The result

Architecture A throughout: frozen RAVDESS `speaker_independent` test folds, only
the training side varies. Head is `LinearSVC` — see §3 for why, and why that
choice turned out to matter enormously.

| Added to training | Macro-F1 | Δ vs control | 95% CI | p | Significant |
|---|---|---|---|---|---|
| **(RAVDESS only)** | **0.7080** | — | — | — | — |
| all external, w = 0.01 | 0.7080 | −0.00 pp | [−3.06, +3.33] | 0.999 | no |
| TESS | 0.7070 | −0.11 pp | [−2.04, +1.78] | 0.960 | no |
| SAVEE | 0.6931 | −1.49 pp | [−4.11, +0.90] | 0.232 | no |
| all external, w = 0.1 | 0.6924 | −1.57 pp | [−4.87, +1.86] | 0.374 | no |
| TESS+SAVEE | 0.6832 | −2.48 pp | [−4.73, −0.02] | 0.049 | yes |
| all external, w = 1 | 0.6826 | −2.54 pp | [−6.26, +1.32] | 0.176 | no |
| CREMA-D+TESS+SAVEE | 0.6826 | −2.54 pp | [−6.26, +1.32] | 0.176 | no |
| CREMA-D+TESS | 0.6788 | −2.92 pp | [−6.12, +0.52] | 0.089 | no |
| CREMA-D+SAVEE | 0.6644 | −4.37 pp | [−7.73, −1.02] | 0.009 | yes |
| CREMA-D | 0.6642 | −4.38 pp | [−7.82, −0.82] | 0.010 | yes |

**Nothing beats the control.** But unlike Phase 12, where every subset hurt and
most hurt severely, **7 of 10 variants here are statistically indistinguishable
from using no external data at all**. The worst case is −4.38 pp against Phase
12's −11.89 pp.

Two internal consistency checks passed: `all external w=1` reproduces
`CREMA-D+TESS+SAVEE` to four decimals (the sample-weight path is equivalent to
no weighting, as it must be), and the control reproduces across reruns.

A caution on reading this table: TESS+SAVEE (−2.48 pp) is flagged significant
while the *larger* CREMA-D+TESS+SAVEE (−2.54 pp) is not. Near-identical deltas,
opposite verdicts — the paired bootstrap's variance depends on *which* clips
each model gets wrong, not just how many. It is a reason not to over-read a
single p-value near 0.05.

## 2. The claim this appeared to support

Comparing the table above against Phase 12 suggested something clean:

| Added | Phase 12 (handcrafted, SVM-RBF) | Phase 13b (wav2vec2, LinearSVC) |
|---|---|---|
| CREMA-D | −10.36 pp | −4.38 pp |
| TESS | −5.17 pp | −0.11 pp |
| SAVEE | −1.80 pp | −1.49 pp |
| all three | −11.11 pp | −2.54 pp |

Every cell's damage shrinks, some fourfold. The obvious reading — and the one
recorded mid-run — was that **better representations absorb domain shift
better**: wav2vec2 is pretrained on ~960 hours of out-of-domain speech, so a few
thousand extra out-of-domain clips perturb it far less.

That reading was wrong, or at least unsupported. The two columns differ in *two*
things, not one: the features changed **and the classifier changed**.

## 3. Why the head had to change, and why that mattered

Phase 13's winner used `LogisticRegression(max_iter=2000)`. On 1,140 RAVDESS
clips that is instant; on 8,882 clips × 768 dimensions it takes **≈35 minutes
per fit**, and this phase needs 50 of them. `LinearSVC` fits the same data in 41
seconds. The swap was made for tractability and documented as internally
consistent — every cell in the phase, including the control, uses it.

Internally consistent it is. But it also means the Phase 12 ↔ Phase 13b
comparison above is not like-for-like, and the head is not a minor detail: on
*identical* wav2vec2 features, multinomial logistic regression scores 0.7944 and
one-vs-rest LinearSVC scores 0.7080. An 8.6 pp gap between two linear models,
with no convergence failure in either.

## 4. The control that settles it

`ser/run_head_control.py` runs the Phase 12 question again on **handcrafted
features with the Phase 13b head**, so features are the only thing that varies:

| Added | handcrafted + **SVM-RBF** | handcrafted + **LinearSVC** | wav2vec2 + **LinearSVC** |
|---|---|---|---|
| (control) | 0.6311 | 0.5405 | 0.7080 |
| CREMA-D | −10.36 pp | **−2.30 pp** | −4.38 pp |
| TESS | −5.17 pp | **−3.12 pp** | −0.11 pp |
| SAVEE | −1.80 pp | **+2.36 pp** | −1.49 pp |
| all three | −11.11 pp | **−4.82 pp** | −2.54 pp |

**The shrinkage is mostly the head, not the features.** Swapping SVM-RBF for
LinearSVC on the *same handcrafted features* cuts the all-three penalty from
−11.11 pp to −4.82 pp — more than half the effect attributed to wav2vec2.

And comparing like-for-like (the two right-hand columns), wav2vec2 is **better on
two cells and worse on two**: much better on TESS (−0.11 vs −3.12) and on all
three (−2.54 vs −4.82), worse on CREMA-D (−4.38 vs −2.30) and SAVEE (−1.49 vs
+2.36). There is no consistent advantage.

So the §2 claim is withdrawn. **The magnitude of the domain-shift penalty is not
a stable property of the features; it depends at least as much on the
classifier.** An RBF kernel is far more sensitive to training-set composition
than a linear boundary is — which makes sense, since a kernel machine's decision
surface is built directly from the training points, and 10,722 of them arriving
from elsewhere reshape it.

One oddity worth recording: **SAVEE + handcrafted + LinearSVC is the only
positive cell anywhere in this project** (+2.36 pp). It is not significant
(p = 0.104) and it sits on the weakest baseline in the table (0.5405), where
there is most room to improve. It should not be reported as "SAVEE helps".

## 5. What survives

Stated carefully, because two readings have already had to be withdrawn:

**Robust.** *No subset of the external corpora significantly improves RAVDESS
performance under any feature/head combination tested* — 8 subsets × handcrafted
× SVM-RBF, 10 variants × wav2vec2 × LinearSVC, 5 cells × handcrafted ×
LinearSVC, plus the four covariate-shift remedies in Phase 12b. That conclusion
is now stronger than when it rested on one configuration, precisely because it
survived a head swap that moved every individual number.

**Not supported.** That wav2vec2 embeddings absorb domain shift better than
handcrafted features. The evidence is mixed once the head is held constant.

**Open.** Whether the penalty would vanish under a head chosen properly for each
feature set, rather than one chosen for tractability. Phase 13's headline
(0.7944, logistic regression) could not be run across this sweep at all within
the compute budget; whether the merged corpus helps *that* model is genuinely
untested.

## 6. A practical note

The `TESS+SAVEE` cell took **5,544 seconds** — 92 minutes — while the larger
`CREMA-D+TESS+SAVEE` took 222. No convergence warning was emitted, so it did
converge; liblinear's coordinate descent is simply pathological on that
particular geometry. If this sweep is rerun, budget for the variance, or switch
to a solver whose cost is predictable from the data size.

All runs are in `results/results.csv` under notes beginning `phase-13b`.
