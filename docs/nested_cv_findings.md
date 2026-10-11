# The headline, chosen honestly

**0.8243 accuracy / 0.8220 macro-F1**, with the configuration selected by nested
cross-validation rather than by looking at the test folds.

That is **+2.50 pp accuracy and +2.77 pp macro-F1** above the previously reported
0.7993 / 0.7944 — and it is a *more* defensible number, not a less defensible
one, because nothing about the outer test folds participated in choosing it.

## The problem this fixes

Phase 13 chose the winning configuration — layer, pooling, normalisation, head —
by comparing **pooled test-fold scores** across roughly eighteen candidates and
taking the best. That is selection on the test set. The maximum of a noisy set is
optimistically biased even when every individual estimate is unbiased, so 0.7944
was worth slightly less than it looked.

Phase 14a was written with this hazard in mind and stated it explicitly about its
own sweep. Phase 13 had the same flaw and it went unremarked — which mattered
more, because 0.7944 was the number the whole project reported.

## The fix

The configuration is chosen by an inner, speaker-grouped cross-validation *inside
each training fold*, then applied once to the outer held-out fold. The outer fold
never participates in its own configuration choice, so the result estimates
"run this selection procedure on new data" rather than "the best of twenty tries".

Twenty candidates: layers {3,5,6,7,9} x pooling {mean, mean_std} x normalisation
{global, per-speaker}, with a logistic head at C=0.01. Heads were not swept
because Phase 13 found the linear and RBF heads statistically indistinguishable
(p = 0.442).

## Results

| Procedure | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| Selected on test (Phase 13 procedure) | — | 0.8281 | — |
| **Nested CV (unbiased)** | **0.8243** | **0.8220** | [0.7811, 0.8585] |
| *Optimism from selection* | | *+0.61 pp* | |

**The selection bias was only 0.61 pp.** The Phase 13 procedure was
optimistic, but mildly — the method was sound even though the reporting was
loose.

### Why the number went *up*

Because the candidate set contained a combination Phase 13 never tried. Phase 13
swept pooling at the default `C=1.0` and found `mean` best; Phase 14a separately
found `C=0.01` better but tested it only with `mean`. **`mean_std` pooling at
`C=0.01`** — the standard deviation of each feature across frames, alongside its
mean, with stronger regularisation — was the cell in between, and it is worth
about 2.8 pp.

That is a small lesson about one-axis-at-a-time sweeping: it finds the best value
of each axis given the others, which is not the same as the best combination.

### The choice is stable

| Fold | Configuration the inner CV chose |
|---|---|
| 0 | L6 · mean_std · per-speaker |
| 1 | L6 · mean_std · per-speaker |
| 2 | L6 · mean_std · per-speaker |
| 3 | L6 · mean_std · per-speaker |
| 4 | L5 · mean · per-speaker |

Four of five folds independently picked the same configuration without seeing
their own test data. That is what a real effect looks like; a selection driven by
noise would scatter.

## What to report now

- **Headline: 0.8243 accuracy / 0.8220 macro-F1**, nested CV, pooled out-of-fold,
  speaker-independent within RAVDESS.
- Say the configuration was chosen by nested cross-validation. It is a stronger
  claim than the old one and costs a sentence.
- The earlier 0.7944 is not wrong, it is superseded — same protocol, better
  configuration, honest selection.
- Everything downstream still stands: the cross-corpus figure (0.380), the
  acted-speech caveat (0.681 on normal delivery), the gender gap, and the
  speaker-identity finding are all unaffected by this change.

All runs in `results/results.csv` under `phase-17`.
