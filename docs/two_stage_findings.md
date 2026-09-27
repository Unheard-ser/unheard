# Using all four corpora without losing accuracy

**You can include all 12,162 clips at no statistically measurable cost — if the
extra data is used for representation and the decision layer stays on RAVDESS.**

Every earlier attempt pooled the external clips into one training set. That lost
11 pp on handcrafted features and 4.7 pp here, and four covariate-shift remedies
failed to beat using none of it. But pooling is not how transfer learning
normally works, and the sequential recipe had never been tested.

## Results

Frozen RAVDESS test folds throughout. MLP (256 hidden units) so the model can be
*continued* — logistic regression cannot be, which is why the control here is an
MLP at 0.7938 rather than the 0.7944 headline. The comparison is internal.

| How the extra data is used | Accuracy | Macro-F1 | Δ | 95% CI | p | Verdict |
|---|---|---|---|---|---|---|
| RAVDESS only (control) | 0.7993 | **0.7938** | — | — | — | — |
| **Two-stage, frozen backbone** | 0.7854 | **0.7822** | −1.17 pp | [−3.56, +1.18] | 0.343 | **not distinguishable** |
| Two-stage, full fine-tune | 0.7729 | 0.7687 | −2.51 pp | [−4.69, −0.41] | 0.016 | significantly worse |
| Pooled, all at once | 0.7507 | 0.7471 | −4.68 pp | [−7.50, −1.92] | 0.001 | significantly worse |

## The ordering is the finding

Nothing about the data changed across these four arms — only how much influence
the 10,722 foreign clips are allowed over the final decision:

- **Pooled.** One boundary serves four recording setups at once, and the foreign
  clips outvote the 1,140 that match the test distribution by nearly 10:1.
- **Two-stage, full fine-tune.** The model learns from everything, then continues
  training on RAVDESS alone. Better — it gets to move back — but the fine-tuning
  does not fully undo the pretraining.
- **Two-stage, frozen backbone.** The hidden layer is learned from all 12,162
  clips and then frozen; only the output layer is relearned on RAVDESS. The
  foreign data shapes *what features exist*; RAVDESS alone decides *where the
  boundaries go*.

**The less the foreign data touches the final decision, the less it costs.**
Monotone across four arms, which is what an explanation looks like rather than a
coincidence.

## What this does and does not license

**Does:** including all four corpora is defensible. The cost is 1.17 pp with a
confidence interval spanning zero — indistinguishable from not using them at all.
The dataset a teammate built is genuinely usable rather than merely a negative
result.

**Does not:** it is still not an *improvement*. RAVDESS-only remains the highest
number in every comparison run in this project. The claim is **"free, not
beneficial"**, and it should not drift into "more data helped" in the write-up.

**Why it might be free rather than harmful:** with a frozen backbone the extra
data only has to produce a representation that is *not worse* than one learned
from RAVDESS alone. That is a much lower bar than producing a decision boundary
that transfers, which is what pooling demands of it.

## What remains untested

A backbone pretrained on the merged corpus and fine-tuned with a
**corpus-adversarial** loss — a second head trained to identify the source
corpus, with the gradient reversed so the backbone learns to defeat it. That
directly removes the corpus signature the frozen-backbone arm merely avoids
depending on, and it is the natural next step if the merged data is ever to help
rather than merely not hurt.

All runs in `results/results.csv` under `phase-16`.
