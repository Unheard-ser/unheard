# Classical baselines — Phase 5

**The floor every later model must beat to justify its complexity.** Seven
estimators, library defaults, on the Phase 4 winning feature configuration, each
under both protocols. Then hyperparameter tuning on the winner only.

Reproduce with:

```bash
.venv/Scripts/python.exe -m ser.run_phase5
```

Features: `mfcc20-d-dd-mean_std-trim-chroma+spectral_centroid+spectral_contrast+zcr-per_speaker`
(162 features). Fixed across all seven models, so every difference is the
estimator and nothing else.

---

## 1. The comparison

Ranked by **speaker-independent macro-F1** — the primary protocol, pooled
out-of-fold over all 1,440 clips.

| Model | SI accuracy | SI macro-F1 | SI fold sd | RS accuracy | Leakage gap |
|---|---|---|---|---|---|
| **svm-rbf** | **0.6389** | **0.6311** | 0.0509 | 0.7271 | 8.82 pp |
| xgboost | 0.6042 | 0.5956 | 0.0620 | 0.6882 | 8.40 pp |
| logistic-regression | 0.5931 | 0.5925 | 0.0498 | 0.6993 | 10.63 pp |
| random-forest | 0.5688 | 0.5488 | 0.0665 | 0.6486 | 7.99 pp |
| gaussian-nb | 0.5160 | 0.5071 | 0.0433 | 0.5486 | 3.26 pp |
| knn | 0.4861 | 0.4602 | 0.0312 | 0.5764 | 9.03 pp |
| adaboost | 0.4507 | 0.4504 | 0.0785 | 0.4757 | 2.50 pp |

**SVM-RBF wins**, by 3.6 pp macro-F1 over XGBoost. That is a real gap, though
not enormous relative to the ~5 pp fold-to-fold standard deviation.

### What the ranking says

**Margin-based methods beat tree ensembles here.** SVM-RBF and logistic
regression both outperform random forest, and the reason is the data shape: 162
dense, continuous, standardised features over only 1,140 training clips. Trees
split one feature at a time and need volume to find good thresholds; a kernel
method draws a single smooth boundary in the whole space. XGBoost recovers most
of the gap that plain random forest loses, but does not close it.

**AdaBoost is worst, and its fold sd is the highest (0.0785).** With
`SAMME` on 8 classes it fits shallow stumps that cannot express this boundary,
and the instability across folds says it is fitting whichever actors it happens
to see.

**KNN's low variance (sd 0.0312) is not a virtue.** It is the most stable model
and the second worst. It scores consistently badly because distance in a
162-dimensional space is dominated by the many weakly-informative dimensions.

---

## 2. Tuning bought almost nothing

`GridSearchCV` on SVM-RBF, **inside** the training folds, grouped by speaker so
the inner search cannot leak across actors either.

| | Accuracy | Macro-F1 |
|---|---|---|
| svm-rbf, defaults | 0.6389 | 0.6311 |
| **svm-rbf, tuned** | **0.6389** | **0.6333** |
| Gain | **+0.00 pp** | **+0.22 pp** |

**Accuracy did not move at all. Macro-F1 moved by 0.22 pp — a fifth of a
percentage point, against a fold-to-fold sd of 5.09 pp.** The tuning gain is
roughly one twenty-third of the noise. It is not a real improvement.

This is worth stating plainly because it is the opposite of what people expect
from a tuning step, and it has a clear cause: **the features were already doing
the work.** Phase 4 moved macro-F1 by +13.31 pp by changing what the model sees.
Phase 5 moved it by +0.22 pp by changing how the model draws its boundary. On
this dataset, feature design dominates estimator configuration by roughly sixty
to one.

The selected hyperparameters were also unstable across folds — `C` landed on 5
three times, 10 once, and `gamma` split between `scale` and `0.001`. A grid
search that picks a different answer per fold is telling you the surface is
flat.

**Implication for the remaining phases:** effort spent tuning is likely wasted.
Effort spent on representation — better features, or pretrained embeddings in
Phase 6 — is where the remaining gains are.

---

## 3. The leakage gap — headline number

| | Speaker-independent | Random-stratified | Gap |
|---|---|---|---|
| svm-rbf, defaults | 0.6389 | 0.7271 | **8.82 pp** |
| svm-rbf, tuned | 0.6389 | 0.7486 | **10.97 pp** |

**A random train/test split makes this model look like it scores 74.9%. Held out
by speaker — which is what deployment actually looks like — it scores 63.9%.**

### Tuning made the leakage worse

The gap *widened* from 8.82 pp to 10.97 pp. Tuning improved the random-split
score by 2.15 pp while leaving the speaker-independent score untouched.

That is not a coincidence. Under the random split, the inner CV can select
hyperparameters that exploit speaker-specific structure, because the same actors
appear on both sides. Under the speaker-independent split there is no such
structure to exploit, so the same search finds nothing. **Tuning against a leaky
protocol manufactures improvement that does not exist.**

Anyone who reports a tuned random-split number for RAVDESS is reporting this
inflation twice over — once from the split, once from tuning into it.

---

## 4. Gate

**Passed.** The baseline every later model must beat:

> **SVM-RBF, 0.6389 accuracy / 0.6333 macro-F1, pooled out-of-fold under
> `speaker_independent`, fold sd 0.0509.**

A Phase 6 deep model must clear roughly **0.68 macro-F1** to be distinguishable
from this at one standard deviation — and it should be held to more than that,
since it costs far more compute and complexity than an SVM that trains in 0.8
seconds.

## 5. Honest notes

- **All seven models used library defaults**, except `max_iter=2000` on logistic
  regression, which is a convergence fix rather than tuning — at the default 100
  it does not converge on 162 features and the score would measure the iteration
  cap, not the model.
- **XGBoost required a label-encoding wrapper** because it will not accept string
  labels. Wrapped behind the standard fit/predict interface so all seven models
  are driven identically and no caller special-cases it.
- **Every model went through the same scaler pipeline**, including the tree
  models that do not need it. Keeping preprocessing identical means a score
  difference cannot be an artefact of different preprocessing.
- **Fold standard deviations are large** (0.031–0.079). With 24 actors in 5
  folds, which speakers land in the test fold matters. Never quote a
  speaker-independent number without its spread.
