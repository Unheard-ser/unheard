# Frozen splits and the evaluation harness

**`splits/folds.csv` is frozen as of 2026-08-22.** Every experiment by every team
member scores against these exact fold assignments. That is what makes five
people's numbers comparable — and it only works if nobody regenerates them.

This is CLAUDE.md rule 1, and it is enforced in code: `ser.splits.write_folds`
raises `FileExistsError` on an existing file, and a test asserts the committed
CSV still matches a fresh `build_folds()`. Accidental regeneration fails the
suite rather than silently invalidating every result logged so far.

---

## How to use it

```python
from ser.splits import get_split
from ser.evaluate import evaluate, evaluate_pooled

train_files, test_files = get_split("speaker_independent", fold=0)
```

`get_split` returns **filenames, not row indices** — an index list means
something different when applied to a differently-ordered DataFrame, and that
failure would be invisible.

Score every run through `ser.evaluate.evaluate`. A model that has not been
through it has not been measured (CLAUDE.md rule 5).

---

## The four protocols

| Protocol | Folds | Test size | Purpose |
|---|---|---|---|
| `speaker_independent` | 5 | 300 × 4, 240 | **Primary.** Every result reports this. |
| `random_stratified` | 5 | 288 | Leakage comparison **only** |
| `statement_holdout` | 1 | 720 | Lexical invariance |
| `statement_holdout_si` | 5 | 150 × 4, 120 | Lexical invariance, speakers held out too |

### 1. `speaker_independent` — the primary protocol

5-fold `GroupKFold` grouped by actor. No actor appears on both sides of a fold,
so a model cannot score by recognising a voice it has already heard.

| fold | train | test | test actors | % male | neutral |
|---|---|---|---|---|---|
| 0 | 1140 | 300 | 5 | 40.0 | 20 |
| 1 | 1140 | 300 | 5 | 60.0 | 20 |
| 2 | 1140 | 300 | 5 | 40.0 | 20 |
| 3 | 1140 | 300 | 5 | 60.0 | 20 |
| 4 | 1200 | **240** | **4** | 50.0 | 16 |

### 2. `random_stratified` — the leakage comparison, never reported alone

5-fold `StratifiedKFold` on emotion, seed 42. **All 24 actors appear on both
sides of every fold.** That is not a bug — it is the defect being quantified.
The gap between this and `speaker_independent` is the headline leakage number
for Phase 5. Reporting it on its own would be actively misleading (CLAUDE.md
rule 4), and a test asserts the overlap exists so nobody "fixes" it.

### 3. `statement_holdout` — lexical invariance

Train on statement 01 ("Kids are talking by the door"), test on statement 02
("Dogs are sitting by the door"). 720/720, perfectly balanced on gender and
emotion both sides.

**Caveat:** all 24 actors appear on both sides, so a good score here is
ambiguous — it could be prosodic generalisation, or it could be speaker
familiarity. Which is why protocol 4 exists.

### 4. `statement_holdout_si` — lexical invariance with speakers held out

Reuses the `speaker_independent` actor partition, then restricts train to
statement 01 and test to statement 02. Both the speaker *and* the sentence
differ across the split.

| fold | train | test | actor overlap | statement overlap |
|---|---|---|---|---|
| 0–3 | 570 | 150 | **0** | **0** |
| 4 | 600 | 120 | **0** | **0** |

All 720 statement-02 clips are held out exactly once. **The gap between
protocol 3 and protocol 4 isolates lexical generalisation from speaker
memorisation** — that is the Phase 7 hypothesis, and it needs both numbers.

---

## Why the folds are unequal, and why we accepted it

24 actors over 5 folds forces fold sizes of 5, 5, 5, 5, **4** actors. This was
investigated rather than assumed:

**Emotion balance is a non-issue.** All 24 actors have byte-identical
composition — 4 neutral, 8 of each other emotion. Any actor-grouped split is
therefore automatically proportional on emotion. Nothing to fix.

**Gender imbalance at k=5 is irreducible.** Folds run 40/60 male. A 5-actor
fold has odd parity and can never be 50/50. A hand-balanced assignment —
dealing male and female actors round-robin in opposite directions — produces
the **identical** 20 pp gap. `GroupKFold` is already optimal at k=5.

**The obvious fix is a trap.** `GroupKFold` at k=6 or k=4 divides 24 evenly and
looks attractive. With equal-sized groups it produces **single-gender folds —
0% or 100% male**, which would silently destroy the Phase 7 fairness analysis.
A *hand-assigned* k=6 is genuinely perfect (4 actors = 2M+2F, 240 clips, 0 pp
gap), but k=5 was kept as specified in PLAN.md.

**Consequence, and what we do about it:** fold 4 is 20% smaller, so averaging
five per-fold metrics over-weights each of its clips by ~25%, and its macro-F1
rests on just 16 neutral clips. Hence pooled scoring.

---

## Pooled scoring is the headline number

`evaluate_pooled` concatenates every clip's held-out prediction into one
1,440-row vector and scores it once:

```python
pooled = evaluate_pooled(per_fold, "speaker_independent",
                         model="svm-rbf", feature_config="mfcc40")
```

Every clip counts equally, and the confusion matrix is dense enough for the
per-class and neutral analysis to be stable. **Per-fold rows are still logged**
so variance stays visible — quote the pooled number with the per-fold spread
beside it. Pooled rows carry `fold="pooled"`.

---

## `results/results.csv`

Schema, exactly as PLAN.md specifies — do not add columns, teammates write
readers against this:

```
run_id, timestamp, owner, protocol, fold, feature_config,
model, hyperparams, accuracy, macro_f1, notes
```

- `run_id` — random 8 hex chars, so concurrent appends from five machines
  cannot collide.
- `owner` — read from `git config user.name` automatically; override with
  `owner=`.
- `hyperparams` — sorted JSON, so identical configs produce identical strings.
- `fold` — the fold number, or `pooled`.

`evaluate` also returns `weighted_f1`, `n`, `per_class`, `confusion_matrix` and
the gender/intensity/emotion slices. Those are deliberately **not** CSV columns.

The committed file holds only its header. Regenerate demo rows any time with:

```bash
.venv/Scripts/python.exe -m ser.evaluate --demo
```

---

## Rules for the team

1. **Never edit or regenerate `splits/folds.csv`.** If you believe a split is
   wrong, stop and raise it — do not fix it locally.
2. **Always report `speaker_independent`.** `random_stratified` is only ever
   shown *beside* it, as the leakage comparison.
3. **Score through `ser.evaluate.evaluate`.** Never compute accuracy inline in
   a notebook; the row in `results.csv` is what makes a result exist.
4. **Never pass `results_path` outside tests.** Tests write to `tmp_path` so the
   shared log stays clean.
5. **Fit nothing on the test fold** — scalers, PCA, feature selection, class
   weights and per-speaker statistics are all fit on train only.
