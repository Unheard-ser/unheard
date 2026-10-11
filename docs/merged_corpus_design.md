# Merged corpus: design and pre-registered decision criterion

**Written 2026-09-22, before any merged-corpus result existed.** That timing is
the point of this document. Two architectures are being built, and if the
choice between them were made after seeing the numbers, the choice would be
free to flatter whichever produced the larger one. The criterion is fixed here
first, and applied in the open.

## Where this came from

A teammate built `speech_emotion_dataset_builder.ipynb`, which merges CREMA-D
(7,442 clips), TESS (2,800) and SAVEE (480) onto RAVDESS's 8-label schema —
12,162 clips and 121 speakers in total. His RAVDESS handling was verified
against `ser.metadata` and agrees on all 1,440 clips (see
`docs/dataset_merge.md`). The question this document settles is not whether the
merge is correct — it is — but what it is allowed to be *used for*.

## The problem the design has to solve

Two confounds make a naive merge misleading.

**`calm` exists only in RAVDESS.** It is 192 of 12,162 clips (1.6%), and 100%
of them carry RAVDESS recording conditions. A model that learns "this sounds
like a RAVDESS recording" gets `calm` for free, without learning anything about
emotion.

**Corpus, recording conditions and speaker are the same variable.** Holding out
speakers does not hold out corpora. The four corpora were recorded at four
different sample rates (RAVDESS 48 kHz, CREMA-D 16 kHz, TESS 24.414 kHz, SAVEE
44.1 kHz), and only CREMA-D reaches the model's 16 kHz without resampling — so
corpus identity may be audible as a band-limit artefact.

## The two architectures

### A — RAVDESS-test

Test side is the frozen `splits/folds.csv` `speaker_independent` fold,
unchanged. External corpora are added to the **training** side only.

The property that makes this strong is not an argument, it is arithmetic:
every test clip is RAVDESS, so "is this RAVDESS?" is constant across the test
set and has zero discriminative value. The `calm` shortcut is not mitigated, it
is *unavailable*. `tests/test_splits_merged.py` asserts the test side contains
no external clip.

Training speakers rise from 19 to 116, which is itself a leakage reducer: the
Phase 7 finding was that the model can identify the speaker with 92.7%
accuracy, and memorising 116 voices is harder than memorising 19.

Results remain directly comparable to the 0.6389 / 0.6333 floor and to all 446
logged runs.

### B — merged-test

New folds over all 12,162 clips, frozen in `splits/folds_merged.csv`:
`speaker_independent_merged`, `corpus_holdout` (leave-one-corpus-out), and
`random_stratified_merged` (leakage comparison only). Run at 8 labels and at 7
(`calm` folded into `neutral`); the gap between those two *is* the `calm`
shortcut, quantified.

B's numbers are computed over a different clip set and are **not** comparable
to A's or to anything logged before today.

## The criterion

The headline protocol is the one that satisfies these, **in this order**:

1. **The 8-class problem statement is intact.** The assignment is 8-class
   emotion recognition on RAVDESS. A protocol that drops or merges classes to
   raise a number does not answer the question that was asked.
2. **No class's accuracy is attributable to a corpus cue.** Tested by the
   corpus-identity probe: a classifier trained on the same features to predict
   `source`, speaker-grouped. If corpus is near-perfectly identifiable *and*
   the protocol tests on a mixture of corpora, then any class unevenly spread
   across corpora — `calm` above all — is partly measuring corpus detection.
3. **The smaller speaker-independent ↔ random-stratified leakage gap.** The
   handcrafted baseline's gap is 8.82 pp. A protocol that widens it is
   measuring memorisation, whatever its headline accuracy says.

Macro-F1 breaks ties **only after all three are satisfied**. It is not itself a
criterion, because the entire point of this project is that a well-measured
lower number beats a badly-measured higher one.

**Both tables are published in full regardless of which wins**, with the
denominators stated. A result that loses under this criterion is reported, not
dropped.

## What would falsify the design

Stated in advance, so the design is answerable to evidence:

- If the corpus-identity probe scores near the majority-class baseline, the
  corpora are *not* trivially separable, confound (2) is weaker than argued,
  and Architecture B becomes considerably more defensible.
- If Architecture A's macro-F1 does not move against the RAVDESS-only control
  (C2 in `ser/run_early_read.py`), then the extra data adds nothing at this
  feature resolution and should be reported as a negative result rather than
  retained for appearances.
- If `corpus_holdout` scores near `speaker_independent_merged`, then recording
  conditions are not doing the work, and the whole concern was overstated. That
  would be a finding worth reporting in its own right.

## Corpus selection is an experiment, not a setting

`ser.splits_merged.augmented_train` takes `sources`, and Phase 12 sweeps
`{}`, `{TESS}`, `{SAVEE}`, `{CREMA-D}`, `{TESS, SAVEE}`, `{CREMA-D, TESS,
SAVEE}`. CREMA-D contributes no `calm` and no `surprised`, so it may shift the
priors against exactly the two classes RAVDESS is weakest on. Whether more data
helps is measured per corpus rather than assumed in aggregate.
