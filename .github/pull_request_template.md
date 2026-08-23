## What changed

<!-- One or two sentences. What does this PR do that the repo could not do before? -->

## Protocol used

<!-- Which split protocol did you run? Tick every one that applies. -->

- [ ] `speaker_independent` — the primary protocol
- [ ] `random_stratified` — reported only alongside speaker-independent, never alone
- [ ] `statement_holdout`
- [ ] `statement_holdout_si`
- [ ] No experiment in this PR (docs, plumbing, refactor)

Headline number is **pooled out-of-fold**, not the mean of per-fold scores.
If you are reporting one, paste it here:

| metric | value |
| --- | --- |
| accuracy | |
| macro-F1 | |
| n | |

## results.csv

- [ ] `results/results.csv` has a new row for every run in this PR
- [ ] Not applicable — this PR runs no experiments

<!-- If you ran something and did not log it, the run does not exist
     (CLAUDE.md rule 5). Go back and log it before asking for review. -->

## Checklist

- [ ] Nothing was fit on held-out data — scalers, PCA, selection, class weights, per-speaker stats are all fit on the training fold only
- [ ] Augmentation, if any, touches training folds only
- [ ] `splits/folds.csv` is untouched
- [ ] Seeds are fixed (`SEED = 42`)
- [ ] No audio and no `features/*.parquet` in the diff
- [ ] `pytest` passes locally
- [ ] Accuracy **and** macro-F1 reported, with the confusion matrix and the gender / intensity / emotion slices
