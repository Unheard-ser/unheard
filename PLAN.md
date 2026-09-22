# PLAN.md — phase checklist

One phase per Claude Code session. Enter plan mode, approve, execute, run the
gate check, commit, then move on. Do not start a phase before its gate passes.

## ⚠ Execution order changed: Phase 3 runs BEFORE Phase 2

**Decided 2026-08-22.** The numbered phases below are kept as written for
traceability, but the actual order is **0 → 1 → 3 → 2 → 4 → …**

**Why.** Rigorous measurement is this project's central claim. EDA inspects the
data directly, so running it across all 1,440 clips before the folds exist means
looking at held-out data before the experiment is designed. Even with no model
fitted, that leaks through the analyst: every choice made afterwards — which
features to try, which classes to focus on, which confusions to chase — is
informed by the test set. It is unquantifiable and unfixable after the fact, and
it undercuts exactly the claim the report is built on.

Building the frozen folds first means Phase 2 EDA can be restricted to training
folds only, and any figure that must use all the data is a deliberate, labelled
exception rather than an accident.

**Consequences:**
- Phase 2 EDA runs on **training folds only**. The note already in Phase 2
  ("run EDA on training folds only once Phase 3 exists") becomes binding rather
  than aspirational.
- The 2-D projection coloured by actor (the leakage story) still needs all
  actors to be meaningful. Produce it on training folds, and label any
  all-data version explicitly as a diagnostic, not as evidence for a result.
- Class-balance and per-emotion count charts are properties of the corpus, not
  of the test signal, and may use all 1,440 — stated as such in the caption.

---

## Phase 0 — Foundation
- [x] Repo initialised, `.gitignore` excludes `data/`, `features/`, `__pycache__`
- [x] `requirements.txt` pinned; environment reproducible from scratch
- [x] `CLAUDE.md` committed
- [x] `ser/metadata.py` parses all filename fields into a DataFrame
      (5 fields as packaged, not 7 — see CLAUDE.md Data facts)
- [x] Asserts pass: 1,440 rows, 96 neutral, 192 each other class, 24 actors,
      12 male / 12 female, no strong-intensity neutral
- [x] pytest covering those invariants

**Gate:** `pytest` green, and `python -c "from ser.metadata import load; print(load().shape)"` prints (1440, N).

---

## Phase 1 — Data audit  → *Data Understanding, 10 marks*
- [x] Every file loads without error; corrupt or zero-length files listed
- [x] Sample rate, bit depth, channel count per file — report any inconsistency
- [x] Duration distribution overall and per emotion
- [x] Leading/trailing silence measured; proportion of each clip that is silence
- [x] Amplitude range; any clipping detected
- [x] Per-actor clip counts (expect 60 each)
- [x] Written `docs/data_audit.md` stating what is clean, what is not, and what
      preprocessing each finding implies

**Gate:** the audit doc names at least three concrete preprocessing decisions
justified by evidence, not convention.

---

## Phase 2 — EDA and visualisation  → *EDA 10 + Visualization 10*
**Runs AFTER Phase 3.** See the reordering note at the top of this file.
Binding: EDA uses **training folds only**. Corpus-level counts (class balance)
may use all 1,440 and must say so in the caption; anything that touches signal
— pitch, energy, duration, projections — is restricted to training folds.
- [x] Class balance chart, with the neutral explanation annotated
- [x] Waveform + mel-spectrogram grid: one representative clip per emotion
- [x] Pitch (F0) distribution by emotion; energy/RMS distribution by emotion
- [x] Speaking rate / duration by emotion
- [x] Same analyses split by gender, and by intensity (normal vs strong)
- [x] 2-D projection (PCA and UMAP) of pooled features coloured by emotion,
      then the same plot coloured by **actor** — if actor clusters are tighter
      than emotion clusters, that is the leakage story, visually
- [x] Every figure saved to `figures/` with a one-sentence interpretation in
      `docs/eda_findings.md`

**Gate:** PASSED 2026-08-25. 17 figures in `figures/`, each with an
interpretation in `docs/eda_findings.md`. Signal analysis restricted to the
1,140 training clips of fold 0 (19 actors); only the class-balance count uses
all 1,440 and says so. **Headline: raw features cluster by ACTOR (silhouette
+0.0122) but not by EMOTION (-0.0359)** -- the leakage mechanism, visible in
figures 13/14 and quantified in 17.

---

## Phase 3 — Frozen evaluation harness  *(runs BEFORE Phase 2)*
- [x] `ser/splits.py` defines three protocols:
      - `speaker_independent` — 5-fold GroupKFold grouped by actor
      - `random_stratified` — 5-fold StratifiedKFold (leakage comparison only)
      - `statement_holdout` — train statement 01, test statement 02
      - `statement_holdout_si` — ADDED: the speaker-disjoint variant, so the
        gap between the two isolates lexical generalisation from speaker
        memorisation (see docs/splits.md)
- [x] Fold assignments written to `splits/folds.csv` and committed
- [x] Test asserting no actor appears in both sides of any speaker-independent fold
- [x] `ser/evaluate.py`: one function, returns accuracy, macro-F1, per-class
      precision/recall/F1, confusion matrix, plus slices by gender / intensity /
      emotion; appends one row to `results/results.csv`
- [x] `results.csv` schema: run_id, timestamp, owner, protocol, fold, feature_config,
      model, hyperparams (json), accuracy, macro_f1, notes

**Gate:** PASSED 2026-08-22. Two synthetic models scored through `evaluate.py`
produced comparable rows (pooled acc 0.735 vs 0.303, identical schema).
Headline metric is POOLED out-of-fold, not the mean of per-fold metrics —
folds are unequal (300/300/300/300/240) so averaging over-weights the small
fold. **Folds are frozen — announce to the team.** See docs/splits.md.

---

## Phase 4 — Feature engineering as experiment  → *feeds Models & Approaches, 20 marks*
Not plumbing. Each axis below is a variable to compare, not a default to assume.
- [x] `ser/features.py` parameterised over:
      - MFCC count (13 / 20 / 40), with and without delta / delta-delta
      - aggregation: mean · mean+std · mean+std+min+max · percentiles
      - extra descriptors: chroma, spectral contrast/centroid/rolloff, ZCR, F0 stats
      - normalisation: none · global z-score · **per-speaker z-score**
      - silence trimming on/off
      - framing: n_fft / hop_length / delta width (ADDED -- were invisible
        librosa defaults until a teammate asked)
- [x] Feature cache keyed by config hash; never recompute an existing config
- [x] Ablation grid run with ONE fixed model (SVM-RBF) across configs, so the
      comparison isolates features
- [x] `docs/feature_findings.md`: which axes mattered, which did not

**Gate:** PASSED 2026-08-23. `docs/feature_findings.md` ranks 19 configs plus 5
combinations, with per-speaker (+6.53 pp macro-F1) compared explicitly against
global. Winner `mfcc20-...-per_speaker`: 0.6389 acc / 0.6311 macro-F1, up
+13.31 pp on the floor with fewer features. **The 25 ms framing fix LOST**
(-1.25 pp) -- a negative result, documented. Leakage gap narrowed 13.75 -> 8.82 pp.

---

## Phase 5 — Classical baselines
- [x] SVM-RBF, RandomForest, XGBoost, LogisticRegression, KNN, AdaBoost, GaussianNB
- [x] Each run under `speaker_independent` AND `random_stratified`
- [x] Comparison table; winner identified
- [x] Hyperparameter tuning on the winner only, via CV inside training folds
- [x] **The leakage gap reported as a headline number**

**Gate:** PASSED 2026-08-25. **SVM-RBF, 0.6389 acc / 0.6333 macro-F1**, pooled
speaker-independent, fold sd 0.0509 -- the floor. See docs/baseline_findings.md.
Tuning gained only +0.22 pp macro-F1 (a fifth of the 5.09 pp fold sd): feature
design moved the number +13.31 pp in Phase 4, estimator tuning moved it 0.22 pp.
**Leakage gap 8.82 pp untuned, widening to 10.97 pp tuned** -- tuning against a
leaky protocol manufactures improvement that does not exist.

---

## Phase 6 — Advanced tracks (parallel, one owner each)
> **Owned by a teammate** (agreed 2026-08-25) — not implemented in this stream.
> They must clear **0.6333 macro-F1** (the Phase 5 floor) to justify the
> complexity, and log to `results/results.csv` under the same frozen folds.
- [ ] 6a MLP on aggregated features
- [ ] 6b CNN over mel-spectrograms
- [ ] 6c CNN-LSTM / 1D-CNN over MFCC sequences (the RNN track)
- [ ] 6d Frozen pretrained embeddings (Whisper / wav2vec2 / WavLM) + light head
- [ ] Augmentation (pitch-shift, time-stretch, AWGN) — train folds only —
      as an ablation, on/off, not silently on

**Gate:** every track logged to the same `results.csv` under the same folds.

---

## Phase 7 — Novelty experiments  → *Storytelling + Metrics marks*
- [x] **Leakage diagnosis:** train a classifier to predict actor identity from
      the features. If it succeeds, the features encode identity. Then test
      whether per-speaker normalisation narrows the accuracy gap.
- [x] **Lexical invariance:** train on statement 01, test on 02. Does prosody
      generalise across content?
- [x] **Intensity conditioning:** accuracy on normal vs strong clips reported
      separately; optionally strong-first curriculum training.
- [x] **Gender fairness:** per-gender accuracy and macro-F1. Links directly to
      the demographic-bias risk line in the business case slides.
- [x] **Arousal-first two-stage:** high/low arousal, then emotion within branch.
      Compare against flat 8-way.
- [x] Confusion-structure analysis: do errors cluster by arousal?

**Gate:** PASSED 2026-08-25. Six experiments in docs/novelty_findings.md, each
with hypothesis / result / interpretation. Three confirmed, one partly refuted,
one refuted -- all reported in equal detail.
**Headline: raw features identify the speaker with 92.7% accuracy, 22.3x
chance.** Per-speaker normalisation cuts that to 21.2% but not to zero, which
is why the leakage gap narrowed rather than closed. Arousal two-stage FAILED
(-1.62 pp): the gate is only 85.1% accurate and its errors are unrecoverable.
Gender gap 12.8 pp macro-F1, worse on male -- must be disclosed.

---

## Phase 8 — Deliverables
- [x] All notebooks executed top to bottom, outputs saved, zero errors
- [ ] Files named `Group-nn`; no .zip; code as .ipynb not PDF  <- needs group number
- [ ] Assignment Submission Form with all names + PGIDs  <- YOURS: I have no names/PGIDs
- [x] Deck follows the Session 1 structure (abstract → motivation → problem →
      data → EDA → approach → results → deployment → conclusion → references)
- [ ] Code demo video recorded — 10 marks, owner assigned  <- YOURS: I cannot record

---

# Final-review arc — added 2026-09-22

A teammate delivered a merged corpus (RAVDESS + CREMA-D + TESS + SAVEE, 12,162
clips, 121 speakers) and the wav2vec2 track (Phase 6d) is the last modelling
work open. Both land here. The governing decisions are in
`docs/merged_corpus_design.md`, written **before** any result existed, and in
CLAUDE.md rule 9.

**Two architectures, both built, headline decided by pre-registered criterion:**
- **A** — external corpora in *training only*; test side is the frozen RAVDESS
  fold. 8-class problem intact, comparable to the 0.6333 floor, `calm` shortcut
  unavailable by construction.
- **B** — new frozen folds over all 12,162 clips (`splits/folds_merged.csv`).

## Phase 9 — Corpus integration and audit
- [x] `ser/corpora.py` — the teammate's builder as an importable module;
      RAVDESS delegated to `ser.metadata`, not re-parsed
- [x] `data_external/` gitignored; CLAUDE.md rules 1/4/7 amended, rule 9 added
- [x] CREMA-D / TESS / SAVEE fetched
- [ ] `ser/run_corpus_audit.py` → `docs/dataset_merge.md`
- [x] `tests/test_corpora.py` (29 tests)

**Gate:** counts match published sizes, no cross-corpus duplicates, both
confounds named in the doc.

## Phase 9.5 — Early read
- [ ] `ser/run_early_read.py`: one model (SVM-RBF, defaults) across four cells
      — C1 phase-5 reproduction (drift check), C2 RAVDESS-only on the merged
      matrix (the honest control), C3 Architecture A, C4 Architecture B — plus
      the corpus-identity probe

**Gate:** C1 reproduces 0.6389/0.6333. Without that nothing else means anything.

## Phase 10 — EDA on the merged corpus
- [ ] Training folds only, as Phase 2. Class balance by corpus, clips per
      speaker, duration, spectral rolloff, RMS loudness, F0 per emotion faceted
      by corpus
- [ ] **2-D projection coloured by CORPUS vs by EMOTION with silhouettes** —
      the direct analogue of Phase 2's actor-vs-emotion figure
- [ ] `docs/merged_eda_findings.md`

## Phase 11 — Splits
- [x] `ser/splits_merged.py` — Architecture A helper + three Architecture B
      protocols, own write-once freeze guard
- [x] `tests/test_splits_merged.py` (17 tests) — including a test that hashes
      `splits/folds.csv` either side of every merged operation
- [ ] Freeze `splits/folds_merged.csv`

## Phase 12 — The seven classical models, re-run
- [x] Corpus-selection sweep: all 8 subsets. **Every one hurts**, monotonically
      with volume (Pearson r = −0.98). Best remains RAVDESS alone.
- [x] `docs/merged_baseline_findings.md`
- [ ] 2 × 2 × 7 grid: {handcrafted, wav2vec2} × {Arch A, Arch B} × seven models
- [ ] `random_stratified` counterparts so every cell has a leakage gap

## Phase 12b — Applying the external corpora correctly
Four covariate-shift remedies, `ser/run_transfer.py`. **None beats the control;
the ceiling is a tie.**
- [x] Uniform down-weighting (sweep + weight chosen by inner CV)
- [x] Importance weighting (density ratio), −5.01 pp
- [x] Class-selective via inner CV — **worst result, −17.26 pp**; selecting
      classes for individual benefit wrecks the priors
- [x] Scaler-only — exactly ties the control
- [x] **Inner CV chose weight = 0.0 in all five folds**, with no access to test.
      The data itself says discard the external corpora.

## Phase 12c — Uncertainty  *(new; applies to every phase)*
- [x] `ser/stats.py` — cluster bootstrap (resamples **speakers**, not clips),
      paired bootstrap on differences, exact McNemar
- [x] `tests/test_stats.py` (16 tests), incl. clip-level intervals proven >3×
      too narrow on clustered data
- [x] `ser/run_significance.py` → the headline 0.6311 carries a **±4.5 pp**
      95% interval
- [ ] Reword Phase 5's +0.22 pp tuning gain and Phase 4's −1.25 pp framing loss:
      both are inside the noise and are currently written as results

## Phase 13 — wav2vec2, frozen embeddings  *(CLOSES Phase 6d)*
- [x] `ser/embeddings.py` — one pass caches **all 13 layers**, so the layer
      sweep costs no extra forward passes. 60 min, 295 ms/clip, 971 MB.
- [x] `tests/test_embeddings.py` (34 tests)
- [x] Layer → pooling → normalisation → head sweep
- [x] Full 13-layer curve + figure 18
- [x] Actor-ID probe re-run on learned features + figure 19

**Gate: PASSED. 0.7993 acc / 0.7944 macro-F1** [0.7479, 0.8355], pooled
speaker-independent — **+16.33 pp** over the 0.6311 floor, clearing the 0.6333
bar set for a deep track. `wav2vec2-base`, **layer 6**, mean pooling,
per-speaker normalisation, **linear** head. See `docs/wav2vec2_findings.md`.

Three headlines:
- **Emotion lives mid-network.** Layer 6 beats the final layer by 17.6 pp;
  layers 11-12 fall *below* the handcrafted floor. Taking `last_hidden_state`,
  which every quickstart shows, would have cost 17.6 points.
- **Per-speaker normalisation is worth +16.01 pp here**, against +6.53 pp on
  MFCCs — the largest single effect in the sweep.
- **The best emotion features are the best speaker features.** wav2vec2
  identifies the actor at **94.9%** (22.8x chance) vs MFCCs' 92.7%, and retains
  *more* residual identity after per-speaker normalisation (34.9% vs 21.2%).
  Better representations do not dissolve the leakage problem.

Not claimed: logistic-regression's +0.88 pp over svm-rbf is inside the noise
(p = 0.442) and is written up as indistinguishable.

## Phase 13b — does the merged corpus help once features are learned?
- [x] Corpus-subset sweep + down-weighting on wav2vec2 embeddings
- [x] **Head control** (`ser/run_head_control.py`) — handcrafted features
      through the Phase 13b head, so features are the only variable

**Gate: PASSED, with a claim withdrawn.** Still nothing helps — but **7 of 10
variants are now statistically indistinguishable from using no external data**,
against Phase 12 where every subset hurt badly.

The mid-run reading was "better representations absorb domain shift better"
(damage shrinking fourfold). **The control overturned it.** Swapping SVM-RBF for
LinearSVC on the *same handcrafted features* cuts the all-three penalty from
−11.11 pp to −4.82 pp — over half the effect attributed to wav2vec2. Held
like-for-like, wav2vec2 is better on two cells and worse on two.

What survives, and is now stronger for surviving a head swap that moved every
individual number: **no subset of the external corpora significantly improves
RAVDESS performance under any feature/head combination tested.**

Still open: Phase 13's actual winner (logistic regression, 0.7944) could not be
run across this sweep inside the compute budget — 35 min per fit at 8,882×768.
Whether the merged corpus helps *that* model is untested.

See `docs/wav2vec2_transfer_findings.md`.

## Phase 14 — Pushing wav2vec2 further
- [ ] Learnable layer-weighted sum (SUPERB) — exact from cached per-layer means
- [ ] Speaker-adversarial training (gradient reversal) — attacks the 92.7%
      finding head-on; before/after on emotion F1 *and* actor-ID accuracy
- [ ] Prosody fusion: wav2vec2 ⊕ cached handcrafted F0/energy/RMS
- [ ] Encoder comparison: WavLM-base-plus, HuBERT-base
- [ ] Class-weighted heads for neutral (6.7%) and calm (1.6%)
- [ ] Colab fine-tuning notebook written, not run — documented as future work

## Phase 15 — Consolidation
- [ ] `notebooks/03_merged_corpus.ipynb`, `notebooks/04_wav2vec2.ipynb`
- [ ] README, DECK, and the final comparison table with the pre-registered
      criterion applied in the open

---

## Status summary — 2026-08-25

| Phase | State | Headline |
|---|---|---|
| 0 Foundation | done | 1,440 clips parsed, 5-field schema |
| 1 Data audit | done | 50.6% silence; loudness carries emotion |
| 3 Frozen harness | done | 4 protocols, folds frozen |
| 2 EDA | done | actor silhouette +0.012 vs emotion -0.036 |
| 4 Features | done | +13.31 pp; 25 ms framing LOST |
| 5 Baselines | done | **SVM-RBF 0.6389 / 0.6333** -- the floor |
| 6 Deep tracks | **teammate** | must beat 0.6333 macro-F1 |
| 7 Novelty | done | **92.7% actor ID**; two-stage failed |
| 8 Deliverables | partial | notebooks + deck done; video/form yours |

Tests: 163 passing. Logged runs: 173.
