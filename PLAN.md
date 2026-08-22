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
- [ ] Class balance chart, with the neutral explanation annotated
- [ ] Waveform + mel-spectrogram grid: one representative clip per emotion
- [ ] Pitch (F0) distribution by emotion; energy/RMS distribution by emotion
- [ ] Speaking rate / duration by emotion
- [ ] Same analyses split by gender, and by intensity (normal vs strong)
- [ ] 2-D projection (PCA and UMAP) of pooled features coloured by emotion,
      then the same plot coloured by **actor** — if actor clusters are tighter
      than emotion clusters, that is the leakage story, visually
- [ ] Every figure saved to `figures/` with a one-sentence interpretation in
      `docs/eda_findings.md`

**Gate:** each figure has a written interpretation. A chart with no sentence
attached earns nothing.

---

## Phase 3 — Frozen evaluation harness  *(runs BEFORE Phase 2)*
- [ ] `ser/splits.py` defines three protocols:
      - `speaker_independent` — 5-fold GroupKFold grouped by actor
      - `random_stratified` — 5-fold StratifiedKFold (leakage comparison only)
      - `statement_holdout` — train statement 01, test statement 02
- [ ] Fold assignments written to `splits/folds.csv` and committed
- [ ] Test asserting no actor appears in both sides of any speaker-independent fold
- [ ] `ser/evaluate.py`: one function, returns accuracy, macro-F1, per-class
      precision/recall/F1, confusion matrix, plus slices by gender / intensity /
      emotion; appends one row to `results/results.csv`
- [ ] `results.csv` schema: run_id, timestamp, owner, protocol, fold, feature_config,
      model, hyperparams (json), accuracy, macro_f1, notes

**Gate:** two different models scored through `evaluate.py` produce comparable
rows. Announce to the team that folds are frozen.

---

## Phase 4 — Feature engineering as experiment  → *feeds Models & Approaches, 20 marks*
Not plumbing. Each axis below is a variable to compare, not a default to assume.
- [ ] `ser/features.py` parameterised over:
      - MFCC count (13 / 20 / 40), with and without delta / delta-delta
      - aggregation: mean · mean+std · mean+std+min+max · percentiles
      - extra descriptors: chroma, spectral contrast/centroid/rolloff, ZCR, F0 stats
      - normalisation: none · global z-score · **per-speaker z-score**
      - silence trimming on/off
- [ ] Feature cache keyed by config hash; never recompute an existing config
- [ ] Ablation grid run with ONE fixed model (SVM-RBF) across configs, so the
      comparison isolates features
- [ ] `docs/feature_findings.md`: which axes mattered, which did not

**Gate:** a table showing feature configs ranked, with per-speaker normalisation
explicitly compared against global. This is a contribution, not a setting.

---

## Phase 5 — Classical baselines
- [ ] SVM-RBF, RandomForest, XGBoost, LogisticRegression, KNN, AdaBoost, GaussianNB
- [ ] Each run under `speaker_independent` AND `random_stratified`
- [ ] Comparison table; winner identified
- [ ] Hyperparameter tuning on the winner only, via CV inside training folds
- [ ] **The leakage gap reported as a headline number**

**Gate:** a baseline exists that every later model must beat to justify itself.

---

## Phase 6 — Advanced tracks (parallel, one owner each)
- [ ] 6a MLP on aggregated features
- [ ] 6b CNN over mel-spectrograms
- [ ] 6c CNN-LSTM / 1D-CNN over MFCC sequences (the RNN track)
- [ ] 6d Frozen pretrained embeddings (Whisper / wav2vec2 / WavLM) + light head
- [ ] Augmentation (pitch-shift, time-stretch, AWGN) — train folds only —
      as an ablation, on/off, not silently on

**Gate:** every track logged to the same `results.csv` under the same folds.

---

## Phase 7 — Novelty experiments  → *Storytelling + Metrics marks*
- [ ] **Leakage diagnosis:** train a classifier to predict actor identity from
      the features. If it succeeds, the features encode identity. Then test
      whether per-speaker normalisation narrows the accuracy gap.
- [ ] **Lexical invariance:** train on statement 01, test on 02. Does prosody
      generalise across content?
- [ ] **Intensity conditioning:** accuracy on normal vs strong clips reported
      separately; optionally strong-first curriculum training.
- [ ] **Gender fairness:** per-gender accuracy and macro-F1. Links directly to
      the demographic-bias risk line in the business case slides.
- [ ] **Arousal-first two-stage:** high/low arousal, then emotion within branch.
      Compare against flat 8-way.
- [ ] Confusion-structure analysis: do errors cluster by arousal?

**Gate:** each experiment has a stated hypothesis, a result, and an
interpretation — including the ones that fail.

---

## Phase 8 — Deliverables
- [ ] All notebooks executed top to bottom, outputs saved, zero errors
- [ ] Files named `Group-nn`; no .zip; code as .ipynb not PDF
- [ ] Assignment Submission Form with all names + PGIDs
- [ ] Deck follows the Session 1 structure (abstract → motivation → problem →
      data → EDA → approach → results → deployment → conclusion → references)
- [ ] Code demo video recorded — 10 marks, owner assigned
