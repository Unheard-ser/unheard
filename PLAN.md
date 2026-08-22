# PLAN.md — phase checklist

One phase per Claude Code session. Enter plan mode, approve, execute, run the
gate check, commit, then move on. Do not start a phase before its gate passes.

---

## Phase 0 — Foundation
- [ ] Repo initialised, `.gitignore` excludes `data/`, `features/`, `__pycache__`
- [ ] `requirements.txt` pinned; environment reproducible from scratch
- [ ] `CLAUDE.md` committed
- [ ] `ser/metadata.py` parses all 7 filename fields into a DataFrame
- [ ] Asserts pass: 1,440 rows, 96 neutral, 192 each other class, 24 actors,
      12 male / 12 female, no strong-intensity neutral
- [ ] pytest covering those invariants

**Gate:** `pytest` green, and `python -c "from ser.metadata import load; print(load().shape)"` prints (1440, N).

---

## Phase 1 — Data audit  → *Data Understanding, 10 marks*
- [ ] Every file loads without error; corrupt or zero-length files listed
- [ ] Sample rate, bit depth, channel count per file — report any inconsistency
- [ ] Duration distribution overall and per emotion
- [ ] Leading/trailing silence measured; proportion of each clip that is silence
- [ ] Amplitude range; any clipping detected
- [ ] Per-actor clip counts (expect 60 each)
- [ ] Written `docs/data_audit.md` stating what is clean, what is not, and what
      preprocessing each finding implies

**Gate:** the audit doc names at least three concrete preprocessing decisions
justified by evidence, not convention.

---

## Phase 2 — EDA and visualisation  → *EDA 10 + Visualization 10*
Note: run EDA on training folds only once Phase 3 exists; for now use all data
but flag anything you would need to redo.
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

## Phase 3 — Frozen evaluation harness
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
