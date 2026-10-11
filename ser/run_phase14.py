"""Phase 14a — making the model better, not measuring the data again.

Phase 13 established the representation: wav2vec2-base, layer 6, mean pooled,
per-speaker normalised, 0.7944 macro-F1. Everything since has been about
whether external corpora help (they do not, under any configuration tried).
This phase goes back to the model.

Six levers, none of which has been tried on this project:

``regularisation``
    768 dimensions against 1,140 training clips. That is badly
    over-parameterised, and ``C`` has been at its library default throughout.
    On MFCCs (162 dims) Phase 5 found tuning bought +0.22 pp, inside the noise;
    at 768 dims with a 1:1.5 sample-to-dimension ratio there is far more reason
    to expect regularisation to matter.

``dimensionality``
    PCA to k components, or LDA to 7 (one fewer than the class count). The
    textbook response to that sample-to-dimension ratio, and never tested here.

``multi_layer``
    Layers 5, 6 and 7 all score within 1.5 pp of each other. Using only layer 6
    discards two nearly-as-good views. Concatenating or averaging them costs
    nothing -- the cache holds every layer.

``prosody_fusion``
    wav2vec2 normalises its input waveform, so absolute loudness is gone by
    construction. Phase 1 found loudness *carries* emotion. The handcrafted
    features (F0, RMS, spectral descriptors) should therefore be complementary
    rather than redundant.

``class_weight``
    ``neutral`` is 6.7% of RAVDESS and was the weakest class throughout the
    handcrafted phases. Balanced class weights cost nothing to try.

``calibration``
    Not a lever on accuracy, but the deployment story in the deck claims a
    confidence-scored QA tool. Whether the probabilities mean anything is a
    separate question from whether the argmax is right.

Discipline
----------
**Every hyperparameter is chosen by inner cross-validation inside the training
fold**, grouped by speaker, never on the test folds. Phase 12b's class-selective
experiment is the cautionary tale: selecting on test-set per-class deltas
produced the worst result in the project. A sweep reported for insight is
labelled as a curve; only the inner-CV-selected value is a claim.

All of it runs on the cached embedding tensor, so the whole phase is minutes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser import SEED, set_seeds
from ser import corpora
from ser.embeddings import EmbeddingConfig, select
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR, feature_columns, per_speaker_normalise
from ser.metadata import load as load_metadata
from ser.run_early_read import MERGED_CACHE
from ser.run_phase5 import WINNER as HANDCRAFTED
from ser.run_wav2vec2 import FLOOR_F1, load_cached, ravdess_view
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

#: Phase 13's winner, the baseline every lever is measured against.
BASE = EmbeddingConfig(layer=6, pooling="mean", normalisation="per_speaker")
BASE_F1 = 0.794382

#: Regularisation grid. Spans four orders of magnitude because the right answer
#: at 768 dims and 1,140 samples is genuinely unknown.
C_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)

#: PCA component counts to consider, plus LDA handled separately.
PCA_GRID: tuple[int, ...] = (32, 64, 128, 256)

INNER_SPLITS = 3


def head(C: float = 1.0, class_weight: str | None = None) -> LogisticRegression:
    """The Phase 13 winning head, with regularisation exposed."""
    return LogisticRegression(
        C=C, max_iter=2000, random_state=SEED, class_weight=class_weight
    )


def ravdess_matrix(layers: tuple[int, ...] = (6,), pooling: str = "mean"):
    """Per-speaker-normalised RAVDESS features from one or more layers.

    Concatenating layers before normalising (rather than after) keeps each
    layer's contribution on the same footing, since per-speaker z-scoring is
    applied per column either way.

    Returns:
        ``(features, labels, index)`` all keyed by filename.
    """
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    tensor, index = ravdess_view(tensor, index)

    blocks = [
        select(tensor, EmbeddingConfig(layer=layer, pooling=pooling))
        for layer in layers
    ]
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    matrix = per_speaker_normalise(matrix, index["speaker_id"].to_numpy())

    columns = [f"L{layer}_{i:03d}" for layer in layers for i in range(blocks[0].shape[1])]
    features = pd.DataFrame(matrix, index=index["filename"], columns=columns)
    labels = pd.Series(
        index["emotion_label"].to_numpy(), index=index["filename"], name="emotion_label"
    )
    return features, labels, index


def prosody_matrix(index: pd.DataFrame) -> pd.DataFrame:
    """The Phase 4 handcrafted features for the same RAVDESS clips.

    Loaded from the merged cache and restricted to RAVDESS, per-speaker
    normalised the same way, so the two blocks are directly concatenable.
    """
    from ser.features import extract

    merged_index = corpora.load_index()
    frame = extract(HANDCRAFTED, merged_index, cache_path=MERGED_CACHE, n_jobs=-1)
    columns = feature_columns(frame, meta=merged_index)
    indexed = frame.set_index("filename")

    keep = index["filename"].tolist()
    values = per_speaker_normalise(
        indexed.loc[keep, columns].to_numpy(),
        indexed.loc[keep, "speaker_id"].to_numpy(),
    )
    return pd.DataFrame(values, index=keep, columns=[f"hc_{c}" for c in columns])


def inner_select(
    train_files: list[str],
    features: pd.DataFrame,
    labels: pd.Series,
    index: pd.DataFrame,
    candidates: list[tuple[str, object]],
) -> tuple[str, object]:
    """Pick a configuration by speaker-grouped CV inside the training fold.

    The test fold is never touched. Returns the winning ``(label, factory)``.
    """
    from sklearn.metrics import f1_score

    speakers = index.set_index("filename").loc[train_files, "speaker_id"].to_numpy()
    names = np.asarray(train_files)
    splitter = GroupKFold(n_splits=INNER_SPLITS)
    y = labels.loc[train_files].to_numpy()

    best, best_score = candidates[0], -np.inf
    for label, factory in candidates:
        scores = []
        for inner_train, inner_test in splitter.split(names, y, speakers):
            tr, te = list(names[inner_train]), list(names[inner_test])
            model = factory()
            model.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
            predicted = model.predict(features.loc[te].to_numpy())
            scores.append(
                f1_score(labels.loc[te].to_numpy(), predicted,
                         average="macro", zero_division=0)
            )
        mean = float(np.mean(scores))
        if mean > best_score:
            best, best_score = (label, factory), mean
    return best


def score_cells(
    label: str,
    features: pd.DataFrame,
    labels: pd.Series,
    index: pd.DataFrame,
    factory,
    notes: str,
    results_path: Path | None = None,
    selector=None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit and score across the frozen folds, optionally selecting per fold."""
    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )
    indexed = index.set_index("filename")
    per_fold, collected, chosen = [], [], []
    started = time.time()

    for fold in fold_numbers:
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        make = factory
        if selector is not None:
            picked_label, make = selector(train_files)
            chosen.append(picked_label)

        model = make()
        model.fit(features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy())
        predicted = model.predict(features.loc[test_files].to_numpy())

        meta = indexed.loc[test_files].reset_index()
        evaluate(
            labels.loc[test_files].to_numpy(), predicted,
            "speaker_independent", fold, meta,
            model="logreg", feature_config=label,
            hyperparams={"lever": label},
            notes=notes, results_path=results_path,
        )
        per_fold.append((labels.loc[test_files].to_numpy(), predicted, meta))
        collected.append(pd.DataFrame({
            "y_true": labels.loc[test_files].to_numpy(),
            "y_pred": predicted,
            "speaker": meta["speaker_id"].to_numpy(),
        }))

    pooled = evaluate_pooled(
        per_fold, "speaker_independent",
        model="logreg", feature_config=label,
        hyperparams={"lever": label, "chosen": chosen or None},
        notes=notes + " (pooled out-of-fold)", results_path=results_path,
    )
    result = {
        "lever": label,
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "dim": features.shape[1],
        "chosen": ",".join(sorted(set(chosen))) if chosen else "",
        "seconds": round(time.time() - started, 1),
    }
    print(
        f"  {label:<38s} acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f} "
        f"d={result['dim']:<5d} {result['chosen']}",
        flush=True,
    )
    return result, pd.concat(collected, ignore_index=True)


def run(results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    features, labels, index = ravdess_matrix()
    print(f"baseline features: {features.shape}\n", flush=True)

    rows, kept = [], {}

    def add(label, feats, factory, notes, selector=None):
        result, preds = score_cells(
            label, feats, labels, index, factory, notes, results_path, selector
        )
        rows.append(result)
        kept[label] = preds

    # --- 0. baseline --------------------------------------------------------
    print("baseline (Phase 13 winner)")
    add("L6 logreg C=1 (baseline)", features, lambda: head(),
        "phase-14a baseline")

    # --- 1. regularisation --------------------------------------------------
    print("\nlever 1 -- regularisation (sweep is a curve; inner-CV is the claim)")
    for C in C_GRID:
        add(f"L6 logreg C={C:g}", features, (lambda c=C: head(c)),
            f"phase-14a regularisation sweep C={C:g}")

    def select_C(train_files):
        return inner_select(
            train_files, features, labels, index,
            [(f"C={c:g}", (lambda cc=c: head(cc))) for c in C_GRID],
        )

    add("L6 logreg C by inner CV", features, None,
        "phase-14a regularisation, C selected by inner CV", selector=select_C)

    # --- 2. dimensionality --------------------------------------------------
    print("\nlever 2 -- dimensionality reduction")
    for k in PCA_GRID:
        add(f"L6 PCA{k} + logreg", features,
            (lambda kk=k: Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=kk, random_state=SEED)),
                ("clf", head()),
            ])),
            f"phase-14a PCA {k}")
    add("L6 LDA + logreg", features,
        lambda: Pipeline([
            ("lda", LinearDiscriminantAnalysis(n_components=7)),
            ("clf", head()),
        ]),
        "phase-14a LDA 7")

    # --- 3. multi-layer -----------------------------------------------------
    print("\nlever 3 -- combining layers")
    for layers in ((5, 6, 7), (4, 5, 6, 7, 8)):
        multi, _, _ = ravdess_matrix(layers=layers)
        tag = "+".join(f"L{n}" for n in layers)
        add(f"{tag} concat + logreg", multi, lambda: head(),
            f"phase-14a multi-layer concat {tag}")

    # --- 4. prosody fusion --------------------------------------------------
    print("\nlever 4 -- prosody fusion (wav2vec2 + handcrafted)")
    prosody = prosody_matrix(index)
    fused = pd.concat([features, prosody.loc[features.index]], axis=1)
    add("L6 + handcrafted prosody", fused, lambda: head(),
        "phase-14a prosody fusion")

    # --- 5. class weighting -------------------------------------------------
    print("\nlever 5 -- balanced class weights")
    add("L6 logreg balanced", features, lambda: head(class_weight="balanced"),
        "phase-14a balanced class weights")

    table = pd.DataFrame(rows)
    table["vs_base_pp"] = ((table["macro_f1"] - BASE_F1) * 100).round(2)
    table["vs_floor_pp"] = ((table["macro_f1"] - FLOOR_F1) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # --- significance against the Phase 13 baseline -------------------------
    print("\npaired vs the Phase 13 baseline")
    baseline = kept["L6 logreg C=1 (baseline)"]
    comparisons = []
    for label, frame in kept.items():
        if label.endswith("(baseline)"):
            continue
        boot = paired_cluster_bootstrap(
            frame["y_true"], frame["y_pred"], baseline["y_pred"],
            frame["speaker"], "macro_f1",
        )
        comparisons.append({
            "lever": label,
            "delta_pp": round(100 * boot["delta"], 2),
            "lo_pp": round(100 * boot["lo"], 2),
            "hi_pp": round(100 * boot["hi"], 2),
            "p": round(boot["p_value"], 4),
            "significant": boot["significant"],
        })

    significance = pd.DataFrame(comparisons).sort_values("delta_pp", ascending=False)
    best = table.iloc[0]
    interval = cluster_bootstrap_ci(
        kept[best["lever"]]["y_true"], kept[best["lever"]]["y_pred"],
        kept[best["lever"]]["speaker"], "macro_f1",
    )
    print(
        f"\n  best: {best['lever']} = {best['macro_f1']:.4f} "
        f"[{interval['lo']:.4f}, {interval['hi']:.4f}]",
        flush=True,
    )
    wins = significance[significance["significant"] & (significance["delta_pp"] > 0)]
    print(f"  levers significantly better than the baseline: {len(wins)}", flush=True)

    return table, significance


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table, significance = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase14a_levers.parquet", index=False)
    significance.to_parquet(CACHE_DIR / "phase14a_significance.parquet", index=False)

    print("\n=== Phase 14a: model levers, frozen RAVDESS folds ===")
    print(table.to_string(index=False))
    print("\n=== paired vs Phase 13 baseline ===")
    print(significance.to_string(index=False))
