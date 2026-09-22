"""Phase 13 — frozen wav2vec2 embeddings, the sweep that closes Phase 6d.

One `ser.embeddings.extract` pass caches the pooled statistics of all 13 hidden
states for every clip, so everything below is a slice of a memory-mapped file
rather than another forward pass. That is what makes a four-axis sweep
affordable on CPU.

The axes, one at a time, in the Phase 4 style
---------------------------------------------
1. **layer** -- which hidden state carries emotion. Middle layers routinely beat
   the last on paralinguistic tasks, because the last specialises toward the
   phonetic content an ASR-style objective rewards. If that holds here it is a
   finding worth a figure.
2. **pooling** -- mean, standard deviation, or both. Emotion is partly *variation*
   over a clip, so the std channel may carry signal the mean discards.
3. **normalisation** -- none / global / per-speaker. Per-speaker z-scoring bought
   +6.53 pp on handcrafted features and cut actor identifiability from 92.7% to
   21.2%. Whether it still helps when the features are learned is an open
   question: a self-supervised encoder may already have factored speaker out.
4. **head** -- the same seven estimators Phase 5 used, so the comparison against
   the 0.6311 floor isolates the features rather than the classifier.

Protocol
--------
Architecture A throughout (CLAUDE.md rule 9): the frozen RAVDESS
`speaker_independent` folds, RAVDESS-only training. Every number here is
therefore directly comparable to the handcrafted floor of **0.6389 accuracy /
0.6311 macro-F1**, and to `docs/merged_baseline_findings.md`.

Whether the external corpora help at *this* feature resolution is asked
separately, in Phase 13b -- it is a different question and deserves its own
controls.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser import corpora
from ser.embeddings import EmbeddingConfig, extract, feature_names, select
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import per_speaker_normalise
from ser.models.classical import MODELS, build, default_hyperparams
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

#: The handcrafted floor every cell is measured against (Phase 5, untuned).
FLOOR_ACC, FLOOR_F1 = 0.638889, 0.631106

LAYERS: tuple[int, ...] = (0, 3, 6, 9, 12)
POOLINGS: tuple[str, ...] = ("mean", "std", "mean_std")
NORMALISATIONS: tuple[str, ...] = ("none", "global", "per_speaker")


def load_cached(config: EmbeddingConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the cached per-layer tensor and the merged index it aligns to."""
    index = corpora.load_index()
    tensor, df = extract(config, index, use_cache=True, progress=False)
    return tensor, df


def ravdess_view(
    tensor: np.ndarray, index: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Restrict the cached tensor to RAVDESS rows.

    One extraction pass covers the whole merged corpus; Architecture A needs
    only its RAVDESS subset, and slicing is free.
    """
    mask = (index["source"].astype(str) == "RAVDESS").to_numpy()
    return np.asarray(tensor[mask]), index.loc[mask].reset_index(drop=True)


def build_matrix(
    tensor: np.ndarray,
    index: pd.DataFrame,
    config: EmbeddingConfig,
    layer_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice the tensor to a feature frame, applying per-speaker normalisation.

    ``per_speaker`` is applied here because it is not a fold-level fit at all --
    see the stated exception in CLAUDE.md rule 2. ``global`` is a train-fold fit
    and is applied inside the fold loop in `score`.
    """
    matrix = select(tensor, config, layer_weights=layer_weights)
    columns = feature_names(matrix.shape[1], config.pooling)

    if config.normalisation == "per_speaker":
        matrix = per_speaker_normalise(matrix, index["speaker_id"].to_numpy())

    frame = pd.DataFrame(matrix, index=index["filename"], columns=columns)
    return frame, pd.Series(
        index["emotion_label"].to_numpy(), index=index["filename"], name="emotion_label"
    )


def head(model: str, standardise: bool):
    """The estimator, with standardisation under this sweep's control.

    `ser.models.classical.build` wraps every estimator in a `StandardScaler`.
    That is right for Phase 5, where scaling was a fixed decision, but it makes
    a *normalisation axis* impossible to measure: with it in place, ``none``
    still standardises and ``global`` standardises twice, so all three levels
    collapse to the same thing.

    So the scaler is stripped here and reintroduced only when the axis asks for
    it. ``standardise=True`` reproduces `build` exactly.

    This matters more for embeddings than it did for MFCCs: wav2vec2 dimensions
    differ in scale by orders of magnitude, and SVM-RBF is scale-sensitive, so
    "no standardisation" is a genuinely different model rather than a rounding
    difference.

    Args:
        model: A key of `ser.models.classical.MODELS`.
        standardise: Prepend a train-fold `StandardScaler`.

    Returns:
        An unfitted estimator or pipeline.
    """
    pipeline = build(model)
    return pipeline if standardise else pipeline.named_steps["estimator"]


def score(
    features: pd.DataFrame,
    labels: pd.Series,
    indexed: pd.DataFrame,
    config: EmbeddingConfig,
    model: str = "svm-rbf",
    protocol: str = "speaker_independent",
    notes: str = "phase-13",
    results_path: Path | None = None,
) -> dict[str, float]:
    """Fit and score one configuration across every fold of one protocol.

    Standardisation is driven entirely by ``config.normalisation``:

    ``none``
        raw embeddings straight into the estimator
    ``global``
        train-fold `StandardScaler`, i.e. exactly what Phase 5 did
    ``per_speaker``
        per-speaker z-score only, applied in `build_matrix`

    so the three levels are genuinely distinct and the axis measures something.
    """
    folds = load_folds()
    fold_numbers = sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())
    lookup = indexed.set_index("filename")
    per_fold, started = [], time.time()
    standardise = config.normalisation == "global"

    for fold in fold_numbers:
        train_files, test_files = get_split(protocol, fold, folds=folds)
        x_train = features.loc[train_files].to_numpy()
        x_test = features.loc[test_files].to_numpy()

        estimator = head(model, standardise)
        estimator.fit(x_train, labels.loc[train_files].to_numpy())
        predicted = estimator.predict(x_test)

        test_meta = lookup.loc[test_files].reset_index()
        evaluate(
            labels.loc[test_files].to_numpy(), predicted, protocol, fold, test_meta,
            model=model, feature_config=config.name,
            hyperparams=default_hyperparams(model),
            notes=notes, results_path=results_path,
        )
        per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))

    pooled = evaluate_pooled(
        per_fold, protocol,
        model=model, feature_config=config.name,
        hyperparams=default_hyperparams(model),
        notes=notes + " (pooled out-of-fold)", results_path=results_path,
    )
    accuracies = [float((t == p).mean()) for t, p, _ in per_fold]
    # Kept so the winning cell can be compared against the handcrafted floor
    # with a paired test rather than by eyeballing two point estimates.
    predictions = pd.DataFrame(
        {
            "y_true": np.concatenate([t for t, _, _ in per_fold]),
            "y_pred": np.concatenate([p for _, p, _ in per_fold]),
            "speaker": np.concatenate([m["speaker_id"].to_numpy() for _, _, m in per_fold]),
        }
    )
    return {
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "fold_sd": float(np.std(accuracies, ddof=1)),
        "predictions": predictions,
        "dim": features.shape[1],
        "seconds": round(time.time() - started, 1),
    }


def run(results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    base = EmbeddingConfig(batch_size=8)
    print(f"loading cached embeddings: {base.cache_path.name}", flush=True)
    tensor, index = load_cached(base)
    tensor, index = ravdess_view(tensor, index)
    print(f"  RAVDESS view: {tensor.shape} (clips, layers, [mean,std], dim)\n", flush=True)

    rows: list[dict[str, object]] = []
    kept: dict[str, pd.DataFrame] = {}

    def cell(config: EmbeddingConfig, axis: str, model: str = "svm-rbf") -> float:
        features, labels = build_matrix(tensor, index, config)
        result = score(
            features, labels, index, config, model=model,
            notes=f"phase-13 {axis}: {config.name}", results_path=results_path,
        )
        key = f"{axis}|{config.name}|{model}"
        kept[key] = result.pop("predictions")
        rows.append({"axis": axis, "config": config.name, "model": model,
                     "key": key, **result})
        print(
            f"  {config.name:<44s} {model:<20s} acc={result['accuracy']:.4f} "
            f"f1={result['macro_f1']:.4f} d={result['dim']:<5d} ({result['seconds']}s)",
            flush=True,
        )
        return result["macro_f1"]

    # Axes 1 and 2 are swept at ``global`` normalisation -- a train-fold
    # StandardScaler, exactly what Phase 5 used. Sweeping them at ``none``
    # would feed raw embeddings, whose dimensions differ in scale by orders of
    # magnitude, into a scale-sensitive SVM-RBF; every layer would score badly
    # and the comparison would measure the missing scaler rather than the
    # layer. Normalisation is then swept in its own right as axis 3.
    fixed = "global"

    # --- axis 1: layer ------------------------------------------------------
    print(f"axis 1 -- layer (mean pooling, {fixed} normalisation)")
    scores = {
        layer: cell(
            EmbeddingConfig(layer=layer, normalisation=fixed), "layer"
        )
        for layer in LAYERS
    }
    best_layer = max(scores, key=scores.get)
    print(f"  -> best layer {best_layer}\n", flush=True)

    # --- axis 2: pooling ----------------------------------------------------
    print("axis 2 -- pooling")
    scores = {
        pooling: cell(
            EmbeddingConfig(layer=best_layer, pooling=pooling, normalisation=fixed),
            "pooling",
        )
        for pooling in POOLINGS
    }
    best_pooling = max(scores, key=scores.get)
    print(f"  -> best pooling {best_pooling}\n", flush=True)

    # --- axis 3: normalisation ---------------------------------------------
    print("axis 3 -- normalisation")
    scores = {
        norm: cell(
            EmbeddingConfig(layer=best_layer, pooling=best_pooling, normalisation=norm),
            "normalisation",
        )
        for norm in NORMALISATIONS
    }
    best_norm = max(scores, key=scores.get)
    print(f"  -> best normalisation {best_norm}\n", flush=True)

    # --- axis 4: head -------------------------------------------------------
    print("axis 4 -- head")
    winner = EmbeddingConfig(
        layer=best_layer, pooling=best_pooling, normalisation=best_norm
    )
    for model in MODELS:
        cell(winner, "head", model=model)

    table = pd.DataFrame(rows)
    table["vs_floor_pp"] = ((table["macro_f1"] - FLOOR_F1) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # --- is the winner actually distinguishable from anything? --------------
    # With a +/-4.5 pp interval on 24 test speakers, picking the best of five
    # layers on point estimates alone risks selecting noise. Every cell is
    # compared against the best one, paired on the same speakers.
    print("\nsignificance: every cell vs the best, paired cluster bootstrap")
    best_key = str(table.loc[0, "key"])
    best_pred = kept[best_key]
    comparisons = []
    for row in table.itertuples():
        if row.key == best_key:
            continue
        other = kept[row.key]
        result = paired_cluster_bootstrap(
            best_pred["y_true"], best_pred["y_pred"], other["y_pred"],
            best_pred["speaker"], metric="macro_f1",
        )
        comparisons.append({
            "config": row.config, "model": row.model,
            "delta_pp": round(100 * result["delta"], 2),
            "lo_pp": round(100 * result["lo"], 2),
            "hi_pp": round(100 * result["hi"], 2),
            "p": round(result["p_value"], 4),
            "distinguishable": result["significant"],
        })
    significance = pd.DataFrame(comparisons)
    indistinguishable = significance[~significance["distinguishable"]]
    print(
        f"  best cell: {table.loc[0, 'config']} / {table.loc[0, 'model']} "
        f"= {table.loc[0, 'macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"  {len(indistinguishable)} of {len(significance)} other cells are NOT "
        f"distinguishable from it",
        flush=True,
    )

    # --- and against the handcrafted floor ----------------------------------
    interval = cluster_bootstrap_ci(
        best_pred["y_true"], best_pred["y_pred"], best_pred["speaker"],
        metric="macro_f1",
    )
    print(
        f"  winner macro-F1 {interval['point']:.4f} "
        f"[{interval['lo']:.4f}, {interval['hi']:.4f}] "
        f"vs handcrafted floor {FLOOR_F1:.4f}",
        flush=True,
    )

    return table, significance


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table, significance = run(results_path=args.results)
    out = Path("features")
    out.mkdir(exist_ok=True)
    table.drop(columns=["key"]).to_parquet(out / "phase13_sweep.parquet", index=False)
    significance.to_parquet(out / "phase13_significance.parquet", index=False)

    print("\n=== wav2vec2 sweep, frozen RAVDESS folds ===")
    print(table.drop(columns=["key"]).to_string(index=False))
    print("\n=== every cell vs the best (paired) ===")
    print(significance.to_string(index=False))
    print(f"\nhandcrafted floor: {FLOOR_ACC:.4f} acc / {FLOOR_F1:.4f} macro-F1")
