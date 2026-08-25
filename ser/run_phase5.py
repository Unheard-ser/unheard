"""Phase 5 — classical baselines and tuning the winner.

Seven estimators on the Phase 4 winning feature config, each under both
protocols, then hyperparameter tuning on **the winner only** via CV *inside*
the training folds.

The point of this phase is not to find a good number. It is to establish the
floor that every later model -- CNN, LSTM, wav2vec2 -- must beat to justify
its complexity. A deep model that ties an SVM has not earned its place.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import (
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.metadata import load as load_metadata
from ser.models.classical import MODELS, build, default_hyperparams, tune
from ser.splits import get_split, load_folds

PROTOCOLS = ("speaker_independent", "random_stratified")

#: The Phase 4 winner. Fixed across all seven models so the comparison
#: isolates the estimator.
WINNER = FeatureConfig(
    normalisation="per_speaker",
    n_mfcc=20,
    extras=("chroma", "spectral_contrast", "spectral_centroid", "zcr"),
)


def prepare(config: FeatureConfig = WINNER):
    """Load features once, applying per-speaker normalisation if configured."""
    meta = load_metadata()
    frame = extract(config, meta)
    columns = feature_columns(frame, meta)
    indexed = frame.set_index("filename")

    values = indexed[columns].to_numpy()
    if config.normalisation == "per_speaker":
        values = per_speaker_normalise(values, indexed["actor"].to_numpy())

    features = pd.DataFrame(values, index=indexed.index, columns=columns)
    return meta, indexed, features, indexed["emotion_label"]


def score_model(
    name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    indexed: pd.DataFrame,
    folds: pd.DataFrame,
    protocol: str,
    config_name: str,
    hyperparams: dict | None = None,
    pipeline_factory=None,
    notes: str = "phase-5 baseline, default hyperparameters",
    results_path: Path | None = None,
) -> dict[str, float]:
    """Fit and score one model across every fold of one protocol."""
    fold_numbers = sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())
    per_fold, started = [], time.time()

    for fold in fold_numbers:
        train_files, test_files = get_split(protocol, fold, folds=folds)
        pipeline = (pipeline_factory or (lambda: build(name)))()
        pipeline.fit(features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy())
        predicted = pipeline.predict(features.loc[test_files].to_numpy())

        test_meta = indexed.loc[test_files].reset_index()
        evaluate(
            labels.loc[test_files].to_numpy(), predicted, protocol, fold, test_meta,
            model=name, feature_config=config_name,
            hyperparams=hyperparams or default_hyperparams(name),
            notes=notes, results_path=results_path,
        )
        per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))

    pooled = evaluate_pooled(
        per_fold, protocol,
        model=name, feature_config=config_name,
        hyperparams=hyperparams or default_hyperparams(name),
        notes=notes + " (pooled out-of-fold)", results_path=results_path,
    )
    accuracies = [
        float((true == pred).mean()) for true, pred, _ in per_fold
    ]
    return {
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "fold_sd": float(np.std(accuracies, ddof=1)),
        "seconds": round(time.time() - started, 1),
    }


def run(results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all seven models, then tune the winner.

    Returns:
        ``(comparison, tuned)`` -- the seven-model table and the tuned-winner
        rows.
    """
    set_seeds()
    meta, indexed, features, labels = prepare()
    folds = load_folds()
    print(f"features: {WINNER.name}\n{len(features)} clips x {features.shape[1]}\n")

    rows = []
    for name in MODELS:
        for protocol in PROTOCOLS:
            scored = score_model(
                name, features, labels, indexed, folds, protocol,
                WINNER.name, results_path=results_path,
            )
            rows.append({"model": name, "protocol": protocol, **scored})
            print(
                f"  {name:20s} {protocol:20s} acc={scored['accuracy']:.4f} "
                f"f1={scored['macro_f1']:.4f} sd={scored['fold_sd']:.4f} "
                f"({scored['seconds']}s)"
            )

    comparison = pd.DataFrame(rows)
    speaker_independent = comparison[
        comparison["protocol"] == "speaker_independent"
    ].sort_values("macro_f1", ascending=False)
    winner = str(speaker_independent.iloc[0]["model"])
    print(f"\nwinner on speaker_independent: {winner}\n")

    tuned_rows = []
    for protocol in PROTOCOLS:
        fold_numbers = sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())
        per_fold, best_params, started = [], [], time.time()

        for fold in fold_numbers:
            train_files, test_files = get_split(protocol, fold, folds=folds)
            x_train = features.loc[train_files].to_numpy()
            y_train = labels.loc[train_files].to_numpy()
            # Grouped inner CV: the search itself must not leak across
            # speakers, or it picks hyperparameters that only look good.
            groups = indexed.loc[train_files, "actor"].to_numpy()

            search = tune(winner, x_train, y_train, groups=groups)
            best_params.append(search.best_params_)
            predicted = search.best_estimator_.predict(features.loc[test_files].to_numpy())

            test_meta = indexed.loc[test_files].reset_index()
            evaluate(
                labels.loc[test_files].to_numpy(), predicted, protocol, fold, test_meta,
                model=f"{winner}-tuned", feature_config=WINNER.name,
                hyperparams=search.best_params_,
                notes="phase-5 tuned, grid search inside training folds",
                results_path=results_path,
            )
            per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))
            print(f"  tuned {winner} {protocol} fold {fold}: {search.best_params_}")

        pooled = evaluate_pooled(
            per_fold, protocol,
            model=f"{winner}-tuned", feature_config=WINNER.name,
            hyperparams={"per_fold_best": [str(p) for p in best_params]},
            notes="phase-5 tuned (pooled out-of-fold)", results_path=results_path,
        )
        tuned_rows.append({
            "model": f"{winner}-tuned", "protocol": protocol,
            "accuracy": pooled["accuracy"], "macro_f1": pooled["macro_f1"],
            "seconds": round(time.time() - started, 1),
        })
        print(
            f"  {winner}-tuned {protocol} POOLED acc={pooled['accuracy']:.4f} "
            f"f1={pooled['macro_f1']:.4f}\n"
        )

    return comparison, pd.DataFrame(tuned_rows)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    comparison, tuned = run(results_path=args.results)
    out = Path("features")
    out.mkdir(exist_ok=True)
    comparison.to_parquet(out / "phase5_comparison.parquet", index=False)
    tuned.to_parquet(out / "phase5_tuned.parquet", index=False)

    print("\n=== seven-model comparison ===")
    print(comparison.sort_values("macro_f1", ascending=False).to_string(index=False))
    print("\n=== tuned winner ===")
    print(tuned.to_string(index=False))
