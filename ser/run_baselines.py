"""Run the classical baselines across protocols and folds.

Drives one feature config through every (model, protocol, fold) combination,
scoring each through `ser.evaluate.evaluate` so every run lands in
``results/results.csv`` (CLAUDE.md rule 5).

Defaults only — no tuning. This establishes the floor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser.evaluate import evaluate, evaluate_pooled
from ser.features import (
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.metadata import load as load_metadata
from ser.models.classical import build, default_hyperparams
from ser.splits import get_split, load_folds

PROTOCOLS = ("speaker_independent", "random_stratified")
MODELS = ("svm-rbf", "random-forest")


def run(
    config: FeatureConfig | None = None,
    models: tuple[str, ...] = MODELS,
    protocols: tuple[str, ...] = PROTOCOLS,
    notes: str = "phase-5 floor, default hyperparameters",
    results_path: Path | None = None,
) -> pd.DataFrame:
    """Fit and score every (model, protocol, fold), logging each run.

    Args:
        config: Feature configuration. Defaults to `FeatureConfig()`.
        models: Model names from `ser.models.classical.MODELS`.
        protocols: Split protocol names.
        notes: Free text recorded on every row.
        results_path: Override the results log. Tests must pass a temp path.

    Returns:
        One row per (model, protocol, fold) plus one pooled row each, with
        accuracy and macro-F1.
    """
    config = config or FeatureConfig()
    meta = load_metadata()
    frame = extract(config, meta)
    columns = feature_columns(frame, meta)

    indexed = frame.set_index("filename")
    values = indexed[columns].to_numpy()
    if config.normalisation == "per_speaker":
        # Documented exception to rule 2 -- see ser.features.per_speaker_normalise.
        values = per_speaker_normalise(values, indexed["actor"].to_numpy())
    x_all = pd.DataFrame(values, index=indexed.index, columns=columns)
    y_all = indexed["emotion_label"]
    folds = load_folds()

    summary: list[dict[str, object]] = []

    for model_name in models:
        hyperparams = default_hyperparams(model_name)
        for protocol in protocols:
            fold_numbers = sorted(
                folds.loc[folds["protocol"] == protocol, "fold"].unique()
            )
            pooled_inputs = []

            for fold in fold_numbers:
                train_files, test_files = get_split(protocol, fold, folds=folds)

                x_train = x_all.loc[train_files].to_numpy()
                y_train = y_all.loc[train_files].to_numpy()
                x_test = x_all.loc[test_files].to_numpy()
                y_test = y_all.loc[test_files].to_numpy()

                started = time.time()
                pipeline = build(model_name)
                # Scaler is fit here, on the training fold only.
                pipeline.fit(x_train, y_train)
                y_pred = pipeline.predict(x_test)
                elapsed = time.time() - started

                test_meta = indexed.loc[test_files].reset_index()
                result = evaluate(
                    y_test, y_pred, protocol, fold, test_meta,
                    model=model_name,
                    feature_config=config.name,
                    hyperparams=hyperparams,
                    notes=notes,
                    results_path=results_path,
                )
                pooled_inputs.append((y_test, y_pred, test_meta))
                summary.append(
                    {
                        "model": model_name,
                        "protocol": protocol,
                        "fold": fold,
                        "n_test": len(y_test),
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "seconds": round(elapsed, 1),
                    }
                )
                print(
                    f"  {model_name:14s} {protocol:20s} fold {fold} "
                    f"acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f} "
                    f"({elapsed:.1f}s)"
                )

            pooled = evaluate_pooled(
                pooled_inputs, protocol,
                model=model_name,
                feature_config=config.name,
                hyperparams=hyperparams,
                notes=notes + " (pooled out-of-fold)",
                results_path=results_path,
            )
            summary.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "fold": "pooled",
                    "n_test": pooled["n"],
                    "accuracy": pooled["accuracy"],
                    "macro_f1": pooled["macro_f1"],
                    "seconds": None,
                }
            )
            print(
                f"  {model_name:14s} {protocol:20s} POOLED "
                f"acc={pooled['accuracy']:.4f} f1={pooled['macro_f1']:.4f}\n"
            )

    return pd.DataFrame(summary)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    config = FeatureConfig()
    print(f"features: {config.name} (hash {config.hash})\n")
    table = run(config, results_path=args.results)

    out = Path("features") / "baseline_summary.parquet"
    out.parent.mkdir(exist_ok=True)
    # fold mixes ints with the literal "pooled"; parquet needs one type.
    table.assign(fold=table["fold"].astype(str)).to_parquet(out, index=False)
    print(table.to_string(index=False))
