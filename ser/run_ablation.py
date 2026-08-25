"""Phase 4 — the feature ablation.

**One axis at a time** from a fixed baseline, with **one fixed model**
(SVM-RBF) so every difference is attributable to the features rather than to
the classifier. A full cartesian grid over six axes would be 200+ configs and
would tell us less: with one-at-a-time, each row answers exactly one
question.

Run under ``speaker_independent`` only -- the primary protocol. The winner is
re-run under both protocols afterwards, by `ser.run_baselines`.

Every run is scored through `ser.evaluate.evaluate` and lands in
``results/results.csv`` (CLAUDE.md rule 5).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser.evaluate import evaluate_pooled
from ser.features import (
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.metadata import load as load_metadata
from ser.models.classical import build
from ser.splits import get_split, load_folds

#: The point we vary around. Everything else in the grid is this, with one
#: field changed.
BASELINE = FeatureConfig()

#: Speech-conventional framing, the change this phase was prompted by.
SPEECH_FRAMING = {"n_fft": 400, "hop_length": 160}      # 25 ms / 10 ms
MEDIUM_FRAMING = {"n_fft": 640, "hop_length": 160}      # 40 ms / 10 ms


def build_grid(baseline: FeatureConfig = BASELINE) -> list[tuple[str, FeatureConfig]]:
    """One config per question, each differing from ``baseline`` in one axis.

    Returns:
        ``(axis, config)`` pairs. ``axis`` names which question the row
        answers, so `summarise` can group by it.
    """
    grid: list[tuple[str, FeatureConfig]] = [("baseline", baseline)]

    for label, framing in (("25ms/10ms", SPEECH_FRAMING), ("40ms/10ms", MEDIUM_FRAMING)):
        grid.append(("framing", replace(baseline, **framing)))
    for n_mfcc in (13, 20):
        grid.append(("n_mfcc", replace(baseline, n_mfcc=n_mfcc)))
    grid.append(("deltas", replace(baseline, deltas=False, delta_deltas=False)))
    grid.append(("deltas", replace(baseline, deltas=True, delta_deltas=False)))
    for aggregation in ("mean", "mean_std_min_max", "percentiles"):
        grid.append(("aggregation", replace(baseline, aggregation=aggregation)))
    for normalisation in ("global", "per_speaker"):
        grid.append(("normalisation", replace(baseline, normalisation=normalisation)))
    grid.append(("trim", replace(baseline, trim=False)))
    for extras in (("chroma",), ("spectral_contrast",),
                   ("chroma", "spectral_contrast", "spectral_centroid", "zcr")):
        grid.append(("extras", replace(baseline, extras=extras)))
    for n_mels in (40, 80):
        grid.append(("n_mels", replace(baseline, n_mels=n_mels)))
    grid.append(("fmin", replace(baseline, fmin=50.0)))

    return grid


def score_config(
    config: FeatureConfig,
    meta: pd.DataFrame,
    folds: pd.DataFrame,
    model: str = "svm-rbf",
    protocol: str = "speaker_independent",
    notes: str = "phase-4 ablation",
    results_path: Path | None = None,
) -> dict[str, object]:
    """Extract, fit and score one configuration across all folds.

    Scaling happens inside the sklearn `Pipeline`, so it is fit on the
    training fold only. Per-speaker normalisation is the documented
    exception -- see `per_speaker_normalise`.

    Args:
        config: The configuration to evaluate.
        meta: Metadata frame.
        folds: Frozen fold table.
        model: Model name; fixed across the grid so features are isolated.
        protocol: Split protocol.
        notes: Recorded on every results row.
        results_path: Override the results log (tests only).

    Returns:
        Accuracy, macro-F1, feature count and wall time.
    """
    frame = extract(config, meta)
    columns = feature_columns(frame, meta)
    indexed = frame.set_index("filename")

    values = indexed[columns].to_numpy()
    if config.normalisation == "per_speaker":
        # Applied across the whole matrix, deliberately: a held-out speaker
        # has no training clips, so their statistics can only come from their
        # own audio. No labels are touched.
        values = per_speaker_normalise(values, indexed["actor"].to_numpy())
    scaled = pd.DataFrame(values, index=indexed.index, columns=columns)

    labels = indexed["emotion_label"]
    fold_numbers = sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())

    started = time.time()
    per_fold = []
    for fold in fold_numbers:
        train_files, test_files = get_split(protocol, fold, folds=folds)
        pipeline = build(model)
        pipeline.fit(scaled.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy())
        predicted = pipeline.predict(scaled.loc[test_files].to_numpy())
        per_fold.append(
            (labels.loc[test_files].to_numpy(), predicted, indexed.loc[test_files].reset_index())
        )

    pooled = evaluate_pooled(
        per_fold, protocol,
        model=model, feature_config=config.name,
        hyperparams={"n_features": len(columns)},
        notes=notes, results_path=results_path,
    )
    return {
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "n_features": len(columns),
        "seconds": round(time.time() - started, 1),
    }


def run(
    baseline: FeatureConfig = BASELINE,
    results_path: Path | None = None,
) -> pd.DataFrame:
    """Run the whole grid and return it ranked.

    Returns:
        One row per config: axis, name, feature count, accuracy, macro-F1,
        and the delta against the baseline.
    """
    set_seeds()
    meta = load_metadata()
    folds = load_folds()
    grid = build_grid(baseline)

    rows: list[dict[str, object]] = []
    for index, (axis, config) in enumerate(grid, start=1):
        scored = score_config(config, meta, folds, results_path=results_path)
        rows.append({"axis": axis, "config": config.name,
                     "window_ms": config.window_ms, "hop_ms": config.hop_ms, **scored})
        print(
            f"[{index:2d}/{len(grid)}] {axis:14s} {config.name[:44]:44s} "
            f"acc={scored['accuracy']:.4f} f1={scored['macro_f1']:.4f} "
            f"({scored['n_features']:3d}f, {scored['seconds']}s)"
        )

    table = pd.DataFrame(rows)
    base_accuracy = float(table.loc[table["axis"] == "baseline", "accuracy"].iloc[0])
    base_f1 = float(table.loc[table["axis"] == "baseline", "macro_f1"].iloc[0])
    table["d_acc_pp"] = ((table["accuracy"] - base_accuracy) * 100).round(2)
    table["d_f1_pp"] = ((table["macro_f1"] - base_f1) * 100).round(2)
    return table.sort_values("macro_f1", ascending=False).reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    print(f"baseline: {BASELINE.name}\n")
    ranked = run(results_path=args.results)

    out = Path("features") / "ablation.parquet"
    out.parent.mkdir(exist_ok=True)
    ranked.to_parquet(out, index=False)
    print("\n" + ranked.to_string(index=False))
