"""Attach uncertainty to the comparisons this project has been reporting.

`docs/merged_baseline_findings.md` reports differences of 11.11 pp, 5.01 pp and
0.42 pp as though they were the same kind of statement. They are not. With 24
test speakers, some of those gaps are far larger than the noise and some are
well inside it, and the write-up currently gives a reader no way to tell.

Every comparison here is **paired**: both models are scored on the identical
frozen test clips, so the bootstrap resamples the same speakers for both and
bootstraps the difference. Speakers, not clips, are the resampling unit --
`ser.stats` explains why at length, but briefly: 60 clips from one actor
succeed or fail together, so treating them as independent understates the
variance in the direction that makes weak results look strong.

McNemar's exact test is reported alongside for accuracy. It is sharper, because
it conditions on the clips the two models disagree about, but it assumes
independent clips, so it should be read as a complement to the cluster
bootstrap rather than a replacement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser import corpora
from ser.features import CACHE_DIR
from ser.run_early_read import prepare_merged
from ser.run_transfer import (
    WEIGHT_GRID,
    _fit_predict,
    _uniform_weights,
    importance_weights,
    scaler_only_predict,
)
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, mcnemar, paired_cluster_bootstrap

#: Fewer resamples than `ser.stats.N_BOOT` would be faster, but these intervals
#: are the ones quoted in the report, so they use the full default.
N_BOOT = 2000


def pooled_predictions(predict_fn, features, labels, indexed) -> pd.DataFrame:
    """Collect held-out predictions for every fold into one frame.

    Returns one row per test clip, with the true label, the prediction and the
    speaker -- everything `ser.stats` needs.
    """
    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )
    rows = []
    for fold in fold_numbers:
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        predicted = predict_fn(train_files, test_files)
        rows.append(
            pd.DataFrame(
                {
                    "filename": test_files,
                    "y_true": labels.loc[test_files].to_numpy(),
                    "y_pred": predicted,
                    "speaker": indexed.loc[test_files, "speaker_id"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    index = corpora.load_index()
    indexed, features, labels = prepare_merged(index)
    is_external = indexed["source"].astype(str) != "RAVDESS"
    external_files = indexed.index[is_external].tolist()

    def control(train, test):
        return _fit_predict(train, test, features, labels)

    def pooled(weight):
        def predict(train, test):
            full = train + external_files
            return _fit_predict(
                full, test, features, labels,
                _uniform_weights(full, is_external, weight),
            )
        return predict

    def importance(train, test):
        full = train + external_files
        return _fit_predict(
            full, test, features, labels,
            importance_weights(train, external_files, features),
        )

    def scaler_only(train, test):
        return scaler_only_predict(train, external_files, test, features, labels)

    strategies = {
        "control (ravdess only)": control,
        "scaler-only": scaler_only,
        f"uniform w={WEIGHT_GRID[-1]:g}": pooled(WEIGHT_GRID[-1]),
        "uniform w=0.05": pooled(0.05),
        "importance weighting": importance,
        "naive pooling (w=1)": pooled(1.0),
    }

    print("collecting pooled out-of-fold predictions ...", flush=True)
    predictions: dict[str, pd.DataFrame] = {}
    for name, fn in strategies.items():
        predictions[name] = pooled_predictions(fn, features, labels, indexed)
        print(f"  {name}", flush=True)

    # --- absolute intervals -------------------------------------------------
    print("\nabsolute macro-F1 with 95% cluster-bootstrap intervals")
    absolute = []
    for name, frame in predictions.items():
        for metric in ("macro_f1", "accuracy"):
            result = cluster_bootstrap_ci(
                frame["y_true"], frame["y_pred"], frame["speaker"],
                metric=metric, n_boot=N_BOOT,
            )
            absolute.append({"strategy": name, "metric": metric, **result})
        row = absolute[-2]
        print(
            f"  {name:<28s} macro-F1 {row['point']:.4f} "
            f"[{row['lo']:.4f}, {row['hi']:.4f}]  (se {row['se']:.4f})",
            flush=True,
        )

    # --- paired differences against the control ----------------------------
    print("\npaired differences vs the control")
    baseline = predictions["control (ravdess only)"]
    paired = []
    for name, frame in predictions.items():
        if name.startswith("control"):
            continue
        assert (frame["filename"].to_numpy() == baseline["filename"].to_numpy()).all(), (
            "paired test requires identical test clips in identical order"
        )
        boot = paired_cluster_bootstrap(
            frame["y_true"], frame["y_pred"], baseline["y_pred"], frame["speaker"],
            metric="macro_f1", n_boot=N_BOOT,
        )
        exact = mcnemar(frame["y_true"], frame["y_pred"], baseline["y_pred"])
        paired.append({
            "strategy": name,
            "delta_pp": round(100 * boot["delta"], 2),
            "lo_pp": round(100 * boot["lo"], 2),
            "hi_pp": round(100 * boot["hi"], 2),
            "boot_p": round(boot["p_value"], 4),
            "significant": boot["significant"],
            "mcnemar_p": round(exact["p_value"], 6),
            "only_new_right": exact["only_a_correct"],
            "only_control_right": exact["only_b_correct"],
        })
        verdict = "SIGNIFICANT" if boot["significant"] else "not distinguishable"
        print(
            f"  {name:<28s} {100*boot['delta']:+6.2f} pp "
            f"[{100*boot['lo']:+6.2f}, {100*boot['hi']:+6.2f}]  "
            f"boot p={boot['p_value']:.4f}  McNemar p={exact['p_value']:.2e}  "
            f"-> {verdict}",
            flush=True,
        )

    return pd.DataFrame(absolute), pd.DataFrame(paired)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    absolute, paired = run()
    CACHE_DIR.mkdir(exist_ok=True)
    absolute.to_parquet(CACHE_DIR / "phase12c_absolute_ci.parquet", index=False)
    paired.to_parquet(CACHE_DIR / "phase12c_paired.parquet", index=False)
    print("\n=== absolute ===")
    print(absolute.to_string(index=False))
    print("\n=== paired vs control ===")
    print(paired.to_string(index=False))
