"""Phase 13b control — isolate the feature source from the classifier head.

Phase 12 measured the cost of adding external corpora using **SVM-RBF on
handcrafted features**. Phase 13b measures it using **LinearSVC on wav2vec2
embeddings**. Comparing those two directly, as an early reading of 13b did,
conflates two changes: the features got better *and* the head changed.

That conflation is not hypothetical. On identical wav2vec2 features, a
multinomial logistic head scores 0.7944 and a one-vs-rest LinearSVC scores
0.7080 -- an 8.6 pp gap between two linear models, with no convergence failure
in either. A head swap of that size can plausibly move a *delta* too.

So this runs the Phase 12 question again on handcrafted features with the
**LinearSVC head from Phase 13b**. Then:

    handcrafted + LinearSVC   vs   wav2vec2 + LinearSVC

differ in exactly one thing, and the claim "better representations absorb
domain shift better" becomes testable rather than suggestive.

Cheap: handcrafted features are 162-dimensional against wav2vec2's 768, so
these fits are much faster than 13b's.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser import corpora
from ser.evaluate import evaluate_pooled
from ser.features import CACHE_DIR
from ser.run_early_read import prepare_merged
from ser.run_phase5 import WINNER as HANDCRAFTED
from ser.run_wav2vec2_transfer import linear_head
from ser.splits import get_split, load_folds
from ser.stats import paired_cluster_bootstrap

#: The same cells Phase 13b runs, so the two tables line up row for row.
CELLS: tuple[tuple[str, ...], ...] = (
    (),
    ("CREMA-D",),
    ("TESS",),
    ("SAVEE",),
    ("CREMA-D", "TESS", "SAVEE"),
)


def run(results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    index = corpora.load_index()
    indexed, features, labels = prepare_merged(index)
    source = indexed["source"].astype(str)
    print(f"handcrafted: {features.shape[0]:,} x {features.shape[1]}\n", flush=True)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )

    kept: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for sources in CELLS:
        label = "+".join(sources) if sources else "(ravdess only)"
        extra = indexed.index[source.isin(sources)].tolist() if sources else []
        started = time.time()
        per_fold, collected = [], []

        for fold in fold_numbers:
            train_files, test_files = get_split("speaker_independent", fold, folds=folds)
            estimator = linear_head()
            estimator.fit(
                features.loc[train_files + extra].to_numpy(),
                labels.loc[train_files + extra].to_numpy(),
            )
            predicted = estimator.predict(features.loc[test_files].to_numpy())
            meta = indexed.loc[test_files].reset_index()
            per_fold.append((labels.loc[test_files].to_numpy(), predicted, meta))
            collected.append(pd.DataFrame({
                "y_true": labels.loc[test_files].to_numpy(),
                "y_pred": predicted,
                "speaker": meta["speaker_id"].to_numpy(),
            }))

        pooled = evaluate_pooled(
            per_fold, "speaker_independent",
            model="linear-svc", feature_config=HANDCRAFTED.name,
            hyperparams={"max_iter": 2000, "head": "LinearSVC"},
            notes=f"phase-13b control: handcrafted + LinearSVC, train += {label} "
                  f"(pooled out-of-fold)",
            results_path=results_path,
        )
        kept[label] = pd.concat(collected, ignore_index=True)
        rows.append({
            "added": label,
            "accuracy": pooled["accuracy"],
            "macro_f1": pooled["macro_f1"],
            "seconds": round(time.time() - started, 1),
        })
        print(
            f"  {label:<26s} acc={pooled['accuracy']:.4f} f1={pooled['macro_f1']:.4f} "
            f"({rows[-1]['seconds']}s)",
            flush=True,
        )

    table = pd.DataFrame(rows)
    control = float(table.loc[table["added"] == "(ravdess only)", "macro_f1"].iloc[0])
    table["vs_control_pp"] = ((table["macro_f1"] - control) * 100).round(2)

    print("\npaired differences vs the control (handcrafted + LinearSVC)")
    baseline = kept["(ravdess only)"]
    comparisons = []
    for label, frame in kept.items():
        if label == "(ravdess only)":
            continue
        boot = paired_cluster_bootstrap(
            frame["y_true"], frame["y_pred"], baseline["y_pred"], frame["speaker"],
            "macro_f1",
        )
        comparisons.append({
            "added": label,
            "delta_pp": round(100 * boot["delta"], 2),
            "lo_pp": round(100 * boot["lo"], 2),
            "hi_pp": round(100 * boot["hi"], 2),
            "p": round(boot["p_value"], 4),
            "significant": boot["significant"],
        })
        print(
            f"  {label:<26s} {100*boot['delta']:+6.2f} pp "
            f"[{100*boot['lo']:+6.2f}, {100*boot['hi']:+6.2f}] p={boot['p_value']:.4f}",
            flush=True,
        )

    return table, pd.DataFrame(comparisons)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table, comparisons = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase13b_head_control.parquet", index=False)
    comparisons.to_parquet(CACHE_DIR / "phase13b_head_control_paired.parquet", index=False)
    print("\n=== handcrafted + LinearSVC (the like-for-like control) ===")
    print(table.to_string(index=False))
    print("\n=== paired vs control ===")
    print(comparisons.to_string(index=False))
