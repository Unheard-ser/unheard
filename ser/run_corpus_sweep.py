"""Phase 12 (first cut): *which* external corpora help, if any?

Phase 9.5 established that adding all three external corpora to training costs
11.33 pp macro-F1 on the RAVDESS test set, and that the loss is concentrated
in `calm` (-36.5 pp) and `surprised` (-28.7 pp) -- precisely the two classes
CREMA-D does not contain. Meanwhile `neutral` *gained* 14.6 pp and `fearful`
8.3 pp.

That pattern says the damage is prior shift rather than domain shift: 7,442
CREMA-D clips that never exhibit calm or surprise teach the model those classes
are rare. If that reading is right, the corpora should behave very differently
from one another, and the subset matters more than the volume.

So the subset is swept rather than assumed. Every run uses the frozen RAVDESS
test folds (Architecture A), so all seven rows are directly comparable to each
other and to the 0.6311 macro-F1 control.
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser import corpora
from ser.evaluate import evaluate_pooled
from ser.features import CACHE_DIR
from ser.models.classical import build, default_hyperparams
from ser.run_early_read import EXTERNAL, MODEL, prepare_merged
from ser.run_phase5 import WINNER
from ser.splits import load_folds
from ser.splits_merged import augmented_train


def subsets() -> list[tuple[str, ...]]:
    """Every subset of the external corpora, smallest first."""
    out: list[tuple[str, ...]] = [()]
    for size in range(1, len(EXTERNAL) + 1):
        out.extend(combinations(EXTERNAL, size))
    return out


def run(results_path: Path | None = None) -> pd.DataFrame:
    set_seeds()
    index = corpora.load_index()
    indexed, features, labels = prepare_merged(index)
    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )

    rows = []
    for sources in subsets():
        started = time.time()
        per_fold = []
        for fold in fold_numbers:
            train_files, test_files = augmented_train(fold, sources=sources, index=index)
            pipeline = build(MODEL)
            pipeline.fit(
                features.loc[train_files].to_numpy(),
                labels.loc[train_files].to_numpy(),
            )
            predicted = pipeline.predict(features.loc[test_files].to_numpy())
            test_meta = indexed.loc[test_files].reset_index()
            per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))

        label = "+".join(sources) if sources else "(ravdess only)"
        pooled = evaluate_pooled(
            per_fold, "speaker_independent",
            model=MODEL, feature_config=WINNER.name,
            hyperparams=default_hyperparams(MODEL),
            notes=f"phase-12 corpus sweep: train += {label} (pooled out-of-fold)",
            results_path=results_path,
        )
        n_train = len(augmented_train(0, sources=sources, index=index)[0])
        rows.append(
            {
                "added": label,
                "n_train_fold0": n_train,
                "accuracy": pooled["accuracy"],
                "macro_f1": pooled["macro_f1"],
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"  {label:<28s} n_train={n_train:>6,}  "
            f"acc={pooled['accuracy']:.4f}  f1={pooled['macro_f1']:.4f}  "
            f"({rows[-1]['seconds']}s)",
            flush=True,
        )

    table = pd.DataFrame(rows)
    baseline = float(table.loc[table["added"] == "(ravdess only)", "macro_f1"].iloc[0])
    table["vs_control_pp"] = ((table["macro_f1"] - baseline) * 100).round(2)
    return table.sort_values("macro_f1", ascending=False)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase12_corpus_sweep.parquet", index=False)
    print("\n=== corpus-selection sweep (frozen RAVDESS test folds) ===")
    print(table.to_string(index=False))
