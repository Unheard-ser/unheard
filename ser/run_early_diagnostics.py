"""Phase 9.5 follow-up: *why* did adding external corpora cost 11 points?

The headline early read says Architecture A loses 11.33 pp macro-F1 against the
RAVDESS-only control on the identical test set. That is a result, not an
explanation. Two questions decide what to do about it:

**Is corpus identity in the features at all?** The first probe ran on
per-speaker-normalised features and scored exactly the majority-class rate --
which is not "corpora are indistinguishable", it is almost certainly
per-speaker z-scoring having already removed the corpus signature, since
speakers nest inside corpora. Phase 7 saw the same shape: 92.7% actor identity
from raw features, 21.2% after per-speaker normalisation. So the probe is run
**both ways** here. Only the raw number answers the question.

**Where did the 11 points go?** If the loss is concentrated in `calm` and
`surprised` -- the two classes CREMA-D does not contain -- then the mechanism
is prior shift from 10,722 external clips that never exhibit them, and the
remedy is class weighting or corpus selection. If the loss is spread evenly,
the mechanism is domain shift and the remedy is different.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ser import set_seeds
from ser import corpora
from ser.features import (
    CACHE_DIR,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.models.classical import build
from ser.run_early_read import EXTERNAL, MERGED_CACHE
from ser.run_phase5 import WINNER
from ser.splits import load_folds
from ser.splits_merged import augmented_train


def load_matrices(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the merged index plus raw and per-speaker-normalised features."""
    frame = extract(WINNER, index, cache_path=MERGED_CACHE, n_jobs=-1)
    columns = feature_columns(frame, meta=index)
    indexed = frame.set_index("filename")

    raw = pd.DataFrame(
        indexed[columns].to_numpy(), index=indexed.index, columns=columns
    )
    normalised = pd.DataFrame(
        per_speaker_normalise(raw.to_numpy(), indexed["speaker_id"].to_numpy()),
        index=indexed.index,
        columns=columns,
    )
    return indexed, raw, normalised


def probe(features: pd.DataFrame, target: np.ndarray, groups: np.ndarray) -> dict:
    """Speaker-grouped probe: how well does ``target`` fall out of the features?"""
    matrix = features.to_numpy()
    predicted = np.empty_like(target)
    for train_idx, test_idx in GroupKFold(n_splits=5).split(matrix, target, groups):
        pipeline = build("logistic-regression")
        pipeline.fit(matrix[train_idx], target[train_idx])
        predicted[test_idx] = pipeline.predict(matrix[test_idx])

    accuracy = float((predicted == target).mean())
    majority = float(pd.Series(target).value_counts(normalize=True).max())
    return {
        "accuracy": accuracy,
        "majority": majority,
        "ratio": accuracy / majority if majority else float("nan"),
    }


def per_class_delta(
    indexed: pd.DataFrame, features: pd.DataFrame, index: pd.DataFrame
) -> pd.DataFrame:
    """Per-class recall for the RAVDESS-only control vs Architecture A.

    Same test clips in both, so every difference is attributable to the extra
    training data and nothing else.
    """
    from sklearn.metrics import classification_report

    labels = indexed["emotion_label"]
    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )

    out = {}
    for tag, sources in (("control", ()), ("arch_a", EXTERNAL)):
        trues, preds = [], []
        for fold in fold_numbers:
            train_files, test_files = augmented_train(fold, sources=sources, index=index)
            pipeline = build("svm-rbf")
            pipeline.fit(
                features.loc[train_files].to_numpy(),
                labels.loc[train_files].to_numpy(),
            )
            preds.extend(pipeline.predict(features.loc[test_files].to_numpy()))
            trues.extend(labels.loc[test_files].to_numpy())
        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        out[tag] = {k: v["recall"] for k, v in report.items() if k in set(corpora.CANONICAL)}

    table = pd.DataFrame(out)
    table["delta_pp"] = ((table["arch_a"] - table["control"]) * 100).round(2)
    table["in_cremad"] = [
        label in set(corpora.CREMAD_EMOTION.values()) for label in table.index
    ]
    table["merged_share"] = [
        round(100 * float((indexed["emotion_label"] == label).mean()), 2)
        for label in table.index
    ]
    return table.sort_values("delta_pp")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    index = corpora.load_index()
    indexed, raw, normalised = load_matrices(index)

    source = indexed["source"].astype(str).to_numpy()
    speakers = indexed["speaker_id"].to_numpy()

    print("corpus-identity probe, speaker-grouped")
    rows = []
    for tag, matrix in (("raw", raw), ("per_speaker", normalised)):
        result = probe(matrix, source, speakers)
        rows.append({"features": tag, **result})
        print(
            f"  {tag:<12s} {result['accuracy']:.1%}  "
            f"(majority {result['majority']:.1%}, {result['ratio']:.2f}x)"
        )
    probes = pd.DataFrame(rows)

    print("\nper-class recall, control vs architecture A (identical test clips)")
    deltas = per_class_delta(indexed, normalised, index)
    print(deltas.to_string())

    return probes, deltas


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    probes, deltas = run()
    CACHE_DIR.mkdir(exist_ok=True)
    probes.to_parquet(CACHE_DIR / "phase95_probes.parquet", index=False)
    deltas.to_parquet(CACHE_DIR / "phase95_class_deltas.parquet")
