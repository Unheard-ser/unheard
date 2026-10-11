"""Leave-one-corpus-out — the only deployment-realistic test in the project.

Every other protocol holds out *speakers* recorded in the same studio, on the
same equipment, reading the same prompts. This one holds out a whole corpus:
train on three, test on a fourth the model has never heard — different people,
different microphones, different room, different sentences. That is what a
deployed model actually meets.

Expect a large drop. The drop is the result.

Metric note that matters here
-----------------------------
CREMA-D contains no ``calm`` and no ``surprised``; TESS and SAVEE contain no
``calm``. Macro-F1 over the project's fixed 8 labels would score those absent
classes 0 and divide by 8, which understates a fold for a reason that has
nothing to do with the model. Both are reported: ``macro_f1_all8`` for
continuity with the rest of the project, and ``macro_f1_present`` over the
classes the test fold actually contains, which is the honest one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ser import set_seeds
from ser import corpora
from ser.embeddings import EmbeddingConfig
from ser.evaluate import evaluate_pooled
from ser.features import CACHE_DIR
from ser.run_early_read import prepare_merged
from ser.run_wav2vec2 import build_matrix, load_cached
from ser.run_wav2vec2_transfer import WINNER as W2V_WINNER
from ser.run_wav2vec2_transfer import linear_head
from ser.splits_merged import get_merged_split, load_merged_folds
from ser.stats import cluster_bootstrap_ci


def score_holdout(
    features: pd.DataFrame,
    labels: pd.Series,
    indexed: pd.DataFrame,
    tag: str,
    config_name: str,
    results_path: Path | None = None,
) -> pd.DataFrame:
    """Train on three corpora, test on the fourth, for each corpus in turn."""
    folds = load_merged_folds()
    protocol = "corpus_holdout"
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == protocol, "fold"].unique()
    )

    rows, per_fold = [], []
    for fold in fold_numbers:
        train_files, test_files = get_merged_split(protocol, fold, folds=folds)
        held_out = sorted(set(indexed.loc[test_files, "source"].astype(str)))
        started = time.time()

        model = linear_head()
        model.fit(
            features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy()
        )
        predicted = model.predict(features.loc[test_files].to_numpy())
        truth = labels.loc[test_files].to_numpy()

        present = sorted(set(truth))
        meta = indexed.loc[test_files].reset_index()
        per_fold.append((truth, predicted, meta))

        interval = cluster_bootstrap_ci(
            truth, predicted, meta["speaker_id"].to_numpy(), "macro_f1"
        )
        rows.append({
            "features": tag,
            "held_out": ",".join(held_out),
            "n_train": len(train_files),
            "n_test": len(test_files),
            "classes_in_test": len(present),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1_all8": float(
                f1_score(truth, predicted, average="macro", zero_division=0)
            ),
            "macro_f1_present": float(
                f1_score(truth, predicted, labels=present, average="macro",
                         zero_division=0)
            ),
            "f1_lo": interval["lo"],
            "f1_hi": interval["hi"],
            "seconds": round(time.time() - started, 1),
        })
        print(
            f"  hold out {rows[-1]['held_out']:<9s} "
            f"train={len(train_files):>6,} test={len(test_files):>6,} "
            f"cls={len(present)}  acc={rows[-1]['accuracy']:.4f} "
            f"f1(present)={rows[-1]['macro_f1_present']:.4f}  "
            f"({rows[-1]['seconds']}s)",
            flush=True,
        )

    evaluate_pooled(
        per_fold, protocol,
        model="linear-svc", feature_config=config_name,
        hyperparams={"head": "LinearSVC", "features": tag},
        notes=f"phase-15 leave-one-corpus-out, {tag} (pooled out-of-fold)",
        results_path=results_path,
    )
    return pd.DataFrame(rows)


def run(results_path: Path | None = None) -> pd.DataFrame:
    set_seeds()
    index = corpora.load_index()
    tables = []

    print("handcrafted features")
    indexed, features, labels = prepare_merged(index)
    tables.append(score_holdout(
        features, labels, indexed, "handcrafted",
        "mfcc20-per_speaker", results_path,
    ))

    print("\nwav2vec2 layer 6 embeddings")
    try:
        tensor, emb_index = load_cached(EmbeddingConfig(batch_size=8))
        emb_features, emb_labels = build_matrix(tensor, emb_index, W2V_WINNER)
        tables.append(score_holdout(
            emb_features, emb_labels, emb_index.set_index("filename"),
            "wav2vec2-L6", W2V_WINNER.name, results_path,
        ))
    except Exception as exc:  # noqa: BLE001 - torch may be unavailable
        print(f"  skipped: {exc!r}", flush=True)

    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase15_corpus_holdout.parquet", index=False)
    print("\n=== leave-one-corpus-out ===")
    print(table.to_string(index=False))
