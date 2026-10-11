"""Phase 9.5 — the early read on whether the merged corpus helps.

One model (SVM-RBF, default hyperparameters) on the Phase 4 winning feature
config, run through four cells and one diagnostic. Deliberately small: the
question here is *directional*, and a single estimator at defaults answers it
days before the full grid in Phase 12 can.

The four cells
--------------
``C1 ravdess-only (phase-5 reproduction)``
    The existing pipeline, untouched. Must reproduce 0.6389 / 0.6333. If it
    does not, something in the environment changed and every comparison below
    is meaningless -- so it runs first and is checked.

``C2 ravdess-only (merged feature matrix)``
    The same frozen folds and the same clips, but features extracted and
    per-speaker normalised in the *merged* pipeline. The honest control for
    C3: it differs from C3 only by the extra training clips, so the C2 -> C3
    delta is attributable to the added data and nothing else.

``C3 architecture A (external corpora added to training)``
    Frozen RAVDESS test folds, training side augmented with CREMA-D, TESS and
    SAVEE. **This is the number that answers "does the merge help?"**

``C4 architecture B (merged test folds)``
    5-fold speaker-independent over all 12,162 clips. A different denominator
    entirely -- not comparable to C1-C3, and labelled as such everywhere.

The diagnostic
--------------
``corpus-id probe`` asks whether a classifier can tell which corpus a clip
came from, using the same features the emotion model uses. If it can do so
near-perfectly then the corpora are trivially separable, which is what makes
``calm`` -- a RAVDESS-only class -- reachable by a shortcut in Architecture B.
It is the direct analogue of the Phase 7 actor-identity probe that produced
this project's headline finding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser import corpora
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import (
    CACHE_DIR,
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.models.classical import build, default_hyperparams
from ser.run_phase5 import WINNER, prepare as prepare_ravdess
from ser.splits import get_split, load_folds
from ser.splits_merged import (
    MERGED_FOLDS_PATH,
    augmented_train,
    get_merged_split,
    load_merged_folds,
    write_merged_folds,
)

MODEL = "svm-rbf"

#: The Phase 5 floor these cells are measured against. This is the **untuned**
#: svm-rbf row, because that is what these cells run -- comparing an untuned
#: model against Phase 5's tuned 0.6333 would read a 0.22 pp tuning gain as
#: environment drift and raise a false alarm on every run.
FLOOR_ACC, FLOOR_F1 = 0.638889, 0.631106

#: Phase 5's tuned result, for context in the write-up only. Tuning bought
#: +0.22 pp macro-F1 -- a fifth of the 5.09 pp fold standard deviation.
TUNED_ACC, TUNED_F1 = 0.638889, 0.633295

#: Distinct from `FeatureConfig.cache_path`: the merged corpus shares a config
#: hash with RAVDESS, so without its own file the two caches evict each other.
MERGED_CACHE: Path = CACHE_DIR / f"features_merged_{WINNER.hash}.parquet"

EXTERNAL: tuple[str, ...] = ("CREMA-D", "TESS", "SAVEE")


def prepare_merged(
    index: pd.DataFrame, config: FeatureConfig = WINNER, n_jobs: int = -1
):
    """Extract and normalise features over the merged corpus.

    Per-speaker normalisation uses ``speaker_id`` rather than ``actor``,
    because speakers are now globally unique across four corpora. It is worth
    noting that this is *better* grounded here than on RAVDESS alone: speakers
    nest within corpora, so z-scoring within a speaker also removes that
    corpus's recording gain -- a principled answer to the problem a global
    peak-normalisation toggle was reaching for.

    Args:
        index: Merged index from `ser.corpora.build_index`.
        config: Feature configuration.
        n_jobs: Extraction workers. ``-1`` uses every core.

    Returns:
        ``(indexed, features, labels)`` all keyed by filename.
    """
    frame = extract(config, index, cache_path=MERGED_CACHE, n_jobs=n_jobs)
    columns = feature_columns(frame, meta=index)
    indexed = frame.set_index("filename")

    values = indexed[columns].to_numpy()
    if config.normalisation == "per_speaker":
        values = per_speaker_normalise(values, indexed["speaker_id"].to_numpy())

    features = pd.DataFrame(values, index=indexed.index, columns=columns)
    return indexed, features, indexed["emotion_label"]


def _score(
    splits: list[tuple[list[str], list[str]]],
    features: pd.DataFrame,
    labels: pd.Series,
    indexed: pd.DataFrame,
    protocol: str,
    notes: str,
    results_path: Path | None = None,
) -> dict[str, float]:
    """Fit and score one model across a list of (train, test) filename splits."""
    per_fold, started = [], time.time()

    for fold, (train_files, test_files) in enumerate(splits):
        pipeline = build(MODEL)
        pipeline.fit(
            features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy()
        )
        predicted = pipeline.predict(features.loc[test_files].to_numpy())

        test_meta = indexed.loc[test_files].reset_index()
        evaluate(
            labels.loc[test_files].to_numpy(), predicted, protocol, fold, test_meta,
            model=MODEL, feature_config=WINNER.name,
            hyperparams=default_hyperparams(MODEL),
            notes=notes, results_path=results_path,
        )
        per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))

    pooled = evaluate_pooled(
        per_fold, protocol,
        model=MODEL, feature_config=WINNER.name,
        hyperparams=default_hyperparams(MODEL),
        notes=notes + " (pooled out-of-fold)", results_path=results_path,
    )
    accuracies = [float((t == p).mean()) for t, p, _ in per_fold]
    return {
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "n_test": int(sum(len(t) for t, _, _ in per_fold)),
        "fold_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "seconds": round(time.time() - started, 1),
    }


def corpus_probe(features: pd.DataFrame, indexed: pd.DataFrame) -> dict[str, float]:
    """Can a classifier identify the corpus from the emotion features?

    Speaker-grouped so the probe cannot win by recognising a voice it has
    already heard -- the same discipline the emotion protocols use. A score
    far above the majority-class rate means corpus identity is encoded in the
    features, and therefore available as a shortcut to any class that is
    unevenly distributed across corpora.

    **Read this number with `ser.run_early_diagnostics`, not alone.** It is
    computed on per-speaker-normalised features, and on those it comes out at
    exactly the majority-class rate -- which looks like "the corpora are
    indistinguishable" and is not. Speakers nest inside corpora, so
    per-speaker normalisation removes the corpus signature along with the
    speaker's. On *raw* features the same probe scores 91.8%. The pair is
    interpretable; either number alone is misleading.

    Args:
        features: Merged feature matrix, keyed by filename.
        indexed: Merged index, keyed by filename.

    Returns:
        ``accuracy``, ``majority`` (the trivial baseline) and ``ratio``.
    """
    from sklearn.model_selection import GroupKFold

    target = indexed["source"].astype(str).to_numpy()
    groups = indexed["speaker_id"].to_numpy()
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


def run(results_path: Path | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """Run the four cells and the corpus probe.

    Returns:
        One row per cell, with accuracy, macro-F1 and the delta against the
        Phase 5 floor where that comparison is meaningful.
    """
    set_seeds()
    rows: list[dict[str, object]] = []

    # --- C1: the control that proves nothing drifted ----------------------
    print("C1  ravdess-only, phase-5 reproduction ...", flush=True)
    meta, r_indexed, r_features, r_labels = prepare_ravdess()
    folds = load_folds()
    splits = [
        get_split("speaker_independent", f, folds=folds)
        for f in sorted(
            folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
        )
    ]
    c1 = _score(
        splits, r_features, r_labels, r_indexed, "speaker_independent",
        "phase-9.5 C1 ravdess-only control", results_path,
    )
    rows.append({"cell": "C1 ravdess-only (phase-5 repro)", "comparable": "yes", **c1})
    drift = abs(c1["macro_f1"] - FLOOR_F1)
    print(
        f"    acc={c1['accuracy']:.4f} f1={c1['macro_f1']:.4f} "
        f"(floor {FLOOR_ACC:.4f}/{FLOOR_F1:.4f}, drift {drift:.4f})",
        flush=True,
    )

    # --- merged corpus ----------------------------------------------------
    print("\nbuilding merged index ...", flush=True)
    index = corpora.build_index(validate=False, write=True)
    print(f"    {len(index):,} clips, {index['speaker_id'].nunique()} speakers", flush=True)

    print("extracting merged features (cached; parallel) ...", flush=True)
    started = time.time()
    m_indexed, m_features, m_labels = prepare_merged(index, n_jobs=n_jobs)
    print(
        f"    {m_features.shape[0]:,} x {m_features.shape[1]} "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )

    # --- C2 / C3: architecture A ------------------------------------------
    for cell, sources, label in (
        ("C2", (), "ravdess-only (merged matrix)"),
        ("C3", EXTERNAL, "architecture A (+external training)"),
    ):
        print(f"\n{cell}  {label} ...", flush=True)
        splits = [
            augmented_train(f, sources=sources, index=index)
            for f in sorted(
                folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
            )
        ]
        scored = _score(
            splits, m_features, m_labels, m_indexed, "speaker_independent",
            f"phase-9.5 {cell} {label}", results_path,
        )
        rows.append({"cell": f"{cell} {label}", "comparable": "yes", **scored})
        print(
            f"    acc={scored['accuracy']:.4f} f1={scored['macro_f1']:.4f} "
            f"(n_test={scored['n_test']:,})",
            flush=True,
        )

    # --- C4: architecture B ------------------------------------------------
    print("\nC4  architecture B (merged test folds) ...", flush=True)
    # First write is the freeze. Done explicitly and announced, because after
    # this point the assignments are a contract: every Architecture B result
    # assumes them, and regenerating them silently would invalidate each one.
    if not MERGED_FOLDS_PATH.exists():
        written = write_merged_folds(index=index)
        print(f"    FROZE merged folds -> {written}", flush=True)
    merged_folds = load_merged_folds()
    protocol = "speaker_independent_merged"
    splits = [
        get_merged_split(protocol, f, folds=merged_folds)
        for f in sorted(
            merged_folds.loc[merged_folds["protocol"] == protocol, "fold"].unique()
        )
    ]
    c4 = _score(
        splits, m_features, m_labels, m_indexed, protocol,
        "phase-9.5 C4 architecture B", results_path,
    )
    rows.append({"cell": "C4 architecture B (merged test)", "comparable": "NO", **c4})
    print(
        f"    acc={c4['accuracy']:.4f} f1={c4['macro_f1']:.4f} "
        f"(n_test={c4['n_test']:,}) -- NOT comparable to C1-C3",
        flush=True,
    )

    # --- the diagnostic ----------------------------------------------------
    print("\ncorpus-id probe ...", flush=True)
    probe = corpus_probe(m_features, m_indexed)
    print(
        f"    corpus identified with {probe['accuracy']:.1%} accuracy "
        f"(majority class {probe['majority']:.1%}, {probe['ratio']:.2f}x)",
        flush=True,
    )

    table = pd.DataFrame(rows)
    table["vs_floor_f1"] = np.where(
        table["comparable"] == "yes",
        (table["macro_f1"] - FLOOR_F1).round(4),
        np.nan,
    )
    return table


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    table = run(results_path=args.results, n_jobs=args.n_jobs)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase95_early_read.parquet", index=False)

    print("\n=== early read ===")
    print(table.to_string(index=False))
