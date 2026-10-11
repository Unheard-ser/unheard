"""Fold definitions for the merged corpus — the two architectures.

Deliberately a separate module from `ser.splits`. That module owns
``splits/folds.csv``, the frozen contract behind every result logged so far,
and it never gains a new code path that writes a fold file. Keeping the new
protocols here means no edit in this file can reach that one.

Architecture A — RAVDESS-test
-----------------------------
External corpora are **training data only**. The test side of every fold is
exactly the frozen RAVDESS ``speaker_independent`` test fold, untouched, so:

* the graded problem stays 8-class RAVDESS, and results stay directly
  comparable to the 0.6333 macro-F1 floor and to all 446 logged runs;
* training speakers rise from 19 to 116, which is itself a leakage reducer --
  there is far less opportunity to memorise a voice;
* **the corpus shortcut is neutralised by construction.** ``calm`` is
  RAVDESS-only, so on a merged *test* set a model could reach it by detecting
  the corpus. Here every test clip is RAVDESS, so "is this RAVDESS?" is
  constant across the test set and carries exactly zero information.

No new fold file is needed: `augmented_train` reads the frozen folds and only
varies the training side.

Architecture B — merged-test
----------------------------
New folds over all clips, frozen in ``splits/folds_merged.csv`` under the same
write-once guard `ser.splits.write_folds` uses.

``speaker_independent_merged``
    5-fold ``GroupKFold`` on ``speaker_id`` across all 121 speakers.

``corpus_holdout``
    Leave-one-corpus-out. The honest cross-corpus generalisation number, and
    the only protocol here where recording conditions differ between train and
    test the way they would in deployment.

``random_stratified_merged``
    5-fold ``StratifiedKFold``. The leakage comparison only, never reported
    alone (CLAUDE.md rule 4).

Fold sizes will be uneven and that is not fixable: TESS has 2 speakers for
2,800 clips while CREMA-D has 91 for 7,442, so any speaker-grouped partition
lands lopsided. Prefer pooled out-of-fold scoring, which `ser.evaluate`
already does, and read `summarise_merged_folds` before trusting a per-fold
number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from ser import SEED  # single source; do not redeclare
from ser import corpora
from ser.splits import FOLDS_COLUMNS, N_SPLITS, get_split

#: The merged fold assignments. Frozen on first write, like ``folds.csv``.
MERGED_FOLDS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "splits" / "folds_merged.csv"
)

MERGED_PROTOCOLS: tuple[str, ...] = (
    "speaker_independent_merged",
    "corpus_holdout",
    "random_stratified_merged",
)

_Split = tuple[np.ndarray, np.ndarray]


# ==========================================================================
# Architecture A -- RAVDESS test side, external clips added to training
# ==========================================================================


def augmented_train(
    fold: int,
    sources: tuple[str, ...] | list[str] = (),
    index: pd.DataFrame | None = None,
    protocol: str = "speaker_independent",
) -> tuple[list[str], list[str]]:
    """Frozen RAVDESS fold, with external corpora added to the training side.

    The test side is returned exactly as ``splits/folds.csv`` defines it. The
    training side is that fold's RAVDESS training clips plus every clip from
    the requested external corpora.

    ``sources`` is a parameter because *which* corpora help is an experiment,
    not a setting: CREMA-D contributes no ``calm`` and no ``surprised``, so it
    may shift the priors against exactly the two classes RAVDESS is weakest on.

    Args:
        fold: Fold number, 0-based, in the frozen RAVDESS protocol.
        sources: External corpora to add to training. ``()`` reproduces the
            RAVDESS-only baseline exactly.
        index: Merged index from `ser.corpora.build_index`. Defaults to
            reading the built index from disk.
        protocol: Which frozen RAVDESS protocol supplies the fold.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If ``sources`` names RAVDESS (it is always present), an
            unknown corpus, or if the resulting test side is not RAVDESS-only.
    """
    requested = tuple(sources)
    if "RAVDESS" in requested:
        raise ValueError(
            "RAVDESS is the test corpus and is always in training; "
            "list only external corpora in `sources`"
        )
    unknown = set(requested) - set(corpora.SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)}")

    train, test = get_split(protocol, fold)

    if requested:
        if index is None:
            index = corpora.load_index()
        extra = index.loc[
            index["source"].isin(requested), "filename"
        ].tolist()
        train = list(train) + extra

    # The property this architecture rests on. Assert it rather than trust it:
    # a single external clip on the test side would reintroduce the corpus cue
    # that holding the test set to RAVDESS exists to eliminate.
    if index is not None:
        lookup = index.set_index("filename")["source"]
        leaked = {
            str(lookup[name])
            for name in test
            if name in lookup.index and str(lookup[name]) != "RAVDESS"
        }
        if leaked:
            raise ValueError(
                f"test side contains non-RAVDESS clips from {sorted(leaked)}; "
                f"Architecture A requires a RAVDESS-only test set"
            )

    return train, test


# ==========================================================================
# Architecture B -- merged test folds
# ==========================================================================


def speaker_independent_merged(
    index: pd.DataFrame, n_splits: int = N_SPLITS
) -> list[_Split]:
    """5-fold GroupKFold on ``speaker_id`` across every corpus.

    No speaker appears on both sides of a fold. Note that this does **not**
    hold out a corpus: all four corpora appear in both training and test, so a
    model may still exploit recording conditions. `corpus_holdout` is the
    protocol that removes that.

    Args:
        index: Merged index frame.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return list(
        splitter.split(index, index["emotion_label"], groups=index["speaker_id"])
    )


def corpus_holdout(index: pd.DataFrame) -> list[_Split]:
    """Leave-one-corpus-out — the honest cross-corpus number.

    One fold per corpus present: train on the other three, test on this one.
    Both the speakers and the recording conditions differ between the sides,
    which is the situation a deployed model actually faces.

    Expect a large drop against `speaker_independent_merged`. That drop is the
    result, not a failure: it is the part of the merged-test score that was
    coming from shared recording conditions.

    A corpus missing a label cannot be scored on it -- CREMA-D has no ``calm``
    and no ``surprised`` -- so per-class metrics on those folds are reported
    over the labels actually present.

    Args:
        index: Merged index frame.

    Returns:
        ``(train_idx, test_idx)`` pairs, one per corpus, in `SOURCES` order.
    """
    positions = np.arange(len(index))
    source = index["source"].astype(str).to_numpy()
    out: list[_Split] = []
    for name in corpora.SOURCES:
        mask = source == name
        if not mask.any():
            continue
        out.append((positions[~mask], positions[mask]))
    return out


def random_stratified_merged(
    index: pd.DataFrame, n_splits: int = N_SPLITS
) -> list[_Split]:
    """5-fold StratifiedKFold on emotion — the leakage comparison only.

    The same speaker, and the same corpus, appear on both sides of every fold.
    That is the defect this protocol exists to quantify by comparison against
    `speaker_independent_merged`. Never report this number alone.

    Args:
        index: Merged index frame.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(index, index["emotion_label"]))


_BUILDERS = {
    "speaker_independent_merged": speaker_independent_merged,
    "corpus_holdout": corpus_holdout,
    "random_stratified_merged": random_stratified_merged,
}


def build_merged_folds(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the long-format fold table for every merged protocol.

    Args:
        index: Merged index frame. Defaults to the built index on disk.

    Returns:
        Columns ``protocol, fold, filename, split``, deterministically sorted
        so the CSV is byte-stable across machines.
    """
    if index is None:
        index = corpora.load_index()

    filenames = index["filename"].to_numpy()
    records: list[pd.DataFrame] = []

    for protocol in MERGED_PROTOCOLS:
        for fold, (train_idx, test_idx) in enumerate(_BUILDERS[protocol](index)):
            for side, idx in (("train", train_idx), ("test", test_idx)):
                records.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "filename": filenames[idx],
                            "split": side,
                        }
                    )
                )

    folds = pd.concat(records, ignore_index=True)
    folds["_order"] = folds["protocol"].map(
        {name: i for i, name in enumerate(MERGED_PROTOCOLS)}
    )
    folds = (
        folds.sort_values(["_order", "fold", "split", "filename"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return folds[list(FOLDS_COLUMNS)]


def write_merged_folds(
    path: Path | None = None,
    force: bool = False,
    index: pd.DataFrame | None = None,
) -> Path:
    """Write the merged fold assignments to CSV, once.

    Refuses to overwrite, for the same reason `ser.splits.write_folds` does:
    the moment a result is logged against these assignments, regenerating them
    silently invalidates it.

    Args:
        path: Destination. Defaults to `MERGED_FOLDS_PATH`.
        force: Overwrite an existing file. Do not use.
        index: Merged index to build from. Defaults to the built index on
            disk. Passed explicitly by tests, which must not depend on a
            corpus having been downloaded.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    target = Path(path) if path is not None else MERGED_FOLDS_PATH
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists and merged folds are frozen. Every "
            f"Architecture B result assumes these exact assignments."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    build_merged_folds(index).to_csv(target, index=False, lineterminator="\n")
    return target


def load_merged_folds(path: Path | None = None) -> pd.DataFrame:
    """Read the frozen merged fold assignments.

    Args:
        path: Source. Defaults to `MERGED_FOLDS_PATH`.

    Returns:
        Columns ``protocol, fold, filename, split``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(path) if path is not None else MERGED_FOLDS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"No merged fold assignments at {source}. Generate them once with "
            f"ser.splits_merged.write_merged_folds(), then commit the file."
        )
    return pd.read_csv(source, dtype={"protocol": str, "filename": str, "split": str})


def get_merged_split(
    protocol: str, fold: int, folds: pd.DataFrame | None = None
) -> tuple[list[str], list[str]]:
    """Return the train and test filenames for one merged (protocol, fold).

    Args:
        protocol: One of `MERGED_PROTOCOLS`.
        fold: Fold number, 0-based.
        folds: Fold table. Defaults to reading the frozen CSV.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If the protocol is unknown or the fold does not exist.
    """
    if protocol not in MERGED_PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {MERGED_PROTOCOLS}"
        )
    table = load_merged_folds() if folds is None else folds
    subset = table[(table["protocol"] == protocol) & (table["fold"] == fold)]
    if subset.empty:
        available = sorted(table.loc[table["protocol"] == protocol, "fold"].unique())
        raise ValueError(
            f"No fold {fold} for protocol {protocol!r}; available: {available}"
        )
    train = subset.loc[subset["split"] == "train", "filename"].tolist()
    test = subset.loc[subset["split"] == "test", "filename"].tolist()
    return train, test


def summarise_merged_folds(
    folds: pd.DataFrame | None = None, index: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-fold composition of every merged protocol's test side.

    The table to read before trusting any per-fold number, because merged
    folds are necessarily uneven.

    Args:
        folds: Fold table. Defaults to the frozen CSV.
        index: Merged index frame. Defaults to the built index.

    Returns:
        One row per (protocol, fold) with sizes, speaker and corpus counts,
        overlap checks, gender split and the two sparse classes' counts.
    """
    table = load_merged_folds() if folds is None else folds
    frame = corpora.load_index() if index is None else index
    lookup = frame.set_index("filename")

    rows = []
    for (protocol, fold), group in table.groupby(["protocol", "fold"], sort=False):
        train = lookup.loc[group.loc[group["split"] == "train", "filename"]]
        test = lookup.loc[group.loc[group["split"] == "test", "filename"]]
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_speakers": train["speaker_id"].nunique(),
                "test_speakers": test["speaker_id"].nunique(),
                "speaker_overlap": len(
                    set(train["speaker_id"]) & set(test["speaker_id"])
                ),
                "corpus_overlap": len(
                    set(train["source"].astype(str)) & set(test["source"].astype(str))
                ),
                "test_corpora": ",".join(sorted(set(test["source"].astype(str)))),
                "test_pct_male": round(100 * (test["gender"] == "male").mean(), 1),
                "test_emotions": test["emotion_label"].nunique(),
                "test_calm": int((test["emotion_label"] == "calm").sum()),
                "test_neutral": int((test["emotion_label"] == "neutral").sum()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    if MERGED_FOLDS_PATH.exists():
        print(f"{MERGED_FOLDS_PATH} exists (frozen). Composition:")
    else:
        print(f"Writing {write_merged_folds()}")
    print(summarise_merged_folds().to_string(index=False))
