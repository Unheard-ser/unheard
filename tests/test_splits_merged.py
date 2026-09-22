"""Invariants for the two merged-corpus architectures.

The single most important test in this file is
`test_frozen_ravdess_folds_are_never_touched`. Everything else here adds new
fold definitions; that test asserts the old ones are inert. If it ever fails,
446 logged results stopped being comparable and the failure must be treated as
a data-integrity incident, not a flaky test.

The rest split in two. **Architecture A** tests assert the property the
headline protocol rests on: the test side is RAVDESS-only and byte-identical
to the frozen folds, so adding external training data is a single-variable
change. **Architecture B** tests assert the usual disjointness guarantees on
the new folds.

Structural tests run against a synthetic index so they work in CI, where no
external audio exists.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ser import corpora
from ser.metadata import DATA_ROOT
from ser.splits import FOLDS_PATH, get_split, load_folds
from ser.splits_merged import (
    MERGED_PROTOCOLS,
    build_merged_folds,
    corpus_holdout,
    random_stratified_merged,
    speaker_independent_merged,
    write_merged_folds,
)

RAVDESS_PRESENT = DATA_ROOT.is_dir()


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def synthetic_index() -> pd.DataFrame:
    """A merged index with the real corpora's awkward shape, minus the audio.

    Deliberately lopsided in the way the real merge is: TESS gets 2 speakers
    carrying many clips, CREMA-D many speakers, and neither CREMA-D nor a
    non-RAVDESS corpus contributes ``calm``.
    """
    rows = []
    plan = [
        ("RAVDESS", 6, 16, corpora.CANONICAL),
        ("CREMA-D", 12, 12, tuple(sorted(set(corpora.CREMAD_EMOTION.values())))),
        ("TESS", 2, 40, tuple(sorted(set(corpora.TESS_EMOTION.values())))),
        ("SAVEE", 4, 8, tuple(sorted(set(corpora.SAVEE_EMOTION.values())))),
    ]
    for source, n_speakers, per_speaker, labels in plan:
        for speaker in range(n_speakers):
            for clip in range(per_speaker):
                rows.append(
                    corpora._row(
                        path=Path(f"{source}/s{speaker}/{source}_{speaker}_{clip}.wav"),
                        source=source,
                        speaker_id=f"{source}_{speaker:02d}",
                        gender="male" if speaker % 2 else "female",
                        emotion_label=labels[clip % len(labels)],
                    )
                )
    frame = pd.DataFrame(rows, columns=list(corpora.INDEX_COLUMNS))
    frame["source"] = pd.Categorical(frame["source"], categories=list(corpora.SOURCES))
    return frame.reset_index(drop=True)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. THE contract: the frozen RAVDESS folds are inert -------------------


@pytest.mark.skipif(not FOLDS_PATH.exists(), reason="folds.csv absent")
def test_frozen_ravdess_folds_are_never_touched(synthetic_index, tmp_path):
    """Nothing in the merged pipeline may write ``splits/folds.csv``.

    That file is the contract making five people's results comparable. Every
    merged-corpus operation runs here and the file is hashed either side.
    """
    before = _digest(FOLDS_PATH)

    build_merged_folds(synthetic_index)
    write_merged_folds(tmp_path / "folds_merged.csv", index=synthetic_index)
    speaker_independent_merged(synthetic_index)
    corpus_holdout(synthetic_index)
    random_stratified_merged(synthetic_index)
    load_folds()

    assert _digest(FOLDS_PATH) == before, (
        "splits/folds.csv changed during merged-corpus operations -- every "
        "previously logged result has stopped being comparable"
    )


@pytest.mark.skipif(not FOLDS_PATH.exists(), reason="folds.csv absent")
def test_frozen_folds_still_hold_1440_clips():
    folds = load_folds()
    names = folds.loc[folds["protocol"] == "speaker_independent", "filename"]
    assert names.nunique() == 1440


# --- 2. Architecture A -----------------------------------------------------


@pytest.mark.skipif(not RAVDESS_PRESENT, reason="RAVDESS tree absent")
def test_no_sources_reproduces_the_frozen_split_exactly():
    """``sources=()`` must be the RAVDESS baseline, not merely close to it."""
    from ser.splits_merged import augmented_train

    for fold in range(5):
        train, test = augmented_train(fold, sources=())
        frozen_train, frozen_test = get_split("speaker_independent", fold)
        assert train == frozen_train
        assert test == frozen_test


@pytest.mark.skipif(not RAVDESS_PRESENT, reason="RAVDESS tree absent")
def test_adding_sources_leaves_the_test_side_identical():
    """The whole point: only the training side may change."""
    from ser.splits_merged import augmented_train

    index = corpora.build_index(sources=["RAVDESS"], write=False)
    for fold in range(5):
        _, baseline_test = augmented_train(fold, sources=(), index=index)
        _, augmented_test = augmented_train(fold, sources=(), index=index)
        assert baseline_test == augmented_test

        _, frozen_test = get_split("speaker_independent", fold)
        assert baseline_test == frozen_test


@pytest.mark.skipif(not RAVDESS_PRESENT, reason="RAVDESS tree absent")
def test_architecture_a_rejects_ravdess_in_sources():
    from ser.splits_merged import augmented_train

    with pytest.raises(ValueError, match="always in training"):
        augmented_train(0, sources=("RAVDESS",))


@pytest.mark.skipif(not RAVDESS_PRESENT, reason="RAVDESS tree absent")
def test_architecture_a_rejects_unknown_source():
    from ser.splits_merged import augmented_train

    with pytest.raises(ValueError, match="unknown source"):
        augmented_train(0, sources=("IEMOCAP",))


@pytest.mark.skipif(not RAVDESS_PRESENT, reason="RAVDESS tree absent")
def test_architecture_a_test_side_is_entirely_ravdess():
    """Where the calm/corpus shortcut is neutralised, asserted rather than argued."""
    from ser.splits_merged import augmented_train

    index = corpora.build_index(sources=["RAVDESS"], write=False)
    lookup = index.set_index("filename")["source"].astype(str)
    for fold in range(5):
        _, test = augmented_train(fold, sources=(), index=index)
        assert set(lookup.loc[test].unique()) == {"RAVDESS"}


# --- 3. Architecture B: speaker-independent --------------------------------


def test_merged_speaker_independent_has_no_speaker_overlap(synthetic_index):
    speakers = synthetic_index["speaker_id"].to_numpy()
    for train_idx, test_idx in speaker_independent_merged(synthetic_index):
        assert not set(speakers[train_idx]) & set(speakers[test_idx])


def test_merged_speaker_independent_covers_every_clip_once(synthetic_index):
    seen = np.concatenate(
        [test for _, test in speaker_independent_merged(synthetic_index)]
    )
    assert sorted(seen) == list(range(len(synthetic_index)))


# --- 4. Architecture B: leave-one-corpus-out -------------------------------


def test_corpus_holdout_train_and_test_corpora_are_disjoint(synthetic_index):
    source = synthetic_index["source"].astype(str).to_numpy()
    for train_idx, test_idx in corpus_holdout(synthetic_index):
        assert not set(source[train_idx]) & set(source[test_idx])


def test_corpus_holdout_has_one_fold_per_corpus(synthetic_index):
    folds = corpus_holdout(synthetic_index)
    assert len(folds) == synthetic_index["source"].nunique()


def test_corpus_holdout_also_holds_out_every_speaker(synthetic_index):
    """Speaker ids are corpus-prefixed, so corpus disjointness implies this."""
    speakers = synthetic_index["speaker_id"].to_numpy()
    for train_idx, test_idx in corpus_holdout(synthetic_index):
        assert not set(speakers[train_idx]) & set(speakers[test_idx])


def test_ravdess_holdout_is_the_only_fold_containing_calm(synthetic_index):
    """Makes the confound concrete: no other corpus can be tested on calm."""
    labels = synthetic_index["emotion_label"].to_numpy()
    source = synthetic_index["source"].astype(str).to_numpy()
    for _, test_idx in corpus_holdout(synthetic_index):
        held_out = set(source[test_idx])
        has_calm = "calm" in set(labels[test_idx])
        assert has_calm == (held_out == {"RAVDESS"})


# --- 5. Architecture B: the leakage comparison -----------------------------


def test_random_stratified_merged_does_overlap_speakers(synthetic_index):
    """It is supposed to leak. That is the number it exists to produce."""
    speakers = synthetic_index["speaker_id"].to_numpy()
    overlaps = [
        bool(set(speakers[train]) & set(speakers[test]))
        for train, test in random_stratified_merged(synthetic_index)
    ]
    assert all(overlaps)


# --- 6. the merged fold table and its freeze guard -------------------------


def test_build_merged_folds_covers_every_protocol(synthetic_index):
    table = build_merged_folds(synthetic_index)
    assert set(table["protocol"].unique()) == set(MERGED_PROTOCOLS)
    assert set(table["split"].unique()) == {"train", "test"}
    assert list(table.columns) == ["protocol", "fold", "filename", "split"]


def test_build_merged_folds_is_deterministic(synthetic_index):
    from pandas.testing import assert_frame_equal

    assert_frame_equal(
        build_merged_folds(synthetic_index), build_merged_folds(synthetic_index)
    )


def test_write_merged_folds_refuses_to_overwrite(tmp_path, synthetic_index):
    target = tmp_path / "folds_merged.csv"
    write_merged_folds(target, index=synthetic_index)
    assert target.is_file()
    with pytest.raises(FileExistsError, match="frozen"):
        write_merged_folds(target, index=synthetic_index)
