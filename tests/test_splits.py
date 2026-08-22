"""Invariants for the frozen fold assignments.

The no-leakage assertions here are the ones that matter most in the whole
project: if an actor appears on both sides of a speaker-independent fold,
every number downstream is inflated and the report's central claim is false.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ser.metadata import load
from ser.splits import (
    FOLDS_COLUMNS,
    FOLDS_PATH,
    PROTOCOLS,
    build_folds,
    get_split,
    load_folds,
    random_stratified,
    speaker_independent,
    statement_holdout,
    statement_holdout_si,
    summarise_folds,
    write_folds,
)


@pytest.fixture(scope="module")
def meta():
    return load()


@pytest.fixture(scope="module")
def folds(meta):
    return build_folds(meta)


def _sides(folds: pd.DataFrame, protocol: str, fold: int) -> tuple[set[str], set[str]]:
    sub = folds[(folds["protocol"] == protocol) & (folds["fold"] == fold)]
    return (
        set(sub.loc[sub["split"] == "train", "filename"]),
        set(sub.loc[sub["split"] == "test", "filename"]),
    )


def _fold_numbers(folds: pd.DataFrame, protocol: str) -> list[int]:
    return sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())


# --- 1. the core no-leakage assertion --------------------------------------


def test_no_actor_on_both_sides_of_a_speaker_independent_fold(meta):
    """The single most important invariant in the project."""
    actor = meta.set_index("filename")["actor"]
    for fold, (train_idx, test_idx) in enumerate(speaker_independent(meta)):
        train_actors = set(meta.iloc[train_idx]["actor"])
        test_actors = set(meta.iloc[test_idx]["actor"])
        assert train_actors & test_actors == set(), f"fold {fold} leaks actors"
    # And again through the frozen table, not just the generator.
    table = build_folds(meta)
    for fold in _fold_numbers(table, "speaker_independent"):
        train, test = _sides(table, "speaker_independent", fold)
        assert set(actor[list(train)]) & set(actor[list(test)]) == set()


# --- 2. the speaker-disjoint statement holdout -----------------------------


def test_statement_holdout_si_is_disjoint_on_actor_and_statement(meta, folds):
    lookup = meta.set_index("filename")
    for fold in _fold_numbers(folds, "statement_holdout_si"):
        train, test = _sides(folds, "statement_holdout_si", fold)
        tr, te = lookup.loc[list(train)], lookup.loc[list(test)]
        assert set(tr["actor"]) & set(te["actor"]) == set(), f"fold {fold}: actor leak"
        assert set(tr["statement"]) & set(te["statement"]) == set()
        assert set(tr["statement"]) == {"01"}
        assert set(te["statement"]) == {"02"}


# --- 3. every clip held out exactly once -----------------------------------


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        ("speaker_independent", 1440),
        ("random_stratified", 1440),
        ("statement_holdout", 720),
        ("statement_holdout_si", 720),
    ],
)
def test_each_clip_appears_in_exactly_one_test_fold(folds, protocol, expected):
    test_rows = folds[(folds["protocol"] == protocol) & (folds["split"] == "test")]
    assert len(test_rows) == expected
    assert not test_rows["filename"].duplicated().any()


# --- 4 & 5. set algebra per fold -------------------------------------------


def test_train_and_test_never_overlap(folds):
    for protocol in PROTOCOLS:
        for fold in _fold_numbers(folds, protocol):
            train, test = _sides(folds, protocol, fold)
            assert train & test == set(), f"{protocol} fold {fold} overlaps"


@pytest.mark.parametrize(
    "protocol", ["speaker_independent", "random_stratified", "statement_holdout"]
)
def test_train_and_test_cover_every_clip(folds, protocol):
    for fold in _fold_numbers(folds, protocol):
        train, test = _sides(folds, protocol, fold)
        assert len(train | test) == 1440


def test_statement_holdout_si_deliberately_does_not_cover_everything(folds, meta):
    """Its train set is a subset, not the complement -- hence long format."""
    train, test = _sides(folds, "statement_holdout_si", 0)
    assert len(train | test) < 1440


# --- 6. the leakage protocol must actually leak ----------------------------


def test_random_stratified_has_all_actors_on_both_sides(meta):
    """Asserting the flaw exists: this is what the comparison measures."""
    for fold, (train_idx, test_idx) in enumerate(random_stratified(meta)):
        assert meta.iloc[train_idx]["actor"].nunique() == 24
        assert meta.iloc[test_idx]["actor"].nunique() == 24, f"fold {fold}"


# --- 7. plain statement holdout --------------------------------------------


def test_statement_holdout_splits_on_statement(meta):
    splits = statement_holdout(meta)
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert set(meta.iloc[train_idx]["statement"]) == {"01"}
    assert set(meta.iloc[test_idx]["statement"]) == {"02"}
    assert len(train_idx) == len(test_idx) == 720


# --- 8. class distribution per fold ----------------------------------------


def test_speaker_independent_class_distribution_is_proportional(meta, capsys):
    """Every actor has identical composition, so folds are proportional.

    Prints the full composition table -- the plan requires it to be reported.
    """
    rows = []
    for fold, (_, test_idx) in enumerate(speaker_independent(meta)):
        test = meta.iloc[test_idx]
        n_actors = test["actor"].nunique()
        counts = test["emotion_label"].value_counts()
        assert counts["neutral"] == 4 * n_actors
        for label in counts.index:
            if label != "neutral":
                assert counts[label] == 8 * n_actors, (fold, label)
        rows.append({"fold": fold, "n_actors": n_actors, **counts.to_dict()})

    with capsys.disabled():
        print("\nspeaker_independent test-fold composition:")
        print(pd.DataFrame(rows).to_string(index=False))


# --- 9. the documented structure -------------------------------------------


def test_fold_sizes_match_the_documented_structure(meta):
    sizes, actor_counts = [], []
    for _, test_idx in speaker_independent(meta):
        sizes.append(len(test_idx))
        actor_counts.append(meta.iloc[test_idx]["actor"].nunique())
    assert sorted(sizes, reverse=True) == [300, 300, 300, 300, 240]
    assert sorted(actor_counts, reverse=True) == [5, 5, 5, 5, 4]


def test_statement_holdout_si_sizes(meta):
    sizes = [(len(tr), len(te)) for tr, te in statement_holdout_si(meta)]
    assert sorted(sizes, reverse=True) == [
        (600, 120), (570, 150), (570, 150), (570, 150), (570, 150),
    ]


# --- 10. determinism --------------------------------------------------------


def test_build_folds_is_deterministic(meta):
    from pandas.testing import assert_frame_equal

    assert_frame_equal(build_folds(meta), build_folds(meta))


def test_folds_columns_are_exactly_the_schema(folds):
    assert tuple(folds.columns) == FOLDS_COLUMNS
    assert set(folds["split"].unique()) == {"train", "test"}
    assert set(folds["protocol"].unique()) == set(PROTOCOLS)


# --- 11. the committed file matches what the code produces -----------------


@pytest.mark.skipif(
    not FOLDS_PATH.exists(), reason="folds.csv not generated yet (first run)"
)
def test_committed_folds_match_a_fresh_build(meta):
    """Catches accidental regeneration or hand-editing of the frozen file."""
    from pandas.testing import assert_frame_equal

    committed = load_folds()
    fresh = build_folds(meta)
    assert_frame_equal(
        committed.reset_index(drop=True),
        fresh.reset_index(drop=True),
        check_dtype=False,
    )


# --- 12. freezing is enforced ----------------------------------------------


def test_write_folds_refuses_to_overwrite(tmp_path):
    target = tmp_path / "folds.csv"
    write_folds(target)
    assert target.is_file()
    with pytest.raises(FileExistsError, match="frozen"):
        write_folds(target)


def test_write_folds_force_overwrites(tmp_path):
    target = tmp_path / "folds.csv"
    write_folds(target)
    write_folds(target, force=True)  # escape hatch exists but is never used
    assert target.is_file()


# --- 13. get_split ----------------------------------------------------------


def test_get_split_returns_valid_filenames(meta, folds):
    known = set(meta["filename"])
    train, test = get_split("speaker_independent", 0, folds=folds)
    assert set(train) <= known and set(test) <= known
    assert len(train) + len(test) == 1440
    assert set(train) & set(test) == set()


def test_get_split_rejects_unknown_protocol(folds):
    with pytest.raises(ValueError, match="Unknown protocol"):
        get_split("nonexistent", 0, folds=folds)


def test_get_split_rejects_unknown_fold(folds):
    with pytest.raises(ValueError, match="No fold 99"):
        get_split("speaker_independent", 99, folds=folds)


# --- the summary table ------------------------------------------------------


def test_summarise_folds_reports_zero_overlap_where_expected(meta, folds, capsys):
    summary = summarise_folds(folds, meta)
    si = summary[summary["protocol"].isin(
        ["speaker_independent", "statement_holdout_si"]
    )]
    assert (si["actor_overlap"] == 0).all()

    rs = summary[summary["protocol"] == "random_stratified"]
    assert (rs["actor_overlap"] == 24).all(), "the leakage protocol must overlap"

    with capsys.disabled():
        print("\nfold composition, all protocols:")
        print(summary.to_string(index=False))
