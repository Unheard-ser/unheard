"""Invariants for the clip inventory.

These encode the RAVDESS facts the rest of the project assumes. If one of
these fails, no downstream result is trustworthy.
"""

from __future__ import annotations

import pytest

from ser import metadata
from ser.metadata import (
    CLIPS_PER_ACTOR,
    EMOTIONS,
    EXPECTED_ACTORS,
    EXPECTED_ROWS,
    NEUTRAL_CLIPS,
    OTHER_EMOTION_CLIPS,
    load,
    parse_filename,
)


@pytest.fixture(scope="module")
def df():
    """The full inventory, parsed once for the module."""
    return load()


# --- 1. size ---------------------------------------------------------------


def test_row_count(df):
    assert len(df) == EXPECTED_ROWS


# --- 2. class balance ------------------------------------------------------


def test_emotion_counts(df):
    counts = df["emotion_label"].value_counts()
    assert counts["neutral"] == NEUTRAL_CLIPS
    for label in EMOTIONS.values():
        if label == "neutral":
            continue
        assert counts[label] == OTHER_EMOTION_CLIPS, label
    assert set(counts.index) == set(EMOTIONS.values())


# --- 3. actors -------------------------------------------------------------


def test_actor_identity_and_counts(df):
    assert sorted(df["actor"].unique()) == list(range(1, EXPECTED_ACTORS + 1))
    assert (df["actor"].value_counts() == CLIPS_PER_ACTOR).all()


# --- 4. gender -------------------------------------------------------------


def test_gender_split_and_parity(df):
    assert df.groupby("gender")["actor"].nunique().to_dict() == {
        "female": 12,
        "male": 12,
    }
    expected = df["actor"].map(lambda a: "male" if a % 2 else "female")
    assert (df["gender"] == expected).all()


# --- 5 & 6. the neutral asymmetry -----------------------------------------


def test_no_strong_intensity_neutral(df):
    """Neutral has half the clips because strong neutral was never recorded."""
    assert ((df["emotion"] == "01") & (df["intensity"] == "02")).sum() == 0
    assert ((df["emotion"] == "01") & (df["intensity"] == "01")).sum() == NEUTRAL_CLIPS


def test_other_emotions_split_evenly_by_intensity(df):
    non_neutral = df[df["emotion"] != "01"]
    per = non_neutral.groupby(["emotion_label", "intensity_label"]).size()
    assert (per == OTHER_EMOTION_CLIPS // 2).all()
    assert len(per) == (len(EMOTIONS) - 1) * 2


# --- 7. statement and repetition ------------------------------------------


def test_statement_and_repetition_balance(df):
    assert df["statement_label"].value_counts().to_dict() == {"kids": 720, "dogs": 720}
    assert df["repetition_label"].value_counts().to_dict() == {"1st": 720, "2nd": 720}


# --- 8. actor field agrees with the directory ------------------------------


def test_actor_field_matches_person_directory(df):
    from_field = df["filename"].str.split("-").str[4].str.replace(".wav", "", regex=False)
    from_dir = df["filepath"].str.extract(r"Person(\d{2})")[0]
    assert (from_field == from_dir).all()
    assert (from_field.astype(int) == df["actor"]).all()


# --- 9. files exist, names unique -----------------------------------------


def test_every_file_exists_and_is_unique(df):
    from pathlib import Path

    assert not df["filename"].duplicated().any()
    assert not df["filepath"].duplicated().any()
    missing = [p for p in df["filepath"] if not Path(p).is_file()]
    assert missing == []


# --- 10. dtypes and nulls --------------------------------------------------


def test_no_nulls_and_actor_is_integer(df):
    assert not df.isna().any().any()
    assert df["actor"].dtype.kind == "i"
    for col in ("emotion", "intensity", "statement", "repetition"):
        assert df[col].map(lambda v: isinstance(v, str) and len(v) == 2).all(), col


def test_constant_ravdess_fields_present(df):
    """modality/vocal_channel are re-added so the frame matches canonical RAVDESS."""
    assert (df["modality"] == "03").all()
    assert (df["vocal_channel"] == "01").all()


# --- 11. the parser is strict ---------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "03-01-06-01-02-01-12.wav",  # canonical seven-field RAVDESS
        "01-01-01-01.wav",  # four fields
        "01-01-01-01-01-01.wav",  # six fields
    ],
)
def test_wrong_field_count_raises(name):
    with pytest.raises(ValueError, match="expected 5 hyphen-separated fields"):
        parse_filename(name)


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("09-01-01-01-01.wav", "unknown emotion code"),
        ("01-03-01-01-01.wav", "unknown intensity code"),
        ("01-01-03-01-01.wav", "unknown statement code"),
        ("01-01-01-03-01.wav", "unknown repetition code"),
        ("01-01-01-01-25.wav", "outside 1..24"),
    ],
)
def test_unknown_codes_raise(name, match):
    with pytest.raises(ValueError, match=match):
        parse_filename(name)


def test_actor_field_disagreeing_with_directory_raises():
    with pytest.raises(ValueError, match="disagrees with directory"):
        parse_filename("data/data/Person01/01-01-01-01-07.wav")


def test_parse_filename_happy_path():
    got = parse_filename("data/data/Person12/06-02-01-02-12.wav")
    assert got["emotion_label"] == "fearful"
    assert got["intensity_label"] == "strong"
    assert got["statement_label"] == "kids"
    assert got["repetition_label"] == "2nd"
    assert got["actor"] == 12
    assert got["gender"] == "female"


# --- 12. determinism -------------------------------------------------------


def test_load_is_deterministic():
    """Row order must not depend on filesystem glob order."""
    from pandas.testing import assert_frame_equal

    assert_frame_equal(load(), load())


def test_sorted_by_actor_then_emotion(df):
    keys = ["actor", "emotion", "intensity", "statement", "repetition"]
    assert df[keys].equals(df[keys].sort_values(keys, kind="mergesort"))


# --- validation rejects a broken frame ------------------------------------


def test_validate_frame_rejects_a_dropped_row(df):
    with pytest.raises(ValueError, match="expected 1440 clips, found 1439"):
        metadata.validate_frame(df.iloc[1:].copy())


def test_validate_frame_rejects_bad_gender(df):
    broken = df.copy()
    broken.loc[broken.index[0], "gender"] = "female" if broken.loc[
        broken.index[0], "gender"
    ] == "male" else "male"
    with pytest.raises(ValueError, match="gender != actor parity"):
        metadata.validate_frame(broken)
