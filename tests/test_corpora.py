"""Invariants for the merged multi-corpus index.

The checks here fall into two groups. The *label* tests guard the promise that
merging four corpora cannot introduce a ninth class or silently relabel a
clip. The *structural* tests guard the ways a multi-corpus merge goes wrong
without anyone noticing: a filename shared by two corpora, a speaker id that
spans corpora, a label index drifting out of sync with its label.

Nothing here needs the external audio, apart from the two tests that say so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ser import corpora
from ser.metadata import DATA_ROOT, EMOTIONS


# --- 1. the eight labels are RAVDESS's, and nothing adds a ninth -----------


def test_canonical_matches_ravdess_metadata():
    assert corpora.CANONICAL == tuple(EMOTIONS[f"{i:02d}"] for i in range(1, 9))
    assert len(corpora.CANONICAL) == 8


@pytest.mark.parametrize(
    "mapping",
    [corpora.CREMAD_EMOTION, corpora.TESS_EMOTION, corpora.SAVEE_EMOTION],
)
def test_external_mappings_stay_inside_the_eight(mapping):
    assert set(mapping.values()) <= set(corpora.CANONICAL)


def test_emotion_to_code_round_trips():
    for label in corpora.CANONICAL:
        code = corpora.EMOTION_TO_CODE[label]
        assert EMOTIONS[code] == label


def test_calm_is_ravdess_only():
    """The confound the whole architecture is designed around."""
    for mapping in (
        corpora.CREMAD_EMOTION,
        corpora.TESS_EMOTION,
        corpora.SAVEE_EMOTION,
    ):
        assert "calm" not in mapping.values()


def test_cremad_has_no_surprised():
    assert "surprised" not in corpora.CREMAD_EMOTION.values()


# --- 2. the SAVEE regex, which is easy to get subtly wrong -----------------


@pytest.mark.parametrize(
    "stem,expected_code",
    [
        ("DC_sa01", "sa"),   # sadness, NOT 's' + 'a01'
        ("DC_su02", "su"),   # surprise, NOT 's' + 'u02'
        ("DC_a01", "a"),
        ("DC_d15", "d"),
        ("DC_n30", "n"),
        ("JE_h07", "h"),
        ("KL_f11", "f"),
    ],
)
def test_savee_codes_match_longest_first(stem, expected_code):
    """``sa``/``su`` must win over ``a``/``s``.

    With a naive single-character alternation every *sad* clip parses as
    *angry* and every *surprised* clip as *sad* -- a silent, total relabel of
    two classes that no count check would catch, because the totals stay
    right.
    """
    match = corpora.SAVEE_RE.match(stem)
    assert match is not None, stem
    assert match.group(2) == expected_code


def test_savee_regex_handles_folder_layout():
    """The per-speaker layout has no filename prefix; speaker comes from the dir."""
    match = corpora.SAVEE_RE.match("sa01")
    assert match is not None
    assert match.group(1) is None
    assert match.group(2) == "sa"


# --- 3. the row builder emits exactly the schema ---------------------------


def test_row_has_exactly_the_index_columns():
    row = corpora._row(
        path=Path("x/y/1001_DFA_ANG_XX.wav"),
        source="CREMA-D",
        speaker_id="CREMA-D_1001",
        gender="male",
        emotion_label="angry",
    )
    assert set(row) == set(corpora.INDEX_COLUMNS)


def test_row_label_and_index_agree():
    row = corpora._row(
        path=Path("a.wav"), source="TESS", speaker_id="TESS_OAF",
        gender="female", emotion_label="surprised",
    )
    assert row["label"] == 8
    assert row["label_idx"] == 7
    assert row["emotion"] == "08"


# --- 4. absent corpora degrade quietly, present ones do not ----------------


@pytest.mark.parametrize(
    "parser", [corpora.parse_cremad, corpora.parse_tess, corpora.parse_savee]
)
def test_parsers_return_empty_frame_when_corpus_absent(parser, tmp_path):
    frame = parser(tmp_path / "nothing-here")
    assert frame.empty
    assert list(frame.columns) == list(corpora.INDEX_COLUMNS)


def test_build_index_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        corpora.build_index(sources=["RAVDESS", "IEMOCAP"], write=False)


# --- 5. validate_index catches the silent multi-corpus failures ------------


def _valid_frame() -> pd.DataFrame:
    rows = [
        corpora._row(
            path=Path(f"f{i}.wav"), source="TESS", speaker_id="TESS_OAF",
            gender="female", emotion_label=corpora.CANONICAL[i % 8],
        )
        for i in range(8)
    ]
    return pd.DataFrame(rows, columns=list(corpora.INDEX_COLUMNS))


def test_validate_rejects_a_ninth_label():
    frame = _valid_frame()
    frame.loc[0, "emotion_label"] = "bored"
    with pytest.raises(ValueError, match="non-canonical"):
        corpora.validate_index(frame)


def test_validate_rejects_label_idx_drift():
    frame = _valid_frame()
    frame.loc[0, "label_idx"] = 99
    with pytest.raises(ValueError, match="label_idx out of sync"):
        corpora.validate_index(frame)


def test_validate_rejects_a_filename_shared_across_corpora():
    """Fold files key on filename, so a collision would merge two clips."""
    frame = _valid_frame()
    frame.loc[1, "filename"] = frame.loc[0, "filename"]
    with pytest.raises(ValueError, match="shared across corpora"):
        corpora.validate_index(frame)


def test_validate_rejects_a_speaker_spanning_two_corpora():
    """A shared speaker id would leak across any grouped split."""
    frame = _valid_frame()
    frame.loc[0, "source"] = "SAVEE"
    with pytest.raises(ValueError, match="spans two corpora"):
        corpora.validate_index(frame)


def test_validate_rejects_duplicate_filepaths():
    frame = _valid_frame()
    frame.loc[1, "filepath"] = frame.loc[0, "filepath"]
    frame.loc[1, "filename"] = "distinct.wav"
    with pytest.raises(ValueError, match="duplicate filepath"):
        corpora.validate_index(frame)


def test_validate_collects_every_problem_not_just_the_first():
    frame = _valid_frame()
    frame.loc[0, "emotion_label"] = "bored"
    frame.loc[1, "filename"] = frame.loc[2, "filename"]
    with pytest.raises(ValueError) as excinfo:
        corpora.validate_index(frame)
    message = str(excinfo.value)
    assert "non-canonical" in message and "shared across corpora" in message


# --- 6. RAVDESS goes through ser.metadata, not a second parser -------------


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="RAVDESS audio tree absent")
def test_ravdess_adapter_matches_the_canonical_parser():
    from ser.metadata import load as load_metadata

    index = corpora.parse_ravdess()
    meta = load_metadata()

    assert len(index) == len(meta) == 1440
    assert list(index.columns) == list(corpora.INDEX_COLUMNS)
    assert (index["source"] == "RAVDESS").all()
    assert index["speaker_id"].nunique() == 24

    merged = index.merge(
        meta[["filename", "emotion_label", "gender"]],
        on="filename", suffixes=("", "_meta"),
    )
    assert len(merged) == 1440
    assert (merged["emotion_label"] == merged["emotion_label_meta"]).all()
    assert (merged["gender"] == merged["gender_meta"]).all()


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="RAVDESS audio tree absent")
def test_ravdess_only_index_validates():
    index = corpora.build_index(sources=["RAVDESS"], write=False)
    corpora.validate_index(index)
    assert len(index) == 1440
