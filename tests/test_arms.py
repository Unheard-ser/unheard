"""Invariants for the four-arm protocol.

These arms exist so that several people can compare different models on the
same question. That only works if every arm is scored on exactly the same
clips, so the tests here are mostly about the test side being untouchable.
"""

from __future__ import annotations

import pytest

from ser.arms import ARMS, ArmSplit, all_arms, arm_split, summarise
from ser.metadata import DATA_ROOT
from ser.splits import get_split

RAVDESS_PRESENT = DATA_ROOT.is_dir()
pytestmark = pytest.mark.skipif(
    not RAVDESS_PRESENT, reason="RAVDESS tree absent"
)


def _index():
    from ser import corpora

    try:
        return corpora.load_index()
    except FileNotFoundError:
        pytest.skip("merged index not built")


# --- 1. the test side is identical everywhere ------------------------------


def test_every_arm_tests_on_the_same_clips():
    """The whole point. Different test sets would make the arms incomparable."""
    index = _index()
    for fold in range(5):
        splits = all_arms(fold, index=index)
        sides = {arm: tuple(s.test) for arm, s in splits.items()}
        assert len(set(sides.values())) == 1, f"fold {fold}: arms disagree on test"


def test_test_side_matches_the_frozen_folds():
    index = _index()
    for fold in range(5):
        _, frozen = get_split("speaker_independent", fold)
        for arm in ARMS:
            assert arm_split(arm, fold, index=index).test == frozen


def test_no_external_clip_is_ever_tested():
    index = _index()
    source = index.set_index("filename")["source"].astype(str)
    for fold in range(5):
        for arm in ARMS:
            test = arm_split(arm, fold, index=index).test
            assert {str(source[n]) for n in test} == {"RAVDESS"}


# --- 2. each arm uses the training data it claims to -----------------------


def test_ravdess_only_touches_no_external_data():
    index = _index()
    source = index.set_index("filename")["source"].astype(str)
    split = arm_split("ravdess_only", 0, index=index)
    assert split.stage1 is None
    assert {str(source[n]) for n in split.stage2} == {"RAVDESS"}


def test_pooled_is_single_stage_over_everything():
    index = _index()
    split = arm_split("pooled", 0, index=index)
    assert split.stage1 is None
    assert len(split.stage2) == 11_862
    assert split.freeze_backbone is False


@pytest.mark.parametrize("arm", ["two_stage", "two_stage_frozen"])
def test_two_stage_arms_pretrain_on_all_and_finetune_on_ravdess(arm):
    index = _index()
    source = index.set_index("filename")["source"].astype(str)
    split = arm_split(arm, 0, index=index)
    assert split.stage1 is not None and len(split.stage1) == 11_862
    assert {str(source[n]) for n in split.stage2} == {"RAVDESS"}
    assert len(split.stage2) == 1_140


def test_only_the_frozen_arm_freezes():
    index = _index()
    flags = {a: arm_split(a, 0, index=index).freeze_backbone for a in ARMS}
    assert flags == {
        "ravdess_only": False, "pooled": False,
        "two_stage": False, "two_stage_frozen": True,
    }


def test_two_stage_arms_share_their_data():
    """They differ only in whether stage 2 may move the representation."""
    index = _index()
    a = arm_split("two_stage", 0, index=index)
    b = arm_split("two_stage_frozen", 0, index=index)
    assert a.stage1 == b.stage1 and a.stage2 == b.stage2 and a.test == b.test
    assert a.freeze_backbone != b.freeze_backbone


# --- 3. no clip is both trained on and tested ------------------------------


@pytest.mark.parametrize("arm", list(ARMS))
def test_training_and_test_never_overlap(arm):
    index = _index()
    for fold in range(5):
        s = arm_split(arm, fold, index=index)
        train = set(s.stage2) | set(s.stage1 or [])
        assert not train & set(s.test), f"{arm} fold {fold}: leak"


# --- 4. interface ----------------------------------------------------------


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        arm_split("kitchen_sink", 0)


def test_summarise_covers_every_arm_and_fold():
    table = summarise(_index())
    assert len(table) == len(ARMS) * 5
    assert set(table.arm) == set(ARMS)


def test_describe_is_a_single_line():
    line = arm_split("two_stage_frozen", 0, index=_index()).describe()
    assert "\n" not in line and "two_stage_frozen" in line
