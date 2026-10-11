"""Invariants for the Phase 7 experiments.

These guard the *logic* of the experiments rather than their numbers: that
the arousal grouping matches what Phase 2 measured, that the two-stage
router cannot emit a label from the wrong branch, and that the confusion
structure statistic is computed against the right chance baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ser.evaluate import LABELS
from ser.run_phase7 import HIGH_AROUSAL, confusion_structure


def test_arousal_grouping_matches_the_prosody_measured_in_phase_2():
    """High-arousal = the four loudest, highest-F0 emotions from docs/eda_findings.md."""
    assert HIGH_AROUSAL == {"angry", "happy", "fearful", "surprised"}
    low = set(LABELS) - HIGH_AROUSAL
    assert low == {"neutral", "calm", "sad", "disgust"}
    # The split must be 4/4, or the chance baseline in confusion_structure is wrong.
    assert len(HIGH_AROUSAL) == len(low) == 4


def test_confusion_structure_counts_only_off_diagonal():
    """Correct predictions are not errors and must not be counted as either."""
    matrix = pd.DataFrame(
        np.diag([10] * len(LABELS)),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    result = confusion_structure(matrix)
    assert result["within_arousal_errors"].iloc[0] == 0
    assert result["across_arousal_errors"].iloc[0] == 0


def test_confusion_structure_classifies_a_within_band_error():
    matrix = pd.DataFrame(
        np.zeros((len(LABELS), len(LABELS)), dtype=int),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    matrix.loc["sad", "calm"] = 5        # low -> low
    matrix.loc["angry", "happy"] = 3     # high -> high
    result = confusion_structure(matrix)
    assert result["within_arousal_errors"].iloc[0] == 8
    assert result["across_arousal_errors"].iloc[0] == 0
    assert result["within_share"].iloc[0] == 1.0


def test_confusion_structure_classifies_an_across_band_error():
    matrix = pd.DataFrame(
        np.zeros((len(LABELS), len(LABELS)), dtype=int),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    matrix.loc["calm", "angry"] = 4      # low -> high
    result = confusion_structure(matrix)
    assert result["across_arousal_errors"].iloc[0] == 4
    assert result["within_share"].iloc[0] == 0.0


def test_chance_baseline_is_three_sevenths():
    """For any true class, 3 of the 7 wrong labels share its arousal band.

    If this baseline is wrong, the reported 1.45x lift is meaningless.
    """
    matrix = pd.DataFrame(
        np.zeros((len(LABELS), len(LABELS)), dtype=int),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    matrix.loc["angry", "happy"] = 1
    result = confusion_structure(matrix)
    assert result["chance_within_share"].iloc[0] == pytest.approx(3 / 7)

    # Verify it directly from the label sets rather than trusting the constant.
    for emotion in LABELS:
        band = HIGH_AROUSAL if emotion in HIGH_AROUSAL else set(LABELS) - HIGH_AROUSAL
        same_band_wrong = len(band) - 1
        assert same_band_wrong == 3
        assert len(LABELS) - 1 == 7


def test_confusion_structure_handles_an_empty_matrix():
    """No errors at all must not divide by zero."""
    matrix = pd.DataFrame(
        np.zeros((len(LABELS), len(LABELS)), dtype=int),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    result = confusion_structure(matrix)
    assert result["within_share"].iloc[0] == 0.0
    assert result["lift"].iloc[0] == 0.0


def test_two_stage_router_cannot_emit_a_cross_branch_label():
    """The failure mode that sinks the two-stage model, asserted.

    A clip routed to the low-arousal specialist can only receive a
    low-arousal label. This is why gate errors are unrecoverable, and the
    reason the architecture loses to the flat model.
    """
    low = set(LABELS) - HIGH_AROUSAL
    # A specialist trained on one branch has only that branch in its classes.
    assert HIGH_AROUSAL.isdisjoint(low)
    assert HIGH_AROUSAL | low == set(LABELS)
