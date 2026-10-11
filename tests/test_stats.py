"""Invariants for the uncertainty estimates.

These matter because a confidence interval is a claim about the experiment,
and a wrong one is worse than none: it makes noise look like a finding. The
central property tested here is that the bootstrap resamples **speakers**, not
clips -- the clip-level version understates variance in exactly the direction
that flatters a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from ser.stats import (
    accuracy,
    cluster_bootstrap_ci,
    format_ci,
    macro_f1,
    mcnemar,
    paired_cluster_bootstrap,
)

LABELS = ("angry", "happy", "sad", "neutral")


@pytest.fixture
def toy():
    """24 speakers x 20 clips, with a model that is right 70% of the time."""
    rng = np.random.default_rng(0)
    speakers = np.repeat([f"a{i:02d}" for i in range(24)], 20)
    y_true = rng.choice(LABELS, size=len(speakers))
    y_pred = np.where(
        rng.random(len(speakers)) < 0.70, y_true, rng.choice(LABELS, size=len(speakers))
    )
    return y_true, y_pred, speakers


# --- 1. the point estimate is the real metric ------------------------------


def test_point_estimate_matches_the_metric(toy):
    y_true, y_pred, speakers = toy
    result = cluster_bootstrap_ci(y_true, y_pred, speakers, "macro_f1", n_boot=200)
    assert result["point"] == pytest.approx(macro_f1(y_true, y_pred))


def test_interval_brackets_the_point_estimate(toy):
    y_true, y_pred, speakers = toy
    result = cluster_bootstrap_ci(y_true, y_pred, speakers, "accuracy", n_boot=400)
    assert result["lo"] <= result["point"] <= result["hi"]


def test_reports_the_number_of_resampling_units(toy):
    y_true, y_pred, speakers = toy
    result = cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=100)
    assert result["n_speakers"] == 24


# --- 2. clustering actually changes the answer -----------------------------


def test_clustered_interval_is_wider_than_the_clip_level_one():
    """The whole reason this module exists.

    Clips from one speaker succeed or fail together. Pretending they are
    independent shrinks the interval and manufactures significance. Here every
    speaker is either wholly right or wholly wrong, which is the extreme case:
    the clip-level bootstrap sees 480 observations, the correct one sees 24.
    """
    rng = np.random.default_rng(1)
    speakers = np.repeat([f"a{i:02d}" for i in range(24)], 20)
    y_true = rng.choice(LABELS, size=len(speakers))

    good = {f"a{i:02d}" for i in range(16)}  # two-thirds of speakers perfect
    y_pred = np.array(
        [
            t if s in good else ("angry" if t != "angry" else "sad")
            for t, s in zip(y_true, speakers, strict=True)
        ],
        dtype=object,
    )

    clustered = cluster_bootstrap_ci(y_true, y_pred, speakers, "accuracy", n_boot=600)
    per_clip = cluster_bootstrap_ci(
        y_true, y_pred, np.arange(len(y_true)).astype(str), "accuracy", n_boot=600
    )

    clustered_width = clustered["hi"] - clustered["lo"]
    per_clip_width = per_clip["hi"] - per_clip["lo"]
    assert clustered_width > 3 * per_clip_width


# --- 3. paired comparison ---------------------------------------------------


def test_identical_models_have_zero_difference_and_are_not_significant(toy):
    y_true, y_pred, speakers = toy
    result = paired_cluster_bootstrap(y_true, y_pred, y_pred, speakers, n_boot=200)
    assert result["delta"] == pytest.approx(0.0)
    assert result["significant"] is False
    assert result["p_value"] == pytest.approx(1.0)


def test_a_large_real_difference_is_detected(toy):
    """A clearly better model must come out significant, or the test is useless."""
    y_true, y_pred, speakers = toy
    rng = np.random.default_rng(2)
    worse = rng.choice(LABELS, size=len(y_true))  # pure noise

    result = paired_cluster_bootstrap(
        y_true, y_pred, worse, speakers, "macro_f1", n_boot=600
    )
    assert result["delta"] > 0
    assert result["lo"] > 0
    assert result["significant"] is True


def test_delta_sign_follows_argument_order(toy):
    y_true, y_pred, speakers = toy
    rng = np.random.default_rng(3)
    worse = rng.choice(LABELS, size=len(y_true))

    forward = paired_cluster_bootstrap(y_true, y_pred, worse, speakers, n_boot=200)
    backward = paired_cluster_bootstrap(y_true, worse, y_pred, speakers, n_boot=200)
    assert forward["delta"] == pytest.approx(-backward["delta"])


# --- 4. determinism ---------------------------------------------------------


def test_same_seed_gives_the_same_interval(toy):
    y_true, y_pred, speakers = toy
    a = cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=200, seed=7)
    b = cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=200, seed=7)
    assert a == b


def test_different_seeds_give_different_draws(toy):
    y_true, y_pred, speakers = toy
    a = cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=200, seed=7)
    b = cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=200, seed=8)
    assert a["lo"] != b["lo"] or a["hi"] != b["hi"]


# --- 5. McNemar -------------------------------------------------------------


def test_mcnemar_on_identical_predictions_is_not_significant(toy):
    y_true, y_pred, _ = toy
    result = mcnemar(y_true, y_pred, y_pred)
    assert result["only_a_correct"] == result["only_b_correct"] == 0
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_mcnemar_counts_only_discordant_clips():
    y_true = np.array(["a", "a", "b", "b", "c"], dtype=object)
    pred_a = np.array(["a", "a", "b", "x", "x"], dtype=object)  # right on 1,2,3
    pred_b = np.array(["a", "x", "x", "b", "x"], dtype=object)  # right on 1,4
    result = mcnemar(y_true, pred_a, pred_b)
    assert result["only_a_correct"] == 2  # clips 2 and 3
    assert result["only_b_correct"] == 1  # clip 4


def test_mcnemar_detects_a_one_sided_difference():
    y_true = np.array(["a"] * 100, dtype=object)
    pred_a = np.array(["a"] * 100, dtype=object)
    pred_b = np.array(["a"] * 40 + ["b"] * 60, dtype=object)
    result = mcnemar(y_true, pred_a, pred_b)
    assert result["only_a_correct"] == 60 and result["only_b_correct"] == 0
    assert result["significant"] is True


# --- 6. input validation and formatting ------------------------------------


def test_length_mismatch_raises(toy):
    y_true, y_pred, speakers = toy
    with pytest.raises(ValueError, match="length mismatch"):
        cluster_bootstrap_ci(y_true, y_pred[:-1], speakers, n_boot=10)


def test_unknown_metric_raises(toy):
    y_true, y_pred, speakers = toy
    with pytest.raises(ValueError, match="unknown metric"):
        cluster_bootstrap_ci(y_true, y_pred, speakers, "precision", n_boot=10)


def test_custom_metric_callable_is_accepted(toy):
    y_true, y_pred, speakers = toy
    result = cluster_bootstrap_ci(y_true, y_pred, speakers, accuracy, n_boot=100)
    assert result["point"] == pytest.approx(accuracy(y_true, y_pred))


def test_format_ci_renders_absolute_and_paired_results(toy):
    y_true, y_pred, speakers = toy
    absolute = format_ci(cluster_bootstrap_ci(y_true, y_pred, speakers, n_boot=100))
    assert "[" in absolute and "]" in absolute

    paired = format_ci(
        paired_cluster_bootstrap(y_true, y_pred, y_pred, speakers, n_boot=100),
        as_pp=True,
    )
    assert "pp" in paired and "+0.00" in paired
