"""Invariants for the EDA module.

Two things matter here and neither is about how a chart looks:

1. **Plot functions return a Figure and never show it.** That is what makes
   them testable at all, and it keeps output paths out of the plotting code.
2. **`training_subset` genuinely excludes held-out clips.** The whole reason
   Phase 3 was run before Phase 2 is so EDA cannot see the test set. If this
   guard is wrong, that reordering bought nothing.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ser.eda import (
    EMOTION_ORDER,
    HIGH_AROUSAL,
    plot_class_balance,
    plot_distribution_by,
    plot_projection,
    save,
    training_subset,
)
from ser.metadata import load
from ser.splits import get_split


@pytest.fixture(scope="module")
def meta():
    return load()


@pytest.fixture(scope="module")
def train(meta):
    return training_subset(meta)


# --- the leakage guard ------------------------------------------------------


def test_training_subset_excludes_every_held_out_clip(meta, train):
    """The invariant the Phase 3-before-2 reordering exists to enable."""
    train_files, test_files = get_split("speaker_independent", 0)
    selected = set(train["filename"])
    assert selected == set(train_files)
    assert selected & set(test_files) == set()


def test_training_subset_excludes_every_held_out_actor(meta, train):
    """Stronger than clip-level: no held-out *speaker* is visible either."""
    _, test_files = get_split("speaker_independent", 0)
    test_actors = set(meta[meta["filename"].isin(test_files)]["actor"])
    assert set(train["actor"]) & test_actors == set()
    assert len(train) == 1140
    assert train["actor"].nunique() == 19


def test_training_subset_respects_the_fold_argument(meta):
    first = training_subset(meta, fold=0)
    second = training_subset(meta, fold=1)
    assert set(first["actor"]) != set(second["actor"])


# --- plot functions return figures and do not display ----------------------


def test_matplotlib_uses_a_headless_backend():
    """A plot that opens a window would hang CI."""
    assert matplotlib.get_backend().lower() == "agg"


def test_class_balance_returns_a_figure(meta):
    fig = plot_class_balance(meta)
    assert isinstance(fig, Figure)
    axis = fig.axes[0]
    assert axis.get_ylabel() == "clips"
    # The neutral asymmetry must be annotated, not left for the reader.
    texts = " ".join(t.get_text() for t in axis.texts)
    assert "neutral" in texts.lower()


def test_distribution_plot_returns_a_figure_with_one_box_per_level(train):
    frame = train.assign(dummy=np.linspace(0, 1, len(train)))
    fig = plot_distribution_by(frame, "dummy", by="emotion_label", ylabel="dummy")
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_ylabel() == "dummy"
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == [e for e in EMOTION_ORDER if e in set(train["emotion_label"])]


def test_distribution_plot_groups_by_gender_too(train):
    frame = train.assign(dummy=np.linspace(0, 1, len(train)))
    fig = plot_distribution_by(frame, "dummy", by="gender")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["female", "male"]


def test_projection_returns_a_figure_with_one_series_per_label():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8))
    labels = pd.Series(["angry", "calm", "sad"] * 20)
    fig = plot_projection(features, labels, method="pca")
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) == 3
    assert "PC1" in fig.axes[0].get_xlabel()


def test_projection_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="pca.*umap"):
        plot_projection(np.zeros((4, 2)), pd.Series(list("abab")), method="tsne")


# --- saving -----------------------------------------------------------------


def test_save_writes_a_file_and_closes_the_figure(meta, tmp_path):
    import matplotlib.pyplot as plt

    fig = plot_class_balance(meta)
    path = save(fig, tmp_path / "sub" / "chart.png")
    assert path.is_file()
    assert path.stat().st_size > 1000
    assert fig not in [plt.figure(num) for num in plt.get_fignums()]


# --- the arousal grouping used by several figures --------------------------


def test_arousal_split_covers_the_emotions_it_claims():
    assert HIGH_AROUSAL <= set(EMOTION_ORDER)
    low = set(EMOTION_ORDER) - HIGH_AROUSAL
    assert low == {"neutral", "calm", "sad", "disgust"}
