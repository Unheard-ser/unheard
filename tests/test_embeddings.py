"""Invariants for the frozen-embedding layer.

None of these load a pretrained model. They cover the parts that are easy to
get quietly wrong: the cache key (which decides whether an expensive forward
pass is reused or silently repeated), the layer/pooling selection arithmetic,
and the algebraic claim that makes the layer-weighted-sum track affordable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ser.embeddings import (
    POOLINGS,
    EmbeddingConfig,
    feature_names,
    select,
)
from ser.preprocess import TARGET_SR


# --- 1. configuration validation -------------------------------------------


def test_defaults_match_the_project_sample_rate():
    """wav2vec2 is trained at 16 kHz and the project targets 16 kHz."""
    assert EmbeddingConfig().sr == TARGET_SR == 16_000


@pytest.mark.parametrize("pooling", POOLINGS)
def test_valid_poolings_accepted(pooling):
    assert EmbeddingConfig(pooling=pooling).pooling == pooling


def test_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="pooling must be"):
        EmbeddingConfig(pooling="median")


def test_rejects_unknown_normalisation():
    with pytest.raises(ValueError, match="unknown normalisation"):
        EmbeddingConfig(normalisation="per_corpus")


def test_rejects_unknown_layer_keyword():
    with pytest.raises(ValueError, match="layer must be"):
        EmbeddingConfig(layer="middle")


def test_rejects_nonpositive_max_duration():
    with pytest.raises(ValueError, match="max_duration_s"):
        EmbeddingConfig(max_duration_s=0)


# --- 2. the cache key ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer": 5},
        {"layer": "weighted"},
        {"pooling": "mean_std"},
        {"normalisation": "per_speaker"},
        {"batch_size": 32},
    ],
)
def test_selection_parameters_share_one_cache(kwargs):
    """A layer or pooling choice must never trigger another forward pass.

    One `extract` call stores every layer's pooled statistics, so these are
    slices of an existing cache. If they entered the hash, sweeping five
    layers would cost five CPU-hours instead of one.
    """
    assert EmbeddingConfig(**kwargs).hash == EmbeddingConfig().hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "microsoft/wavlm-base-plus"},
        {"sr": 8_000},
        {"max_duration_s": 4.0},
        {"peak_norm": True},
        {"trim": True},
        {"top_db": 20},
    ],
)
def test_extraction_parameters_change_the_cache(kwargs):
    """Anything that changes the stored numbers must change the key."""
    assert EmbeddingConfig(**kwargs).hash != EmbeddingConfig().hash


def test_hash_is_stable_across_instances():
    assert EmbeddingConfig().hash == EmbeddingConfig().hash
    assert len(EmbeddingConfig().hash) == 12


def test_cache_and_order_paths_share_the_hash():
    config = EmbeddingConfig()
    assert config.hash in config.cache_path.name
    assert config.hash in config.order_path.name
    assert config.cache_path.suffix == ".npy"


def test_name_reports_the_choices_that_do_not_enter_the_hash():
    name = EmbeddingConfig(layer=6, pooling="mean_std", normalisation="per_speaker").name
    assert "wav2vec2-base" in name
    assert "L6" in name and "mean_std" in name and "per_speaker" in name


# --- 3. selecting from a cached tensor -------------------------------------


@pytest.fixture
def tensor() -> np.ndarray:
    """(clips=4, layers=13, [mean,std]=2, dim=6), deterministic."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(4, 13, 2, 6)).astype(np.float32)


def test_last_resolves_to_the_final_layer(tensor):
    chosen = select(tensor, EmbeddingConfig(layer="last", pooling="mean"))
    assert np.allclose(chosen, tensor[:, -1, 0, :])


def test_integer_layer_indexes_directly(tensor):
    chosen = select(tensor, EmbeddingConfig(layer=3, pooling="mean"))
    assert np.allclose(chosen, tensor[:, 3, 0, :])


def test_std_pooling_reads_the_second_slot(tensor):
    chosen = select(tensor, EmbeddingConfig(layer=2, pooling="std"))
    assert np.allclose(chosen, tensor[:, 2, 1, :])


def test_mean_std_concatenates_and_doubles_width(tensor):
    chosen = select(tensor, EmbeddingConfig(layer=1, pooling="mean_std"))
    assert chosen.shape == (4, 12)
    assert np.allclose(chosen[:, :6], tensor[:, 1, 0, :])
    assert np.allclose(chosen[:, 6:], tensor[:, 1, 1, :])


def test_layer_out_of_range_raises(tensor):
    with pytest.raises(ValueError, match="out of range"):
        select(tensor, EmbeddingConfig(layer=99))


def test_weighted_requires_weights(tensor):
    with pytest.raises(ValueError, match="requires layer_weights"):
        select(tensor, EmbeddingConfig(layer="weighted"))


def test_weighted_rejects_wrong_length(tensor):
    with pytest.raises(ValueError, match="must have shape"):
        select(tensor, EmbeddingConfig(layer="weighted"), layer_weights=np.ones(3))


def test_weighted_one_hot_equals_picking_that_layer(tensor):
    """A one-hot weighting must be exactly the single-layer selection."""
    weights = np.zeros(13, dtype=np.float32)
    weights[7] = 1.0
    weighted = select(tensor, EmbeddingConfig(layer="weighted"), layer_weights=weights)
    direct = select(tensor, EmbeddingConfig(layer=7))
    assert np.allclose(weighted, direct, atol=1e-6)


def test_weighted_sum_commutes_with_mean_pooling():
    """The identity that makes the SUPERB track cost no extra forward passes.

    Weighting layers and then averaging frames gives the same answer as
    averaging frames and then weighting layers, because both are linear:
    ``sum_i w_i * mean(h_i) == mean(sum_i w_i * h_i)``. That is why the cache
    can store per-layer means rather than per-layer frame sequences.
    """
    rng = np.random.default_rng(1)
    clips, layers, frames, dim = 3, 13, 17, 5
    states = rng.normal(size=(clips, layers, frames, dim))
    weights = rng.random(layers)
    weights /= weights.sum()

    # Route A: weight the full frame sequences, then pool.
    weighted_states = np.tensordot(weights, states, axes=([0], [1]))
    route_a = weighted_states.mean(axis=1)

    # Route B: pool first (what the cache stores), then weight.
    pooled = np.stack([states.mean(axis=2), states.std(axis=2)], axis=2)
    route_b = select(
        pooled.astype(np.float32),
        EmbeddingConfig(layer="weighted", pooling="mean"),
        layer_weights=weights,
    )

    assert np.allclose(route_a, route_b, atol=1e-5)


# --- 4. column naming ------------------------------------------------------


@pytest.mark.parametrize("pooling,dim,expected", [
    ("mean", 768, 768),
    ("std", 768, 768),
    ("mean_std", 1536, 1536),
])
def test_feature_names_length_matches_the_matrix(pooling, dim, expected):
    names = feature_names(dim, pooling)
    assert len(names) == expected
    assert len(set(names)) == expected
