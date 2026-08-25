"""Invariants for feature extraction.

The theme of this file: **no parameter that affects the numbers may be
invisible.** A teammate asked what frame duration the MFCCs use and we could
not answer, because we had silently inherited librosa's music-oriented
defaults. These tests make sure that cannot recur unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ser import SEED, set_seeds
from ser.features import (
    DEFAULT_DELTA_CONTEXT_MS,
    FeatureConfig,
    apply_scaler,
    extract_one,
    feature_columns,
    fit_scaler,
    per_speaker_normalise,
)
from ser.metadata import DATA_ROOT

CLIP = DATA_ROOT / "Person01" / "01-01-01-01-01.wav"

#: Speech-standard framing, for comparison against the inherited default.
SPEECH_FRAMING = {"n_fft": 400, "hop_length": 160}  # 25 ms / 10 ms at 16 kHz

#: CI has no audio: `.github/scripts/make_stub_audio.py` rebuilds the tree
#: from splits/folds.csv as *empty* files so the filename- and fold-based
#: tests can run for real. Anything that decodes a waveform must skip there.
#: Most of this module is pure config logic and still runs on the runner.
requires_audio = pytest.mark.skipif(
    not (CLIP.is_file() and CLIP.stat().st_size > 0),
    reason="needs the real clips; CI materialises empty stubs",
)


# --- 1. the framing is stated, in units humans use ------------------------


def test_default_framing_is_the_documented_128ms():
    config = FeatureConfig()
    assert config.window_ms == 128.0
    assert config.hop_ms == 32.0
    assert config.sr == 16_000
    # These are librosa's defaults, kept deliberately so the hygiene change
    # was a no-op. Phase 4 tests whether they are good ones.
    assert config.n_fft == 2048
    assert config.hop_length == 512


def test_speech_framing_gives_the_expected_millisecond_values():
    config = FeatureConfig(**SPEECH_FRAMING)
    assert config.window_ms == 25.0
    assert config.hop_ms == 10.0


def test_describe_names_every_parameter():
    text = FeatureConfig().describe()
    for token in ("window", "hop", "mel bands", "delta width", "aggregation"):
        assert token in text
    assert "128.0 ms" in text


# --- 2. the hygiene change must not have moved any number -----------------


@requires_audio
def test_default_config_reproduces_the_committed_baseline_values():
    """Guards the claim that making defaults explicit changed nothing.

    These values were captured from the pre-change implementation. If this
    fails, the 'explicit params' refactor silently altered the features and
    every logged result is invalidated.
    """
    vector, columns = extract_one(CLIP, FeatureConfig())
    assert len(vector) == 240
    assert len(columns) == 240
    assert columns[0] == "mfcc00_mean"
    assert columns[-1] == "dd_mfcc39_std"
    # Spot-check actual values, not just shape. These were measured from the
    # pre-change implementation and verified byte-identical afterwards
    # (sha256 of the full 1440x240 matrix: deb2b5003bf344f3, both sides).
    assert vector[0] == pytest.approx(-467.5948, abs=1e-2)
    assert float(np.abs(vector).sum()) == pytest.approx(1524.0557, rel=1e-4)


# --- 3 & 4. the cache key must track everything that changes the numbers ---


def test_framing_changes_the_hash():
    """A cache collision here would silently corrupt the whole ablation."""
    base = FeatureConfig()
    assert FeatureConfig(n_fft=400).hash != base.hash
    assert FeatureConfig(hop_length=160).hash != base.hash
    assert FeatureConfig(n_mels=40).hash != base.hash
    assert FeatureConfig(fmin=50.0).hash != base.hash
    assert FeatureConfig(delta_width=29).hash != base.hash


def test_normalisation_alone_shares_a_hash():
    """Normalisation is applied per-fold after loading, so it shares a cache."""
    base = FeatureConfig()
    assert FeatureConfig(normalisation="global").hash == base.hash
    assert FeatureConfig(normalisation="per_speaker").hash == base.hash


def test_distinct_configs_get_distinct_cache_paths():
    configs = [
        FeatureConfig(),
        FeatureConfig(**SPEECH_FRAMING),
        FeatureConfig(n_mfcc=13),
        FeatureConfig(aggregation="mean"),
        FeatureConfig(trim=False),
        FeatureConfig(n_mels=40),
    ]
    paths = {c.cache_path for c in configs}
    assert len(paths) == len(configs)


# --- 5. the delta-width coupling ------------------------------------------


def test_delta_width_holds_time_context_constant_across_hops():
    """librosa measures delta width in FRAMES, so it must scale with hop.

    Without this, changing the hop would change the delta context too, and
    an ablation result could not be attributed to either change.
    """
    coarse = FeatureConfig()                      # 32 ms hop
    fine = FeatureConfig(**SPEECH_FRAMING)        # 10 ms hop

    assert coarse.resolved_delta_width == 9
    assert fine.resolved_delta_width == 29

    for config in (coarse, fine):
        assert config.delta_context_ms == pytest.approx(
            DEFAULT_DELTA_CONTEXT_MS, abs=15
        )


def test_explicit_delta_width_is_respected():
    assert FeatureConfig(delta_width=15).resolved_delta_width == 15


def test_delta_width_must_be_odd_and_at_least_three():
    for bad in (2, 4, 10):
        with pytest.raises(ValueError, match="odd integer"):
            FeatureConfig(delta_width=bad)


# --- 6. finer framing really does give more frames ------------------------


@requires_audio
def test_speech_framing_yields_roughly_three_times_more_frames():
    import librosa

    from ser.preprocess import load_audio

    y, sr = load_audio(CLIP)
    y, _ = librosa.effects.trim(y, top_db=30)

    coarse = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).shape[1]
    fine = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, n_fft=400, hop_length=160
    ).shape[1]
    assert fine > 2.5 * coarse


# --- 7. the config name must stay stable for already-logged runs ----------


def test_default_name_matches_the_rows_already_in_results_csv():
    """24 rows in results.csv carry this exact string."""
    assert FeatureConfig().name == "mfcc40-d-dd-mean_std-trim"


def test_non_default_settings_appear_in_the_name():
    assert "win25ms-hop10ms" in FeatureConfig(**SPEECH_FRAMING).name
    assert "mel40" in FeatureConfig(n_mels=40).name
    assert "fmin50" in FeatureConfig(fmin=50.0).name
    assert "per_speaker" in FeatureConfig(normalisation="per_speaker").name
    assert FeatureConfig(n_mfcc=13).name.startswith("mfcc13")


# --- 8 & 9. seeds ----------------------------------------------------------


def test_set_seeds_makes_numpy_reproducible():
    set_seeds()
    first = np.random.rand(5)
    set_seeds()
    assert np.array_equal(first, np.random.rand(5))


def test_seed_has_exactly_one_definition():
    """SEED used to be redeclared in four files, free to drift apart."""
    from ser.metadata import SEED as metadata_seed
    from ser.models.classical import SEED as model_seed
    from ser.splits import SEED as splits_seed

    assert SEED == metadata_seed == splits_seed == model_seed == 42


# --- config validation -----------------------------------------------------


def test_unknown_aggregation_and_normalisation_raise():
    with pytest.raises(ValueError, match="aggregation must be"):
        FeatureConfig(aggregation="median")
    with pytest.raises(ValueError, match="unknown normalisation"):
        FeatureConfig(normalisation="minmax")


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("mean", 120), ("mean_std", 240), ("mean_std_min_max", 480), ("percentiles", 600)],
)
@requires_audio
def test_aggregation_controls_feature_count(aggregation, expected):
    vector, columns = extract_one(CLIP, FeatureConfig(aggregation=aggregation))
    assert len(vector) == len(columns) == expected


# --- per-speaker normalisation (the rule-2 carve-out) ----------------------


def test_per_speaker_normalise_zeroes_each_speaker_mean():
    rng = np.random.default_rng(0)
    # Two speakers with deliberately different offsets and scales.
    matrix = np.vstack([rng.normal(100, 5, (20, 4)), rng.normal(-50, 20, (20, 4))])
    speakers = np.array([1] * 20 + [2] * 20)

    out = per_speaker_normalise(matrix, speakers)

    for speaker in (1, 2):
        block = out[speakers == speaker]
        assert np.allclose(block.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(block.std(axis=0), 1.0, atol=1e-6)


def test_per_speaker_normalise_uses_no_labels():
    """The carve-out is only defensible because it consumes audio, not labels.

    Signature check: the function takes a matrix and speaker ids. There is
    no parameter it could read an emotion from.
    """
    import inspect

    params = set(inspect.signature(per_speaker_normalise).parameters)
    assert params == {"matrix", "speakers", "eps"}


def test_per_speaker_normalise_survives_a_constant_feature():
    """A speaker with zero variance must not produce inf or nan."""
    matrix = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    out = per_speaker_normalise(matrix, np.array([3, 3, 3]))
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 0], 0.0)


def test_per_speaker_normalise_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="speakers has"):
        per_speaker_normalise(np.zeros((5, 2)), np.array([1, 2]))


def test_fit_scaler_redirects_per_speaker_to_the_right_function():
    with pytest.raises(ValueError, match="per_speaker_normalise"):
        fit_scaler(np.zeros((4, 2)), "per_speaker")


def test_global_scaler_is_fit_on_train_rows_only():
    train = np.array([[0.0], [10.0]])
    test = np.array([[100.0]])
    scaler = fit_scaler(train, "global")
    assert scaler.mean_[0] == pytest.approx(5.0)   # test row not seen
    assert apply_scaler(scaler, test)[0, 0] > 10   # so it scales far out


def test_fit_scaler_none_is_a_passthrough():
    matrix = np.array([[1.0, 2.0]])
    assert fit_scaler(matrix, "none") is None
    assert np.array_equal(apply_scaler(None, matrix), matrix)
