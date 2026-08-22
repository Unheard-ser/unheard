"""Invariants for the shared audio-loading path.

These guard the two decisions in docs/data_audit.md that must hold at every
call site: always 1-D, and never the silent librosa 22050 default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ser.metadata import DATA_ROOT, load
from ser.preprocess import NATIVE_SR, TARGET_SR, channel_report, load_audio

MONO_CLIP = DATA_ROOT / "Person01" / "01-01-01-01-01.wav"
STEREO_CLIP = DATA_ROOT / "Person01" / "02-01-01-02-01.wav"

KNOWN_STEREO = {
    "02-01-01-02-01.wav",
    "08-01-02-02-01.wav",
    "02-01-02-02-05.wav",
    "03-01-02-01-20.wav",
    "06-01-01-02-20.wav",
}


# --- the sample-rate decision ---------------------------------------------


def test_default_rate_is_the_project_target_not_the_librosa_default():
    """librosa.load defaults to 22050; we must never inherit that."""
    assert TARGET_SR == 16_000
    _, sr = load_audio(MONO_CLIP)
    assert sr == TARGET_SR
    assert sr != 22_050


def test_sr_none_preserves_the_native_rate():
    _, sr = load_audio(MONO_CLIP, sr=None)
    assert sr == NATIVE_SR == 48_000


def test_resampling_preserves_duration():
    y_native, sr_native = load_audio(MONO_CLIP, sr=None)
    y_target, sr_target = load_audio(MONO_CLIP)
    assert len(y_native) / sr_native == pytest.approx(
        len(y_target) / sr_target, abs=1e-3
    )


# --- the channel decision --------------------------------------------------


@pytest.mark.parametrize("clip", [MONO_CLIP, STEREO_CLIP])
def test_load_audio_always_returns_1d_float32(clip):
    y, _ = load_audio(clip)
    assert y.ndim == 1
    assert y.dtype == np.float32


def test_downmix_of_the_stereo_files_is_lossless():
    """Averaging bit-identical channels returns the original signal."""
    report = channel_report(STEREO_CLIP)
    assert report["n_channels"] == 2
    assert report["max_channel_diff"] == 0.0

    raw, _ = sf.read(str(STEREO_CLIP), always_2d=True)
    y, _ = load_audio(STEREO_CLIP, sr=None)
    assert np.allclose(y, raw[:, 0], atol=1e-7)


def test_channel_report_on_mono():
    report = channel_report(MONO_CLIP)
    assert report["n_channels"] == 1
    assert report["is_multichannel"] is False
    assert report["max_channel_diff"] == 0.0


@pytest.mark.slow
def test_exactly_five_dual_channel_files_and_all_are_duplicated_mono():
    """The documented finding, asserted across the whole corpus.

    If a re-download changes this, the downmix stops being lossless and the
    audit doc's claim needs revisiting -- so fail loudly.
    """
    multichannel = {}
    for path in load()["filepath"]:
        report = channel_report(path)
        if report["is_multichannel"]:
            multichannel[Path(path).name] = report["max_channel_diff"]

    assert set(multichannel) == KNOWN_STEREO
    assert all(diff == 0.0 for diff in multichannel.values()), (
        "a dual-channel file has genuinely differing channels; "
        "the lossless-downmix justification no longer holds"
    )


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_audio(DATA_ROOT / "Person01" / "does-not-exist.wav")

