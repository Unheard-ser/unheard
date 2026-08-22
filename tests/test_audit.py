"""Invariants for the raw-audio audit."""

from __future__ import annotations

import numpy as np
import pytest

from ser.audit import CLIP_THRESHOLD, probe_file, run_audit, summarise
from ser.metadata import DATA_ROOT, load

#: A known-good mono clip.
MONO_CLIP = DATA_ROOT / "Person01" / "01-01-01-01-01.wav"

#: One of the five dual-channel clips found during the Phase 0 audit.
STEREO_CLIP = DATA_ROOT / "Person01" / "02-01-01-02-01.wav"

#: All five, so a repackaging that changes them is caught.
KNOWN_STEREO = {
    "02-01-01-02-01.wav",
    "08-01-02-02-01.wav",
    "02-01-02-02-05.wav",
    "03-01-02-01-20.wav",
    "06-01-01-02-20.wav",
}


@pytest.fixture(scope="module")
def sample_audit():
    """A 20-clip audit, spread across actors rather than taken from one."""
    df = load().iloc[::72].head(20).reset_index(drop=True)
    return run_audit(df, cache=None)


# --- 1. container format ---------------------------------------------------


def test_probe_reports_native_format():
    got = probe_file(MONO_CLIP)
    assert got["load_error"] is None
    assert got["sr"] == 48000, "must not be the librosa 22050 default"
    assert got["subtype"] == "PCM_16"
    assert got["n_channels"] == 1


# --- 2. the stereo files ---------------------------------------------------


def test_stereo_file_reported_as_stereo_but_downmixed():
    """Channel count comes from the header, before the mono downmix."""
    got = probe_file(STEREO_CLIP)
    assert got["load_error"] is None
    assert got["n_channels"] == 2
    # The downmixed signal still yields scalar amplitude measurements.
    assert 0.0 < float(got["peak_amp"]) <= 1.0
    assert got["duration"] == pytest.approx(got["n_frames"] / got["sr"])


def test_known_stereo_set_is_stable(sample_audit):
    """Guard the documented finding: exactly 5 dual-channel files exist."""
    full = run_audit(load(), cache=None)
    stereo = set(full.loc[full["n_channels"] > 1, "filename"])
    assert stereo == KNOWN_STEREO


# --- 3. measurement bounds -------------------------------------------------


def test_silence_and_duration_bounds(sample_audit):
    a = sample_audit
    assert (a["silence_fraction"] >= 0.0).all()
    assert (a["silence_fraction"] < 1.0).all()
    assert (a["trimmed_duration"] <= a["duration"] + 1e-9).all()
    assert (a["duration"] > 0).all()
    assert (a["lead_silence"] >= 0).all()
    assert (a["trail_silence"] >= 0).all()
    assert (a["peak_amp"] >= a["rms"]).all()


def test_clipping_flag_agrees_with_peak(sample_audit):
    """n_clipped can only be non-zero if the peak reaches the threshold."""
    quiet = sample_audit[sample_audit["peak_amp"] < CLIP_THRESHOLD]
    assert (quiet["n_clipped"] == 0).all()


# --- 4. the audit pass -----------------------------------------------------


def test_sample_audit_shape_and_no_errors(sample_audit):
    assert len(sample_audit) == 20
    assert sample_audit["load_error"].isna().all()
    assert sample_audit["actor"].nunique() > 1, "sample should span actors"


def test_audit_preserves_metadata_columns(sample_audit):
    for col in ("emotion_label", "gender", "intensity_label", "actor"):
        assert col in sample_audit.columns
    assert not sample_audit["filepath"].duplicated().any()


def test_probe_records_error_instead_of_raising(tmp_path):
    """A bad file must be recorded, not abort a 1,440-file pass."""
    bad = tmp_path / "not-audio.wav"
    bad.write_bytes(b"this is not a wav file")
    got = probe_file(bad)
    assert got["load_error"] is not None
    assert np.isnan(got["duration"])


# --- 5. full pass ----------------------------------------------------------


@pytest.mark.slow
def test_full_audit_covers_every_clip():
    meta = load()
    audit = run_audit(meta, cache=None)
    assert len(audit) == len(meta) == 1440
    assert audit["load_error"].isna().all(), "no clip may fail to load"

    stats = summarise(audit)
    assert stats["sample_rates"] == {48000: 1440}
    assert stats["subtypes"] == {"PCM_16": 1440}
    assert stats["n_zero_length"] == 0
    assert set(stats["clips_per_actor"].values()) == {60}

    # Clipping is negligible but not zero: exactly one sample in one file
    # (Person10/03-02-02-01-10.wav). Asserted so a different dataset copy or
    # a changed threshold is noticed rather than assumed away.
    assert stats["n_files_with_clipping"] == 1
    assert stats["total_clipped_samples"] == 1


@pytest.mark.slow
def test_loudness_is_driven_by_emotion_not_by_actor():
    """The audit's central finding, guarded as an invariant.

    Peak amplitude varies far more across emotion/intensity than across
    actors. This is why clips are NOT peak-normalised -- doing so would
    erase a real cue. See docs/data_audit.md.
    """
    audit = run_audit(load(), cache=None)
    by_emotion = audit.groupby(["emotion_label", "intensity_label"])["peak_amp"].mean()
    by_actor = audit.groupby("actor")["rms"].mean()

    emotion_spread = by_emotion.max() / by_emotion.min()
    actor_spread = by_actor.max() / by_actor.min()
    assert emotion_spread > 3 * actor_spread
