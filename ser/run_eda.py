"""Phase 2 — generate every EDA figure.

Signal analysis is restricted to **training folds** (see PLAN.md's reordering
note). Only the class-balance chart uses all 1,440 clips, because counts are a
property of the corpus rather than of the test signal.

Descriptors are cached to ``features/eda_descriptors.parquet`` -- gitignored
and regenerable -- because F0 extraction takes ~13 minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ser import set_seeds
from ser.eda import (
    clip_descriptors,
    plot_class_balance,
    plot_distribution_by,
    plot_projection,
    plot_silhouette_comparison,
    plot_waveform_and_spectrogram_grid,
    save,
    training_subset,
)
from ser.features import (
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.metadata import load as load_metadata

FIGURES = Path("figures")
CACHE = Path("features") / "eda_descriptors.parquet"

#: The Phase 4 winner, so the projections show the space we actually model in.
WINNER = FeatureConfig(
    normalisation="per_speaker",
    n_mfcc=20,
    extras=("chroma", "spectral_contrast", "spectral_centroid", "zcr"),
)


def descriptors(train: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Prosodic descriptors for the training fold, cached."""
    if use_cache and CACHE.is_file():
        cached = pd.read_parquet(CACHE)
        if len(cached) == len(train):
            return cached
    frame = clip_descriptors(train)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE, index=False)
    return frame


def run(use_cache: bool = True) -> list[Path]:
    """Build and save every figure. Returns the paths written."""
    set_seeds()
    meta = load_metadata()
    train = training_subset(meta)
    print(f"training fold: {len(train)} clips, {train['actor'].nunique()} actors")

    written: list[Path] = []

    # Corpus-level -- deliberately all 1,440.
    written.append(save(plot_class_balance(meta), FIGURES / "01_class_balance.png"))

    # Signal-level -- training folds only from here down.
    written.append(
        save(
            plot_waveform_and_spectrogram_grid(train),
            FIGURES / "02_waveform_spectrogram_grid.png",
        )
    )

    print("computing prosodic descriptors (F0 is slow; cached after first run)...")
    described = descriptors(train, use_cache=use_cache)

    panels = [
        ("f0_mean", "emotion_label", "mean F0 (Hz)", "03_f0_by_emotion"),
        ("rms", "emotion_label", "RMS energy", "04_rms_by_emotion"),
        ("speech_duration", "emotion_label", "speech duration (s)", "05_duration_by_emotion"),
        ("f0_mean", "gender", "mean F0 (Hz)", "06_f0_by_gender"),
        ("rms", "intensity_label", "RMS energy", "07_rms_by_intensity"),
        ("speech_duration", "intensity_label", "speech duration (s)", "08_duration_by_intensity"),
    ]
    for column, by, label, stem in panels:
        written.append(
            save(plot_distribution_by(described, column, by=by, ylabel=label),
                 FIGURES / f"{stem}.png")
        )

    # Projections. Produced on BOTH the raw features and the per-speaker
    # normalised ones, because the comparison is the point: if actor clusters
    # are tighter than emotion clusters in the raw space, that is the leakage
    # story, and normalisation is what dissolves them. Projecting only the
    # normalised features would hide the very thing we set out to show.
    frame = extract(WINNER, meta)
    columns = feature_columns(frame, meta)
    indexed = frame.set_index("filename")

    raw = indexed.loc[train["filename"], columns].to_numpy()
    normalised = pd.DataFrame(
        per_speaker_normalise(indexed[columns].to_numpy(), indexed["actor"].to_numpy()),
        index=indexed.index, columns=columns,
    ).loc[train["filename"]].to_numpy()

    spaces = (("raw", raw, "RAW"), ("norm", normalised, "PER-SPEAKER NORMALISED"))
    colourings = (
        ("emotion", train["emotion_label"], "EMOTION"),
        ("actor", train["actor"].astype(str), "ACTOR"),
    )
    index = 9
    for method in ("pca", "umap"):
        for space_key, matrix, space_label in spaces:
            for colour_key, labels, colour_label in colourings:
                written.append(
                    save(
                        plot_projection(
                            matrix, labels, method=method,
                            title=f"{method.upper()} — {space_label} features, "
                                  f"coloured by {colour_label}",
                        ),
                        FIGURES / f"{index:02d}_{method}_{space_key}_by_{colour_key}.png",
                    )
                )
                index += 1

    written.append(save(plot_silhouette_comparison(raw, normalised, train),
                        FIGURES / "17_cluster_tightness.png"))

    for path in written:
        print("  wrote", path)
    return written


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    paths = run(use_cache=not args.no_cache)
    print(f"\n{len(paths)} figures written to {FIGURES}/")
