"""Feature extraction — parameterised, because comparing configs is the experiment.

Every axis below is an argument, not a constant: MFCC count, deltas,
aggregation strategy, trimming, and normalisation. Phase 4 sweeps them. This
module deliberately does not privilege one setting as "correct".

Caching
-------
Each configuration is hashed and cached to ``features/{hash}.parquet``. The
cache is gitignored and regenerable (CLAUDE.md rule 7); an existing config is
never recomputed.

Normalisation and leakage
-------------------------
`extract` applies only **per-clip** transforms — nothing that requires
statistics pooled across clips. Global and per-speaker z-scoring are fit on
training folds only and therefore live in `fit_scaler` / `apply_scaler`,
which take an explicit train/test split. Doing it any other way would fit on
test data (CLAUDE.md rule 2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from ser.metadata import load as load_metadata
from ser.preprocess import TARGET_SR, load_audio

#: Gitignored, regenerable.
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "features"

AGGREGATIONS: tuple[str, ...] = ("mean", "mean_std", "mean_std_min_max", "percentiles")


@dataclass(frozen=True)
class FeatureConfig:
    """One point in the feature-design space.

    Attributes:
        n_mfcc: Number of MFCC coefficients.
        deltas: Append first-order deltas.
        delta_deltas: Append second-order deltas.
        aggregation: How frame-level features collapse to a fixed vector.
        trim: Remove leading/trailing silence before extraction.
        top_db: Silence threshold for trimming, in dB below peak.
        sr: Sample rate to load at.
        extras: Optional descriptors — any of ``chroma``,
            ``spectral_contrast``, ``spectral_centroid``, ``rolloff``,
            ``zcr``, ``rms``.
        normalisation: ``none`` | ``global`` | ``per_speaker``. Applied by
            `fit_scaler`, never inside `extract`, because it needs
            train-fold statistics.
    """

    n_mfcc: int = 40
    deltas: bool = True
    delta_deltas: bool = True
    aggregation: str = "mean_std"
    trim: bool = True
    top_db: int = 30
    sr: int = TARGET_SR
    extras: tuple[str, ...] = field(default_factory=tuple)
    normalisation: str = "none"

    def __post_init__(self) -> None:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}"
            )
        if self.normalisation not in ("none", "global", "per_speaker"):
            raise ValueError(f"unknown normalisation {self.normalisation!r}")

    @property
    def hash(self) -> str:
        """Stable 12-char hash of everything that affects the stored values.

        ``normalisation`` is excluded: it is applied per-fold after loading,
        so configs differing only in normalisation share one cache entry.
        """
        payload = {k: v for k, v in asdict(self).items() if k != "normalisation"}
        payload["extras"] = sorted(payload["extras"])
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        """Short human-readable identifier, for ``results.csv``."""
        parts = [f"mfcc{self.n_mfcc}"]
        if self.deltas:
            parts.append("d")
        if self.delta_deltas:
            parts.append("dd")
        parts.append(self.aggregation)
        parts.append("trim" if self.trim else "notrim")
        if self.extras:
            parts.append("+".join(sorted(self.extras)))
        if self.normalisation != "none":
            parts.append(self.normalisation)
        return "-".join(parts)

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"features_{self.hash}.parquet"


def _aggregate(matrix: np.ndarray, strategy: str) -> tuple[np.ndarray, list[str]]:
    """Collapse a (n_features, n_frames) matrix to a fixed-length vector.

    Args:
        matrix: Frame-level features.
        strategy: One of `AGGREGATIONS`.

    Returns:
        ``(vector, suffixes)`` where ``suffixes`` names each statistic in the
        order it was concatenated.
    """
    if strategy == "mean":
        return matrix.mean(axis=1), ["mean"]
    if strategy == "mean_std":
        return (
            np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)]),
            ["mean", "std"],
        )
    if strategy == "mean_std_min_max":
        return (
            np.concatenate(
                [
                    matrix.mean(axis=1),
                    matrix.std(axis=1),
                    matrix.min(axis=1),
                    matrix.max(axis=1),
                ]
            ),
            ["mean", "std", "min", "max"],
        )
    percentiles = np.percentile(matrix, [10, 25, 50, 75, 90], axis=1)
    return percentiles.reshape(-1), ["p10", "p25", "p50", "p75", "p90"]


def _frame_features(
    y: np.ndarray, sr: int, config: FeatureConfig
) -> tuple[np.ndarray, list[str]]:
    """Build the frame-level feature matrix for one clip."""
    blocks: list[np.ndarray] = []
    names: list[str] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc)
    blocks.append(mfcc)
    names += [f"mfcc{i:02d}" for i in range(config.n_mfcc)]

    if config.deltas:
        blocks.append(librosa.feature.delta(mfcc))
        names += [f"d_mfcc{i:02d}" for i in range(config.n_mfcc)]
    if config.delta_deltas:
        blocks.append(librosa.feature.delta(mfcc, order=2))
        names += [f"dd_mfcc{i:02d}" for i in range(config.n_mfcc)]

    extras = set(config.extras)
    if "chroma" in extras:
        block = librosa.feature.chroma_stft(y=y, sr=sr)
        blocks.append(block)
        names += [f"chroma{i:02d}" for i in range(block.shape[0])]
    if "spectral_contrast" in extras:
        block = librosa.feature.spectral_contrast(y=y, sr=sr)
        blocks.append(block)
        names += [f"contrast{i:02d}" for i in range(block.shape[0])]
    if "spectral_centroid" in extras:
        blocks.append(librosa.feature.spectral_centroid(y=y, sr=sr))
        names.append("centroid")
    if "rolloff" in extras:
        blocks.append(librosa.feature.spectral_rolloff(y=y, sr=sr))
        names.append("rolloff")
    if "zcr" in extras:
        blocks.append(librosa.feature.zero_crossing_rate(y))
        names.append("zcr")
    if "rms" in extras:
        blocks.append(librosa.feature.rms(y=y))
        names.append("rms")

    return np.vstack(blocks), names


def extract_one(path: str | Path, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Extract the feature vector for a single clip.

    Only per-clip transforms are applied. Nothing here uses statistics
    pooled across clips, so calling this on a test clip cannot leak.

    Args:
        path: Path to a ``.wav`` file.
        config: The feature configuration.

    Returns:
        ``(vector, column_names)``.
    """
    y, sr = load_audio(path, sr=config.sr)
    if config.trim:
        y, _ = librosa.effects.trim(y, top_db=config.top_db)
    matrix, base_names = _frame_features(y, sr, config)
    vector, suffixes = _aggregate(matrix, config.aggregation)
    columns = [f"{name}_{suffix}" for suffix in suffixes for name in base_names]
    return vector, columns


def extract(
    config: FeatureConfig | None = None,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Extract features for every clip, caching by config hash.

    Args:
        config: Feature configuration. Defaults to `FeatureConfig()`.
        df: Metadata frame. Defaults to the full inventory.
        use_cache: Read from and write to the parquet cache.

    Returns:
        Metadata columns plus one column per feature, in metadata row order.
    """
    config = config or FeatureConfig()
    if df is None:
        df = load_metadata()

    if use_cache and config.cache_path.is_file():
        cached = pd.read_parquet(config.cache_path)
        if len(cached) == len(df) and (
            cached["filename"].to_numpy() == df["filename"].to_numpy()
        ).all():
            return cached

    vectors: list[np.ndarray] = []
    columns: list[str] | None = None
    for path in df["filepath"]:
        vector, names = extract_one(path, config)
        if columns is None:
            columns = names
        vectors.append(vector)

    features = pd.DataFrame(np.vstack(vectors), columns=columns)
    out = pd.concat([df.reset_index(drop=True), features], axis=1)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(config.cache_path, index=False)
    return out


def feature_columns(frame: pd.DataFrame, meta: pd.DataFrame | None = None) -> list[str]:
    """Names of the numeric feature columns, excluding metadata."""
    meta_cols = set((meta if meta is not None else load_metadata()).columns)
    return [c for c in frame.columns if c not in meta_cols]


def fit_scaler(train: np.ndarray, normalisation: str = "global"):
    """Fit a scaler on the TRAINING FOLD ONLY.

    Args:
        train: Training-fold feature matrix.
        normalisation: ``global`` fits one `StandardScaler`; ``none``
            returns None.

    Returns:
        A fitted scaler, or None.

    Raises:
        ValueError: For ``per_speaker``, which needs speaker labels and is a
            Phase 4 experiment rather than a default.
    """
    from sklearn.preprocessing import StandardScaler

    if normalisation == "none":
        return None
    if normalisation == "global":
        return StandardScaler().fit(train)
    raise ValueError(
        "per-speaker normalisation is a Phase 4 experiment; use "
        "fit_scaler(..., 'global') or handle speakers explicitly"
    )


def apply_scaler(scaler, matrix: np.ndarray) -> np.ndarray:
    """Apply a scaler fitted on the training fold. Returns input if None."""
    return matrix if scaler is None else scaler.transform(matrix)


if __name__ == "__main__":  # pragma: no cover
    cfg = FeatureConfig()
    frame = extract(cfg)
    cols = feature_columns(frame)
    print(f"{cfg.name}  hash={cfg.hash}")
    print(f"{len(frame)} clips x {len(cols)} features -> {cfg.cache_path.name}")
