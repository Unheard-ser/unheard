"""Exploratory plots.

Two rules hold throughout this module.

**Every function returns a `Figure` and never calls ``plt.show``.** The caller
saves it. That keeps the functions testable -- a test can assert on axes and
titles without a display -- and keeps figure paths out of the plotting code.

**Everything that touches signal is restricted to training folds.** Phase 3
was deliberately run before Phase 2 so this is possible (see PLAN.md). Looking
at held-out audio before the experiment is designed leaks the test set through
the analyst: every later decision about which features to try and which
confusions to chase would be informed by it, unquantifiably and
irreversibly. `training_subset` is the guard, and each function's docstring
states which population it is legitimate to call it on.

The exceptions are corpus-level *counts* -- how many clips exist per class is a
property of the dataset, not of the test signal. Those functions say so in
their docstring and annotate the figure itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display on CI or in tests
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ser import SEED  # noqa: E402
from ser.metadata import EMOTIONS  # noqa: E402
from ser.preprocess import load_audio  # noqa: E402
from ser.splits import get_split  # noqa: E402

#: Fixed emotion order, so every figure in the deck reads the same way.
EMOTION_ORDER: tuple[str, ...] = tuple(EMOTIONS.values())

#: Arousal grouping, used to test whether confusions cluster by energy.
HIGH_AROUSAL: frozenset[str] = frozenset(
    {"angry", "happy", "fearful", "surprised"}
)

FIGSIZE = (10, 6)


def training_subset(
    meta: pd.DataFrame,
    protocol: str = "speaker_independent",
    fold: int = 0,
) -> pd.DataFrame:
    """Restrict a metadata frame to one fold's TRAINING clips.

    The guard that makes the rest of this module legitimate. Call it before
    any analysis that looks at audio.

    Args:
        meta: Full metadata frame from `ser.metadata.load`.
        protocol: Split protocol. Defaults to the primary one.
        fold: Which fold's training side to take.

    Returns:
        The subset of ``meta`` in that fold's training split.
    """
    train_files, _ = get_split(protocol, fold)
    return meta[meta["filename"].isin(train_files)].reset_index(drop=True)


def _finish(fig: Figure, title: str, note: str | None = None) -> Figure:
    """Apply the shared title/annotation treatment and tighten layout."""
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


# --- corpus-level: legitimate on all 1,440 --------------------------------


def plot_class_balance(meta: pd.DataFrame) -> Figure:
    """Clips per emotion, split by intensity.

    **Uses all 1,440 clips deliberately.** How many clips exist per class is a
    property of the corpus, not of the test signal, so this is not analyst
    leakage. The figure says so.

    Args:
        meta: Full metadata frame.

    Returns:
        The figure; the caller saves it.
    """
    counts = (
        meta.groupby(["emotion_label", "intensity_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(EMOTION_ORDER)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("")
    ax.set_ylabel("clips")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="intensity")

    ax.annotate(
        "neutral has half the clips:\nRAVDESS never recorded\nstrong-intensity neutral",
        xy=(list(EMOTION_ORDER).index("neutral"), 96),
        xytext=(0.42, 0.72),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#444"},
        bbox={"boxstyle": "round,pad=0.4", "fc": "#FFF6E5", "ec": "#DD8452"},
    )
    return _finish(
        fig,
        "Class balance (all 1,440 clips)",
        "Corpus-level counts, not signal — safe to compute on the full set.",
    )


# --- signal-level: TRAINING FOLDS ONLY ------------------------------------


def plot_waveform_and_spectrogram_grid(
    train: pd.DataFrame, sr: int = 16_000
) -> Figure:
    """One representative clip per emotion: waveform above, mel-spectrogram below.

    **Training folds only** — pass the output of `training_subset`.

    Args:
        train: Training-fold metadata.
        sr: Sample rate to load at.

    Returns:
        A 2 x 8 grid figure.
    """
    import librosa
    import librosa.display

    fig, axes = plt.subplots(2, len(EMOTION_ORDER), figsize=(22, 6))
    for column, emotion in enumerate(EMOTION_ORDER):
        candidates = train[train["emotion_label"] == emotion]
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        signal, rate = load_audio(row["filepath"], sr=sr)
        signal, _ = librosa.effects.trim(signal, top_db=30)

        axes[0, column].plot(
            np.arange(len(signal)) / rate, signal, linewidth=0.4, color="#4C72B0"
        )
        axes[0, column].set_title(emotion, fontsize=10, fontweight="bold")
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

        mel = librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=64)
        librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max),
            sr=rate, ax=axes[1, column], cmap="magma",
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

    axes[0, 0].set_ylabel("waveform", fontsize=9)
    axes[1, 0].set_ylabel("mel spectrogram", fontsize=9)
    return _finish(
        fig,
        "One clip per emotion — waveform and mel-spectrogram",
        "Training folds only. Silence trimmed. Amplitude scales are shared.",
    )


def plot_distribution_by(
    train: pd.DataFrame,
    column: str,
    by: str = "emotion_label",
    ylabel: str | None = None,
) -> Figure:
    """Box plot of one measured quantity, grouped by a metadata column.

    **Training folds only.**

    Args:
        train: Training-fold frame that already carries ``column``.
        column: The measured quantity, e.g. ``f0_mean`` or ``rms``.
        by: Grouping column.
        ylabel: Axis label; defaults to ``column``.

    Returns:
        The figure.
    """
    order = (
        [e for e in EMOTION_ORDER if e in set(train[by])]
        if by == "emotion_label"
        else sorted(train[by].dropna().unique())
    )
    data = [train.loc[train[by] == level, column].dropna().to_numpy() for level in order]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    parts = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, level in zip(parts["boxes"], order):
        high = by == "emotion_label" and level in HIGH_AROUSAL
        patch.set_facecolor("#DD8452" if high else "#4C72B0")
        patch.set_alpha(0.75)
    ax.set_ylabel(ylabel or column)
    ax.tick_params(axis="x", rotation=30)
    if by == "emotion_label":
        ax.plot([], [], color="#DD8452", linewidth=8, label="high arousal")
        ax.plot([], [], color="#4C72B0", linewidth=8, label="low arousal")
        ax.legend(fontsize=8)
    return _finish(
        fig,
        f"{ylabel or column} by {by.replace('_label', '')}",
        "Training folds only.",
    )


def plot_projection(
    features: np.ndarray,
    labels: pd.Series,
    method: str = "pca",
    title: str = "",
    seed: int = SEED,
) -> Figure:
    """2-D projection of the feature space, coloured by ``labels``.

    Run this twice -- once coloured by emotion, once by actor. If the actor
    plot shows tighter clusters than the emotion plot, the features encode
    identity more strongly than they encode emotion. That is the leakage
    story, made visible.

    **Training folds only.**

    Args:
        features: Feature matrix.
        labels: One label per row; drives the colouring.
        method: ``pca`` or ``umap``.
        title: Figure title.
        seed: Fixed for reproducibility.

    Returns:
        The figure.

    Raises:
        ValueError: On an unknown ``method``.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(scaled)
        axis_labels = [
            f"PC{i + 1} ({reducer.explained_variance_ratio_[i]:.1%})" for i in (0, 1)
        ]
    elif method == "umap":
        import umap

        coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(scaled)
        axis_labels = ["UMAP 1", "UMAP 2"]
    else:
        raise ValueError(f"method must be 'pca' or 'umap', got {method!r}")

    levels = (
        [e for e in EMOTION_ORDER if e in set(labels)]
        if set(labels) <= set(EMOTION_ORDER)
        else sorted(pd.unique(labels))
    )
    colours = plt.cm.tab20(np.linspace(0, 1, len(levels)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for level, colour in zip(levels, colours):
        mask = (labels == level).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.65, color=colour, label=str(level), linewidths=0,
        )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(fontsize=7, ncol=2, markerscale=1.6)
    return _finish(fig, title or f"{method.upper()} projection", "Training folds only.")


def save(fig: Figure, path: str | Path, dpi: int = 130) -> Path:
    """Save a figure and close it, so a long run does not leak memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def clip_descriptors(
    train: pd.DataFrame, sr: int = 16_000, top_db: int = 30
) -> pd.DataFrame:
    """Per-clip prosodic descriptors for the EDA plots.

    Measurement rather than plotting, but it lives here because it exists
    only to feed these figures -- the modelling path uses `ser.features`.

    **Call on training folds only.** It opens audio.

    Args:
        train: Training-fold metadata frame.
        sr: Sample rate to load at.
        top_db: Silence threshold for trimming.

    Returns:
        ``train`` with ``duration``, ``speech_duration``, ``silence_fraction``,
        ``rms``, ``f0_mean``, ``f0_std`` and ``voiced_fraction`` joined on.
        F0 is ``nan`` where no voiced frame was detected.
    """
    import librosa

    rows = []
    for path in train["filepath"]:
        signal, rate = load_audio(path, sr=sr)
        duration = len(signal) / rate
        speech, _ = librosa.effects.trim(signal, top_db=top_db)
        speech_duration = len(speech) / rate

        f0, voiced, _ = librosa.pyin(
            speech,
            fmin=float(librosa.note_to_hz("C2")),   # ~65 Hz, below a low male voice
            fmax=float(librosa.note_to_hz("C7")),   # ~2093 Hz, above a high female voice
            sr=rate,
        )
        voiced_f0 = f0[~np.isnan(f0)]

        rows.append(
            {
                "duration": duration,
                "speech_duration": speech_duration,
                "silence_fraction": 1.0 - speech_duration / duration,
                "rms": float(np.sqrt(np.mean(speech**2))),
                "f0_mean": float(voiced_f0.mean()) if voiced_f0.size else np.nan,
                "f0_std": float(voiced_f0.std()) if voiced_f0.size else np.nan,
                "voiced_fraction": float(np.mean(voiced)) if voiced.size else 0.0,
            }
        )
    return pd.concat(
        [train.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )


def silhouette_by_label(
    features: np.ndarray, labels: pd.Series, seed: int = SEED
) -> float:
    """How tightly the points cluster by ``labels``.

    Silhouette runs from -1 to +1. Positive means points sit nearer their own
    group than another's; around zero or below means the label explains no
    structure in this space.

    Args:
        features: Feature matrix.
        labels: One label per row.
        seed: Fixed, since the score is sampled.

    Returns:
        The mean silhouette coefficient.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    return float(
        silhouette_score(
            scaled, labels.to_numpy(), sample_size=len(scaled), random_state=seed
        )
    )


def plot_silhouette_comparison(
    raw: np.ndarray, normalised: np.ndarray, train: pd.DataFrame
) -> Figure:
    """Cluster tightness by actor vs by emotion, before and after normalisation.

    The leakage story as a number rather than an eyeball judgement. If the
    raw features cluster more tightly by actor than by emotion, they encode
    identity more strongly than they encode what we are trying to predict.

    **Training folds only.**

    Args:
        raw: Feature matrix without per-speaker normalisation.
        normalised: The same features, per-speaker z-scored.
        train: Training-fold metadata supplying ``actor`` and
            ``emotion_label``.

    Returns:
        A grouped bar chart.
    """
    scores = {
        space: {
            "actor": silhouette_by_label(matrix, train["actor"].astype(str)),
            "emotion": silhouette_by_label(matrix, train["emotion_label"]),
        }
        for space, matrix in (("raw", raw), ("per-speaker\nnormalised", normalised))
    }

    spaces = list(scores)
    positions = np.arange(len(spaces))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for offset, key, colour in ((-width / 2, "actor", "#C44E52"),
                                (+width / 2, "emotion", "#4C72B0")):
        values = [scores[s][key] for s in spaces]
        bars = ax.bar(positions + offset, values, width, label=f"by {key}", color=colour)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.001 if value >= 0 else -0.003),
                f"{value:+.4f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=9,
            )

    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(positions, spaces)
    ax.set_ylabel("silhouette score")
    ax.legend()
    return _finish(
        fig,
        "Does the feature space encode the speaker, or the emotion?",
        "Training folds only. Higher = tighter clusters. Above zero for actor "
        "means identity is encoded.",
    )
