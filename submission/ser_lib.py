"""Speech emotion recognition on RAVDESS — shared library.

Every function the four notebooks call lives here; the notebooks orchestrate and
report. Grouped in the order the pipeline runs:

    1. Metadata        parse the filename schema into a clip inventory
    2. Audio           the single loading path (mono, 16 kHz)
    3. Splits          the frozen speaker-independent folds
    4. Features        handcrafted MFCC-family descriptors
    5. Embeddings      frozen wav2vec2, pooled and frame-level
    6. Evaluation      the one scoring function; everything is logged
    7. Statistics      speaker-clustered bootstrap intervals
    8. Models          classical estimators
    9. Corpora         the four-dataset index
   10. Multi-corpus    merged folds and the four training arms

Two conventions hold throughout and are what make separate experiments
comparable:

    Splits come from splits/folds.csv and are never regenerated.
    Scoring goes through evaluate_pooled, which appends to results/results.csv.

Random seeds are fixed in set_seeds(). Anything fitted on data — scalers, PCA,
class weights — is fitted on the training fold only. The single documented
exception is per_speaker_normalise; see its docstring.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import io
import json
import librosa
import numpy as np
import os
import pandas as pd
import random
import re
import soundfile as sf
import subprocess
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Any, Callable
from typing import Any, Iterable, Sequence
from typing import Callable, Sequence


# ==========================================================================
#   INIT  
# ==========================================================================
#: The one seed. Import from here; do not redeclare.
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source we can reach (CLAUDE.md rule 6).

    Passing ``random_state=`` to an estimator only covers that estimator.
    This covers the global generators that otherwise make a rerun
    irreproducible -- numpy's legacy global RNG, the stdlib ``random``
    module, and torch once Phase 6 introduces it.

    Args:
        seed: The value to seed with. Defaults to `SEED`.

    Note:
        ``PYTHONHASHSEED`` is set here for completeness, but CPython reads it
        at interpreter startup. To make string hashing deterministic it must
        be exported *before* Python launches::

            PYTHONHASHSEED=42 python -m pytest

        Nothing we currently do depends on string hash order, so this is
        belt-and-braces rather than load-bearing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    # ImportError covers torch simply not being installed -- it lives in
    # requirements-deep.txt, so a classical-only environment is expected and
    # fine. OSError covers torch being *present but unloadable*: on Windows,
    # Smart App Control blocks its unsigned DLLs with WinError 4551, which is
    # an OSError, not an ImportError. Catching only the latter turned a seeding
    # call into a hard crash for every driver in the project, including the
    # ones that never touch torch. See docs/environment.md.
    try:  # pragma: no cover - depends on the local environment
        import torch
    except (ImportError, OSError):
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================================
# METADATA
# ==========================================================================
#: Root holding the per-actor directories. Note the doubled ``data`` segment.
DATA_ROOT: Path = Path(__file__).resolve().parent.parent / "data" / "data"

#: Constant fields stripped from these filenames: audio-only, speech.
MODALITY: str = "03"
VOCAL_CHANNEL: str = "01"

#: Ordered field names of the five-field filename schema.
FIELDS: tuple[str, ...] = (
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor_code",
)

EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITIES: dict[str, str] = {"01": "normal", "02": "strong"}

#: Statement 01 is "Kids are talking by the door", 02 is "Dogs are sitting by
#: the door". Held as short keys because they double as plot labels.
STATEMENTS: dict[str, str] = {"01": "kids", "02": "dogs"}

REPETITIONS: dict[str, str] = {"01": "1st", "02": "2nd"}

#: Actor directories are named ``Person01`` .. ``Person24``.
_ACTOR_DIR_RE = re.compile(r"^Person(\d{2})$")

EXPECTED_ROWS = 1440
EXPECTED_ACTORS = 24
CLIPS_PER_ACTOR = 60
NEUTRAL_CLIPS = 96
OTHER_EMOTION_CLIPS = 192


def parse_filename(path: str | Path) -> dict[str, object]:
    """Parse one RAVDESS filename into its fields.

    Strict by design: a filename that is not exactly five known-valued fields
    raises rather than being coerced. A canonical seven-field RAVDESS file
    dropped into this tree is therefore a loud failure, not a silent
    mis-parse that would shift every field by two positions.

    Args:
        path: Path to a ``.wav`` file. Only the stem and the parent directory
            name are read; the file is never opened and need not exist.

    Returns:
        Field codes, their human-readable labels, ``actor`` as an int,
        ``gender``, and the originating ``filename`` / ``filepath``.

    Raises:
        ValueError: If the stem does not split into exactly five fields, if
            any field holds an unknown code, or if field 5 disagrees with the
            ``PersonNN`` directory the file sits in.
    """
    path = Path(path)
    parts = path.stem.split("-")
    if len(parts) != len(FIELDS):
        raise ValueError(
            f"{path.name!r}: expected {len(FIELDS)} hyphen-separated fields "
            f"({'-'.join(FIELDS)}), got {len(parts)}. Canonical seven-field "
            f"RAVDESS names are not accepted here -- see module docstring."
        )

    emotion, intensity, statement, repetition, actor_code = parts

    for value, valid, name in (
        (emotion, EMOTIONS, "emotion"),
        (intensity, INTENSITIES, "intensity"),
        (statement, STATEMENTS, "statement"),
        (repetition, REPETITIONS, "repetition"),
    ):
        if value not in valid:
            raise ValueError(
                f"{path.name!r}: unknown {name} code {value!r}; "
                f"expected one of {sorted(valid)}"
            )

    if not re.fullmatch(r"\d{2}", actor_code):
        raise ValueError(f"{path.name!r}: actor code {actor_code!r} is not two digits")
    actor = int(actor_code)
    if not 1 <= actor <= EXPECTED_ACTORS:
        raise ValueError(f"{path.name!r}: actor {actor} outside 1..{EXPECTED_ACTORS}")

    dir_match = _ACTOR_DIR_RE.match(path.parent.name)
    if dir_match is not None and int(dir_match.group(1)) != actor:
        raise ValueError(
            f"{path.name!r}: actor field {actor} disagrees with directory "
            f"{path.parent.name!r}"
        )

    return {
        "filename": path.name,
        "filepath": str(path),
        "modality": MODALITY,
        "vocal_channel": VOCAL_CHANNEL,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "emotion_label": EMOTIONS[emotion],
        "intensity_label": INTENSITIES[intensity],
        "statement_label": STATEMENTS[statement],
        "repetition_label": REPETITIONS[repetition],
        # RAVDESS convention: odd-numbered actors are male, even are female.
        "gender": "male" if actor % 2 else "female",
    }


def load(root: Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Build the clip inventory for the whole dataset.

    Rows are sorted by ``(actor, emotion, intensity, statement, repetition)``
    and the index reset, so row order is identical on every machine
    regardless of filesystem glob order. Downstream code that indexes by
    position depends on this.

    Args:
        root: Directory holding the ``PersonNN`` folders. Defaults to
            `DATA_ROOT`.
        validate: Run the dataset invariants and raise on violation. Pass
            ``False`` only to inspect a deliberately incomplete tree.

    Returns:
        One row per ``.wav`` file, with the columns listed in
        `parse_filename`.

    Raises:
        FileNotFoundError: If ``root`` does not exist or holds no ``.wav``
            files.
        ValueError: From `parse_filename` on a malformed name, or from
            `validate_frame` on a dataset-level invariant breach.
    """
    root = Path(root) if root is not None else DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Audio root {root} not found. Audio is gitignored -- place the "
            f"RAVDESS clips in data/data/PersonNN/ before running."
        )

    paths = sorted(root.glob("Person*/*.wav"))
    if not paths:
        raise FileNotFoundError(f"No .wav files under {root}/Person*/")

    df = pd.DataFrame([parse_filename(p) for p in paths])
    df = df.sort_values(
        ["actor", "emotion", "intensity", "statement", "repetition"],
        kind="mergesort",
    ).reset_index(drop=True)

    if validate:
        validate_frame(df)
    return df


def validate_frame(df: pd.DataFrame) -> None:
    """Assert the documented RAVDESS invariants hold on ``df``.

    These are the facts the whole project rests on. They are checked, not
    assumed, because a partial download or a stray file would otherwise
    surface much later as an inexplicable model result. Every failure is
    collected before raising, so one run reports the full picture.

    Args:
        df: A frame in the shape `load` returns.

    Raises:
        ValueError: Listing every invariant that failed, with expected and
            observed values.
    """
    problems: list[str] = []

    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} clips, found {len(df)}")

    if df["filename"].duplicated().any():
        dupes = sorted(df.loc[df["filename"].duplicated(), "filename"].unique())
        problems.append(f"duplicate filenames: {dupes[:5]}")

    actors = sorted(df["actor"].unique())
    if actors != list(range(1, EXPECTED_ACTORS + 1)):
        problems.append(f"expected actors 1..{EXPECTED_ACTORS}, found {actors}")

    per_actor = df["actor"].value_counts()
    bad_actors = per_actor[per_actor != CLIPS_PER_ACTOR]
    if not bad_actors.empty:
        problems.append(
            f"expected {CLIPS_PER_ACTOR} clips per actor; off: {bad_actors.to_dict()}"
        )

    counts = df["emotion_label"].value_counts()
    for label in EMOTIONS.values():
        expected = NEUTRAL_CLIPS if label == "neutral" else OTHER_EMOTION_CLIPS
        found = int(counts.get(label, 0))
        if found != expected:
            problems.append(f"emotion {label}: expected {expected}, found {found}")

    # Neutral was never recorded at strong intensity -- that is why it has
    # half the clips of every other emotion.
    strong_neutral = int(((df["emotion"] == "01") & (df["intensity"] == "02")).sum())
    if strong_neutral:
        problems.append(
            f"found {strong_neutral} strong-intensity neutral clips; RAVDESS has none"
        )

    genders = df.groupby("gender")["actor"].nunique().to_dict()
    if genders != {"female": 12, "male": 12}:
        problems.append(f"expected 12 actors per gender, found {genders}")

    expected_gender = df["actor"].map(lambda a: "male" if a % 2 else "female")
    parity_ok = df["gender"] == expected_gender
    if not parity_ok.all():
        problems.append(f"{int((~parity_ok).sum())} rows where gender != actor parity")

    if df.isna().any().any():
        null_cols = sorted(df.columns[df.isna().any()])
        problems.append(f"null values in columns {null_cols}")

    if problems:
        raise ValueError("Dataset invariants failed:\n  - " + "\n  - ".join(problems))


# ==========================================================================
# PREPROCESS
# ==========================================================================
#: Project-wide target rate. See docs/data_audit.md Decision 3.
TARGET_SR: int = 16_000

#: The rate the files are stored at.
NATIVE_SR: int = 48_000


def load_audio(
    path: str | Path,
    sr: int | None = TARGET_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load one clip as a 1-D float32 signal at a known sample rate.

    Args:
        path: Path to a ``.wav`` file.
        sr: Target sample rate. Defaults to `TARGET_SR` (16 kHz). Pass
            ``None`` to keep the file's native rate -- used by `ser.audit`,
            which must describe the true files rather than a resampled
            version of them.
        mono: Downmix to a single channel. Leave ``True``; the argument
            exists so the audit can assert the downmix is a no-op.

    Returns:
        ``(signal, sample_rate)``. The signal is 1-D float32 when
        ``mono=True``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No audio file at {path}")

    y, out_sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, int(out_sr)


def channel_report(path: str | Path) -> dict[str, object]:
    """Describe a file's channel layout without downmixing it.

    Reads the header for the stored channel count, then, for multi-channel
    files, measures how far apart the channels actually are. This is what
    justifies treating the five dual-channel files as duplicated mono rather
    than as genuine stereo.

    Args:
        path: Path to a ``.wav`` file.

    Returns:
        ``n_channels``, ``is_multichannel``, and ``max_channel_diff`` -- the
        largest absolute difference between any two channels at the same
        sample, or ``0.0`` for a mono file. A value of exactly ``0.0`` on a
        multi-channel file means the channels are bit-identical and the
        downmix is lossless.
    """
    path = Path(path)
    info = sf.info(str(path))
    n_channels = int(info.channels)

    max_diff = 0.0
    if n_channels > 1:
        data, _ = sf.read(str(path), always_2d=True)
        first = data[:, :1]
        max_diff = float(np.abs(data - first).max())

    return {
        "n_channels": n_channels,
        "is_multichannel": n_channels > 1,
        "max_channel_diff": max_diff,
    }


# ==========================================================================
# SPLITS
# ==========================================================================
load_metadata = load





N_SPLITS = 5

#: The frozen fold assignments. Committed; never regenerated.
FOLDS_PATH: Path = Path(__file__).resolve().parent.parent / "splits" / "folds.csv"

PROTOCOLS: tuple[str, ...] = (
    "speaker_independent",
    "random_stratified",
    "statement_holdout",
    "statement_holdout_si",
)

FOLDS_COLUMNS: tuple[str, ...] = ("protocol", "fold", "filename", "split")

_Split = tuple[np.ndarray, np.ndarray]


def speaker_independent(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[_Split]:
    """5-fold GroupKFold grouped by actor — the primary protocol.

    No actor appears on both sides of a fold, so a model cannot score by
    recognising a voice it has already heard.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds. Leave at 5; the committed folds use 5.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(df, df["emotion_label"], groups=df["actor"]))


def random_stratified(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[_Split]:
    """5-fold StratifiedKFold on emotion — the leakage comparison only.

    Clips from the same actor land on both sides of every fold. That is the
    defect this protocol exists to quantify, by comparison against
    `speaker_independent`. Never report this number alone.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED
    )
    return list(splitter.split(df, df["emotion_label"]))


def statement_holdout(df: pd.DataFrame) -> list[_Split]:
    """Single split: train on statement 01, test on statement 02.

    Tests lexical invariance — does prosody carry emotion across different
    spoken content? All 24 actors appear on both sides, so pair this with
    `statement_holdout_si` before drawing a conclusion.

    Args:
        df: Metadata frame from `ser.metadata.load`.

    Returns:
        A single ``(train_idx, test_idx)`` pair, in a list for interface
        consistency with the k-fold protocols.
    """
    positions = np.arange(len(df))
    statement = df["statement"].to_numpy()
    return [(positions[statement == "01"], positions[statement == "02"])]


def statement_holdout_si(
    df: pd.DataFrame, n_splits: int = N_SPLITS
) -> list[_Split]:
    """Statement holdout with the speakers held out too.

    Reuses the `speaker_independent` actor partition, then restricts train to
    statement 01 and test to statement 02. Both the actor and the sentence
    differ between the two sides, so a good score cannot be explained by
    speaker familiarity.

    Clips that are neither (statement 01, training actor) nor (statement 02,
    held-out actor) belong to neither side of that fold. This is the reason
    ``folds.csv`` records membership explicitly rather than as a
    clip-to-test-fold mapping.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    statement = df["statement"].to_numpy()
    out: list[_Split] = []
    for train_idx, test_idx in speaker_independent(df, n_splits=n_splits):
        out.append(
            (
                train_idx[statement[train_idx] == "01"],
                test_idx[statement[test_idx] == "02"],
            )
        )
    return out


_BUILDERS = {
    "speaker_independent": speaker_independent,
    "random_stratified": random_stratified,
    "statement_holdout": statement_holdout,
    "statement_holdout_si": statement_holdout_si,
}


def build_folds(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the long-format fold table for every protocol.

    One row per (protocol, fold, clip, side). Long format rather than a
    compact clip-to-fold mapping because ``statement_holdout_si``'s training
    set is a subset, not the complement, of its test set — a clip can belong
    to neither side.

    Args:
        df: Metadata frame. Defaults to the full inventory.

    Returns:
        Columns ``protocol, fold, filename, split``, sorted deterministically
        so the CSV is byte-stable across machines.
    """
    if df is None:
        df = load_metadata()

    filenames = df["filename"].to_numpy()
    records: list[pd.DataFrame] = []

    for protocol in PROTOCOLS:
        for fold, (train_idx, test_idx) in enumerate(_BUILDERS[protocol](df)):
            for side, idx in (("train", train_idx), ("test", test_idx)):
                records.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "filename": filenames[idx],
                            "split": side,
                        }
                    )
                )

    folds = pd.concat(records, ignore_index=True)
    # Deterministic order: protocols in declared order, then fold, side, name.
    folds["_protocol_order"] = folds["protocol"].map(
        {name: i for i, name in enumerate(PROTOCOLS)}
    )
    folds = (
        folds.sort_values(
            ["_protocol_order", "fold", "split", "filename"], kind="mergesort"
        )
        .drop(columns="_protocol_order")
        .reset_index(drop=True)
    )
    return folds[list(FOLDS_COLUMNS)]


def write_folds(path: Path | None = None, force: bool = False) -> Path:
    """Write the fold assignments to CSV, once.

    Refuses to overwrite an existing file. This is CLAUDE.md rule 1 made
    mechanical: the folds are the contract that makes five people's results
    comparable, so regenerating them silently would invalidate every result
    already logged against them. Nothing in this codebase passes
    ``force=True``.

    Args:
        path: Destination. Defaults to `FOLDS_PATH`.
        force: Overwrite an existing file. Do not use.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    target = Path(path) if path is not None else FOLDS_PATH
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists and folds are frozen (CLAUDE.md rule 1). "
            f"Every experiment logged so far assumes these exact assignments. "
            f"If you genuinely believe a split is wrong, stop and ask the team."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    build_folds().to_csv(target, index=False, lineterminator="\n")
    return target


def load_folds(path: Path | None = None) -> pd.DataFrame:
    """Read the frozen fold assignments.

    Args:
        path: Source. Defaults to `FOLDS_PATH`.

    Returns:
        Columns ``protocol, fold, filename, split``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(path) if path is not None else FOLDS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"No fold assignments at {source}. Generate them once with "
            f"ser.splits.write_folds(), then commit the file."
        )
    return pd.read_csv(source, dtype={"protocol": str, "filename": str, "split": str})


def get_split(
    protocol: str,
    fold: int,
    folds: pd.DataFrame | None = None,
) -> tuple[list[str], list[str]]:
    """Return the train and test filenames for one (protocol, fold).

    Filenames rather than positional indices: an index list silently means
    something different if applied to a differently-ordered DataFrame, and
    that failure mode would be invisible.

    Args:
        protocol: One of `PROTOCOLS`.
        fold: Fold number, 0-based.
        folds: Fold table. Defaults to reading the frozen CSV.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If the protocol is unknown or the fold does not exist.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    table = load_folds() if folds is None else folds
    subset = table[(table["protocol"] == protocol) & (table["fold"] == fold)]
    if subset.empty:
        available = sorted(table.loc[table["protocol"] == protocol, "fold"].unique())
        raise ValueError(
            f"No fold {fold} for protocol {protocol!r}; available folds: {available}"
        )

    train = subset.loc[subset["split"] == "train", "filename"].tolist()
    test = subset.loc[subset["split"] == "test", "filename"].tolist()
    return train, test


def summarise_folds(
    folds: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-fold composition of every protocol's test side.

    This is the table to eyeball before freezing, and the one reproduced in
    ``docs/splits.md``.

    Args:
        folds: Fold table. Defaults to the frozen CSV.
        meta: Metadata frame. Defaults to the full inventory.

    Returns:
        One row per (protocol, fold) with train/test sizes, actor counts,
        actor overlap, gender split and neutral-class count.
    """
    table = load_folds() if folds is None else folds
    df = load_metadata() if meta is None else meta
    lookup = df.set_index("filename")

    rows = []
    for (protocol, fold), group in table.groupby(["protocol", "fold"], sort=False):
        train_names = group.loc[group["split"] == "train", "filename"]
        test_names = group.loc[group["split"] == "test", "filename"]
        train, test = lookup.loc[train_names], lookup.loc[test_names]
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_actors": train["actor"].nunique(),
                "test_actors": test["actor"].nunique(),
                "actor_overlap": len(
                    set(train["actor"]) & set(test["actor"])
                ),
                "test_pct_male": round(100 * (test["gender"] == "male").mean(), 1),
                "test_neutral": int((test["emotion_label"] == "neutral").sum()),
                "test_emotions": test["emotion_label"].nunique(),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# FEATURES
# ==========================================================================
load_metadata = load





#: Gitignored, regenerable.
CACHE_DIR: Path = Path(__file__).resolve().parent / "features"

AGGREGATIONS: tuple[str, ...] = ("mean", "mean_std", "mean_std_min_max", "percentiles")


#: Delta context we hold constant when the hop changes. ``librosa`` measures
#: delta width in FRAMES, so a fixed width silently means a different amount
#: of time at a different hop. See `FeatureConfig.resolved_delta_width`.
DEFAULT_DELTA_CONTEXT_MS: float = 288.0


@dataclass(frozen=True)
class FeatureConfig:
    """One point in the feature-design space.

    Every parameter that affects the numbers is named here. None is left to a
    library default: `librosa` is a *music* library and several of its
    defaults are wrong for speech, but the real problem with an unstated
    default is that nobody can see it, question it, or sweep it.

    Attributes:
        n_mfcc: Number of MFCC coefficients.
        deltas: Append first-order deltas.
        delta_deltas: Append second-order deltas.
        aggregation: How frame-level features collapse to a fixed vector.
        trim: Remove leading/trailing silence before extraction.
        top_db: Silence threshold for trimming, in dB below peak.
        sr: Sample rate to load at.
        n_fft: Analysis window in samples. The default 2048 is **128 ms** at
            16 kHz -- roughly 5x the 20-40 ms conventional for speech. It is
            librosa's default, inherited rather than chosen, and Phase 4
            tests whether it costs us.
        hop_length: Step between frames in samples. 512 is **32 ms** at
            16 kHz, against a ~10 ms speech convention.
        delta_width: Frames spanned when computing deltas. ``None`` derives
            it from ``hop_length`` so the *time* context stays
            `DEFAULT_DELTA_CONTEXT_MS` regardless of framing -- otherwise
            changing the hop would silently change two things at once and no
            ablation result could be attributed.
        n_mels: Mel filterbank bands. 128 is dense for 16 kHz speech, where
            40-80 is typical.
        fmin: Lowest mel frequency. 0 Hz includes rumble below the speech
            range.
        fmax: Highest mel frequency; ``None`` means ``sr / 2``.
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
    # --- framing: librosa defaults, made explicit. Changing these changes
    # --- the numbers, so they are swept in Phase 4 rather than edited here.
    n_fft: int = 2048
    hop_length: int = 512
    delta_width: int | None = None
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    extras: tuple[str, ...] = field(default_factory=tuple)
    normalisation: str = "none"

    def __post_init__(self) -> None:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}"
            )
        if self.normalisation not in ("none", "global", "per_speaker"):
            raise ValueError(f"unknown normalisation {self.normalisation!r}")
        if self.delta_width is not None and (
            self.delta_width < 3 or self.delta_width % 2 == 0
        ):
            raise ValueError(
                f"delta_width must be an odd integer >= 3, got {self.delta_width}"
            )

    # --- human-facing views on the framing -------------------------------

    @property
    def window_ms(self) -> float:
        """Analysis window length in milliseconds."""
        return 1000.0 * self.n_fft / self.sr

    @property
    def hop_ms(self) -> float:
        """Step between consecutive frames, in milliseconds."""
        return 1000.0 * self.hop_length / self.sr

    @property
    def resolved_delta_width(self) -> int:
        """Delta width in frames, derived from the hop when not set.

        Holds the delta's *time* context near
        `DEFAULT_DELTA_CONTEXT_MS`: width 9 at a 32 ms hop, width 29 at a
        10 ms hop. Always odd and at least 3, as librosa requires.
        """
        if self.delta_width is not None:
            return self.delta_width
        width = round(DEFAULT_DELTA_CONTEXT_MS / self.hop_ms)
        width = max(3, width)
        return width if width % 2 else width + 1

    @property
    def delta_context_ms(self) -> float:
        """Time span the delta computation actually sees."""
        return self.resolved_delta_width * self.hop_ms

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
        """Short human-readable identifier, for ``results.csv``.

        Framing, mel and normalisation settings appear **only when they
        differ from the defaults**. That keeps the baseline name
        ``mfcc40-d-dd-mean_std-trim`` byte-stable -- it is already written
        into results.csv rows, and changing it would break comparability
        with every result logged so far.
        """
        default = _DEFAULTS
        parts = [f"mfcc{self.n_mfcc}"]
        if self.deltas:
            parts.append("d")
        if self.delta_deltas:
            parts.append("dd")
        parts.append(self.aggregation)
        parts.append("trim" if self.trim else "notrim")
        if self.extras:
            parts.append("+".join(sorted(self.extras)))
        if (self.n_fft, self.hop_length) != (default["n_fft"], default["hop_length"]):
            parts.append(f"win{self.window_ms:g}ms-hop{self.hop_ms:g}ms")
        if self.delta_width is not None:
            parts.append(f"dw{self.delta_width}")
        if self.n_mels != default["n_mels"]:
            parts.append(f"mel{self.n_mels}")
        if self.fmin != default["fmin"]:
            parts.append(f"fmin{self.fmin:g}")
        if self.fmax != default["fmax"]:
            parts.append(f"fmax{self.fmax:g}")
        if self.normalisation != "none":
            parts.append(self.normalisation)
        return "-".join(parts)

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"features_{self.hash}.parquet"

    def describe(self) -> str:
        """Multi-line summary of every parameter in effect, for notebooks."""
        return "\n".join(
            [
                f"name            {self.name}",
                f"hash            {self.hash}",
                f"sample rate     {self.sr} Hz",
                f"window          {self.n_fft} samples = {self.window_ms:.1f} ms",
                f"hop             {self.hop_length} samples = {self.hop_ms:.1f} ms",
                f"overlap         {100 * (1 - self.hop_length / self.n_fft):.0f}%",
                f"frame rate      {self.sr / self.hop_length:.1f} frames/s",
                f"mel bands       {self.n_mels}  ({self.fmin:g} Hz to "
                f"{self.fmax if self.fmax is not None else self.sr // 2:g} Hz)",
                f"MFCCs           {self.n_mfcc}"
                f"{' +delta' if self.deltas else ''}"
                f"{' +delta2' if self.delta_deltas else ''}",
                f"delta width     {self.resolved_delta_width} frames = "
                f"{self.delta_context_ms:.0f} ms of context",
                f"aggregation     {self.aggregation}",
                f"trim silence    {self.trim} (top_db={self.top_db})",
                f"normalisation   {self.normalisation}",
            ]
        )


#: Default field values, used by `FeatureConfig.name` to decide what to omit.
_DEFAULTS: dict[str, object] = {
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0.0,
    "fmax": None,
}


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

    # Every framing parameter is passed explicitly. Relying on librosa's
    # defaults here is what made the window length invisible in the first
    # place -- and its defaults are tuned for music, not speech.
    spec_kw = {
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
    }
    mel_kw = {
        **spec_kw,
        "n_mels": config.n_mels,
        "fmin": config.fmin,
        "fmax": config.fmax,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, **mel_kw)
    blocks.append(mfcc)
    names += [f"mfcc{i:02d}" for i in range(config.n_mfcc)]

    width = config.resolved_delta_width
    # librosa requires width <= number of frames; short clips can fall under.
    width = min(width, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))
    width = max(3, width if width % 2 else width - 1)

    if config.deltas:
        blocks.append(librosa.feature.delta(mfcc, width=width))
        names += [f"d_mfcc{i:02d}" for i in range(config.n_mfcc)]
    if config.delta_deltas:
        blocks.append(librosa.feature.delta(mfcc, order=2, width=width))
        names += [f"dd_mfcc{i:02d}" for i in range(config.n_mfcc)]

    extras = set(config.extras)
    if "chroma" in extras:
        block = librosa.feature.chroma_stft(y=y, sr=sr, **spec_kw)
        blocks.append(block)
        names += [f"chroma{i:02d}" for i in range(block.shape[0])]
    if "spectral_contrast" in extras:
        block = librosa.feature.spectral_contrast(y=y, sr=sr, **spec_kw)
        blocks.append(block)
        names += [f"contrast{i:02d}" for i in range(block.shape[0])]
    if "spectral_centroid" in extras:
        blocks.append(librosa.feature.spectral_centroid(y=y, sr=sr, **spec_kw))
        names.append("centroid")
    if "rolloff" in extras:
        blocks.append(librosa.feature.spectral_rolloff(y=y, sr=sr, **spec_kw))
        names.append("rolloff")
    if "zcr" in extras:
        blocks.append(
            librosa.feature.zero_crossing_rate(
                y, frame_length=config.n_fft, hop_length=config.hop_length
            )
        )
        names.append("zcr")
    if "rms" in extras:
        blocks.append(
            librosa.feature.rms(
                y=y, frame_length=config.n_fft, hop_length=config.hop_length
            )
        )
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
        # Silence is judged over these same frames, so the trim resolution
        # tracks the analysis resolution instead of librosa's fixed default.
        y, _ = librosa.effects.trim(
            y,
            top_db=config.top_db,
            frame_length=config.n_fft,
            hop_length=config.hop_length,
        )
    matrix, base_names = _frame_features(y, sr, config)
    vector, suffixes = _aggregate(matrix, config.aggregation)
    columns = [f"{name}_{suffix}" for suffix in suffixes for name in base_names]
    return vector, columns


def extract_handcrafted(
    config: FeatureConfig | None = None,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract features for every clip, caching by config hash.

    Args:
        config: Feature configuration. Defaults to `FeatureConfig()`.
        df: Metadata frame. Defaults to the full inventory.
        use_cache: Read from and write to the parquet cache.
        cache_path: Override the cache file. Required when extracting over a
            clip set other than the RAVDESS inventory -- the merged corpus
            shares a `FeatureConfig`, and therefore a hash, with RAVDESS, so
            without a distinct path the two would evict each other on every
            call. The filename check below would prevent a *wrong* read, but
            the thrashing would silently cost hours.
        n_jobs: Worker processes for extraction. ``1`` keeps the original
            serial path exactly. Extraction is CPU-bound and per-clip
            independent, so this scales close to linearly -- which matters
            once the clip count goes from 1,440 to 12,162.

    Returns:
        Metadata columns plus one column per feature, in metadata row order.
    """
    config = config or FeatureConfig()
    if df is None:
        df = load_metadata()

    target = cache_path if cache_path is not None else config.cache_path

    if use_cache and target.is_file():
        cached = pd.read_parquet(target)
        if len(cached) == len(df) and (
            cached["filename"].to_numpy() == df["filename"].to_numpy()
        ).all():
            return cached

    paths = df["filepath"].tolist()
    if n_jobs == 1:
        results = [extract_one(path, config) for path in paths]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(extract_one)(path, config) for path in paths
        )

    vectors = [vector for vector, _ in results]
    columns = results[0][1] if results else []

    features = pd.DataFrame(np.vstack(vectors), columns=columns)
    out = pd.concat([df.reset_index(drop=True), features], axis=1)

    if use_cache:
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
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
        ValueError: For ``per_speaker``, which is not a fold-level fit at all
            -- use `per_speaker_normalise`.
    """
    from sklearn.preprocessing import StandardScaler

    if normalisation == "none":
        return None
    if normalisation == "global":
        return StandardScaler().fit(train)
    raise ValueError(
        "per-speaker normalisation is not a train-fold fit; call "
        "per_speaker_normalise(matrix, speakers) instead"
    )


def apply_scaler(scaler, matrix: np.ndarray) -> np.ndarray:
    """Apply a scaler fitted on the training fold. Returns input if None."""
    return matrix if scaler is None else scaler.transform(matrix)


def per_speaker_normalise(
    matrix: np.ndarray, speakers: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Z-score each speaker's features against that speaker's own statistics.

    Also known as cepstral mean and variance normalisation (CMVN). The intent
    is to strip out what is constant about a voice -- its timbre and the
    recording gain -- leaving what varies within it, which is the emotion.

    **A stated exception to CLAUDE.md rule 2.** That rule says per-speaker
    statistics are fit on the training fold only. Under
    ``speaker_independent`` that is not merely inconvenient, it is impossible:
    a held-out speaker has no training clips by construction. So a test
    speaker's statistics are computed from that speaker's own held-out clips.

    Why this is defensible: it consumes **no labels**, only audio, which is
    equally available at deployment time -- you would have a caller's other
    utterances before deciding anything about them. It is standard practice
    in speech.

    Why it is still worth flagging: it is transductive. The method sees the
    test audio, if not the test labels, so a strict reviewer may discount it.
    `docs/feature_findings.md` states this plainly rather than burying it.

    Args:
        matrix: Feature matrix, one row per clip.
        speakers: Speaker (actor) id per row. Same length as ``matrix``.
        eps: Floor on the divisor, so a speaker with a constant feature does
            not produce a division by zero.

    Returns:
        A new matrix, z-scored within each speaker.

    Raises:
        ValueError: If ``speakers`` does not align with ``matrix``.
    """
    if len(speakers) != len(matrix):
        raise ValueError(
            f"speakers has {len(speakers)} entries but matrix has {len(matrix)} rows"
        )

    out = np.asarray(matrix, dtype=float).copy()
    for speaker in np.unique(speakers):
        mask = speakers == speaker
        block = out[mask]
        out[mask] = (block - block.mean(axis=0)) / (block.std(axis=0) + eps)
    return out


# ==========================================================================
# EMBEDDINGS
# ==========================================================================
load_metadata = load





#: Shared with `ser.features` so every cached representation lives together.
CACHE_DIR: Path = Path(__file__).resolve().parent / "features"

#: Backbones this module has been exercised with. Any HF model exposing
#: ``hidden_states`` works; these are the ones with recorded results.
KNOWN_MODELS: dict[str, str] = {
    "facebook/wav2vec2-base": "wav2vec2-base, 12 layers, 768-d, 95M params",
    "microsoft/wavlm-base-plus": "WavLM-base+, 12 layers, 768-d -- usually "
    "stronger on emotion and speaker tasks",
    "facebook/hubert-base-ls960": "HuBERT-base, 12 layers, 768-d",
}

POOLINGS: tuple[str, ...] = ("mean", "mean_std", "std")


@dataclass(frozen=True)
class EmbeddingConfig:
    """One point in the embedding-design space.

    Every parameter that changes the stored numbers is named here, for the
    same reason `ser.features.FeatureConfig` does it: an unstated default is
    one nobody can see, question, or sweep.

    Attributes:
        model_id: HuggingFace model id. See `KNOWN_MODELS`.
        sr: Sample rate fed to the model. Asserted against the model's own
            feature extractor -- wav2vec2 is trained at 16 kHz and silently
            feeding it another rate produces plausible, wrong numbers.
        layer: Which hidden state to read. ``"last"``, an int index into
            ``hidden_states`` (0 is the CNN feature projection, 1..12 the
            transformer blocks), or ``"weighted"`` for a learned combination.
            A *selection*, not an extraction parameter -- excluded from the
            cache hash because one pass stores all layers.
        pooling: How frame-level states collapse to a fixed vector. Also a
            selection, also excluded from the hash.
        max_duration_s: Truncate clips longer than this. ``None`` keeps every
            clip whole, which is the default because the corpora's own
            durations are short and trimming would discard real speech.
        peak_norm: Divide each clip by its peak amplitude. Off by default:
            Phase 1 found absolute loudness *carries* emotion, so discarding
            it is a trade-off to be measured, not assumed. It exists because
            the four corpora were recorded at different gains.
        trim: Remove leading and trailing silence before encoding.
        top_db: Silence threshold for trimming, in dB below peak.
        normalisation: ``none`` | ``global`` | ``per_speaker``. Applied by the
            caller per fold, never here. Excluded from the hash.
        batch_size: Clips per forward pass. Affects speed only.
    """

    model_id: str = "facebook/wav2vec2-base"
    sr: int = TARGET_SR
    layer: int | str = "last"
    pooling: str = "mean"
    max_duration_s: float | None = None
    peak_norm: bool = False
    trim: bool = False
    top_db: int = 30
    normalisation: str = "none"
    batch_size: int = 8

    #: Parameters that describe a *selection* from the cache rather than the
    #: extraction itself. Excluded from `hash`, so every layer/pooling/
    #: normalisation choice reuses one cached forward pass.
    _SELECTION: tuple[str, ...] = field(
        default=("layer", "pooling", "normalisation", "batch_size"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if self.normalisation not in ("none", "global", "per_speaker"):
            raise ValueError(f"unknown normalisation {self.normalisation!r}")
        if isinstance(self.layer, str) and self.layer not in ("last", "weighted"):
            raise ValueError(
                f"layer must be an int, 'last' or 'weighted', got {self.layer!r}"
            )
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive or None")

    @property
    def hash(self) -> str:
        """Stable 12-char hash of everything that affects the stored tensor."""
        payload = {
            k: v for k, v in asdict(self).items() if k not in self._SELECTION
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        """Short human-readable identifier, for ``results.csv``."""
        parts = [self.model_id.split("/")[-1]]
        parts.append(f"L{self.layer}")
        parts.append(self.pooling)
        if self.trim:
            parts.append("trim")
        if self.peak_norm:
            parts.append("peaknorm")
        if self.max_duration_s is not None:
            parts.append(f"max{self.max_duration_s:g}s")
        if self.normalisation != "none":
            parts.append(self.normalisation)
        return "-".join(parts)

    @property
    def cache_path(self) -> Path:
        """The pooled per-layer tensor, ``(clips, layers, 2, dim)`` float32."""
        return CACHE_DIR / f"w2v_{self.hash}.npy"

    @property
    def order_path(self) -> Path:
        """Filenames in cache row order, so a cache can be checked not assumed."""
        return CACHE_DIR / f"w2v_{self.hash}_order.parquet"

    def describe(self) -> str:
        """Multi-line summary of every parameter in effect, for notebooks."""
        return "\n".join(
            [
                f"name            {self.name}",
                f"hash            {self.hash}",
                f"model           {self.model_id}",
                f"                {KNOWN_MODELS.get(self.model_id, 'not in KNOWN_MODELS')}",
                f"sample rate     {self.sr} Hz",
                f"layer           {self.layer}",
                f"pooling         {self.pooling}",
                f"max duration    {self.max_duration_s or 'whole clip'}",
                f"peak norm       {self.peak_norm}",
                f"trim silence    {self.trim} (top_db={self.top_db})",
                f"normalisation   {self.normalisation} (applied per fold by the caller)",
                f"cache           {self.cache_path.name}",
            ]
        )


def load_signal(path: str | Path, config: EmbeddingConfig) -> np.ndarray:
    """Load one clip exactly as the encoder will see it.

    Goes through `ser.preprocess.load_audio`, the project's single audio entry
    point, so the mono downmix and target rate hold here as everywhere else.

    Args:
        path: Path to a ``.wav``.
        config: Supplies rate, trimming and normalisation choices.

    Returns:
        A 1-D float32 signal.
    """
    signal, _ = load_audio(path, sr=config.sr)
    signal = np.asarray(signal, dtype=np.float32)

    if config.trim:
        import librosa

        signal, _ = librosa.effects.trim(signal, top_db=config.top_db)

    if config.peak_norm:
        peak = float(np.abs(signal).max()) if signal.size else 0.0
        if peak > 0:
            signal = signal / peak

    if config.max_duration_s is not None:
        limit = int(config.max_duration_s * config.sr)
        signal = signal[:limit]

    return signal


def extract_embeddings(
    config: EmbeddingConfig | None = None,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Encode every clip once, caching the pooled statistics of every layer.

    The expensive call. On CPU this is minutes per thousand clips; the cache
    exists so it happens once per ``(model, audio-preparation)`` pair and
    every later layer or pooling question is answered from disk.

    Args:
        config: Embedding configuration. Defaults to `EmbeddingConfig()`.
        df: Frame with ``filepath`` and ``filename`` columns, in the order the
            rows should be cached. Defaults to the RAVDESS inventory.
        use_cache: Read from and write to `EmbeddingConfig.cache_path`.
        progress: Show a progress bar.

    Returns:
        ``(tensor, df)`` where ``tensor`` has shape
        ``(len(df), n_layers, 2, dim)`` -- axis 2 is ``[mean, std]`` over the
        clip's unmasked frames -- and ``df`` is the frame the rows correspond
        to.

    Raises:
        ValueError: If the model's expected sample rate differs from
            ``config.sr``, which would otherwise silently produce wrong
            numbers.
    """
    config = config or EmbeddingConfig()
    if df is None:

        df = load_metadata()

    # The cache check comes before torch is imported, deliberately. Reading a
    # cached tensor is pure numpy, and requiring a 2 GB deep-learning stack to
    # hand back an array that is already on disk makes every downstream driver
    # fail on a machine that cannot load torch -- which is not hypothetical
    # here (see docs/environment.md on Windows Smart App Control).
    if use_cache and config.cache_path.is_file() and config.order_path.is_file():
        order = pd.read_parquet(config.order_path)
        if len(order) == len(df) and (
            order["filename"].to_numpy() == df["filename"].to_numpy()
        ).all():
            return np.load(config.cache_path, mmap_mode="r"), df

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(config.model_id)
    if int(extractor.sampling_rate) != int(config.sr):
        raise ValueError(
            f"{config.model_id} expects {extractor.sampling_rate} Hz but the "
            f"config asks for {config.sr} Hz. Feeding a model the wrong rate "
            f"produces plausible, wrong embeddings."
        )

    model = AutoModel.from_pretrained(config.model_id)
    model.eval()

    paths = df["filepath"].tolist()
    batches = range(0, len(paths), config.batch_size)
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=config.model_id.split("/")[-1], unit="batch")

    out: list[np.ndarray] = []
    for start in batches:
        chunk = paths[start : start + config.batch_size]
        signals = [load_signal(p, config) for p in chunk]
        inputs = extractor(
            signals,
            sampling_rate=config.sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True)

        # (layers, batch, frames, dim)
        states = torch.stack(result.hidden_states, dim=0)

        mask = _frame_mask(model, inputs, states.shape[2])
        # (1, batch, frames, 1) so it broadcasts across layers and dims
        weights = mask[None, :, :, None].to(states.dtype)
        counts = weights.sum(dim=2).clamp(min=1.0)

        mean = (states * weights).sum(dim=2) / counts
        variance = (
            ((states - mean.unsqueeze(2)) ** 2 * weights).sum(dim=2) / counts
        ).clamp(min=0.0)
        std = variance.sqrt()

        # -> (batch, layers, 2, dim)
        pooled = torch.stack([mean, std], dim=2).permute(1, 0, 2, 3).contiguous()
        out.append(pooled.to(torch.float32).numpy())

    tensor = np.concatenate(out, axis=0)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.cache_path, tensor)
        df[["filename"]].reset_index(drop=True).to_parquet(
            config.order_path, index=False
        )
    return tensor, df


def _frame_mask(model, inputs, n_frames: int):
    """Per-frame validity mask for a padded batch.

    The encoder's convolutional front end downsamples, so the sample-level
    attention mask the tokenizer produces does not line up with the frame
    axis of ``hidden_states``. Transformers exposes the length arithmetic;
    where it does not, every frame is treated as valid, which is exact for a
    batch of equal-length clips.
    """
    import torch

    attention = inputs.get("attention_mask")
    if attention is None:
        return torch.ones(inputs["input_values"].shape[0], n_frames)

    getter = getattr(model, "_get_feat_extract_output_lengths", None)
    if getter is None:
        return torch.ones(attention.shape[0], n_frames)

    lengths = getter(attention.sum(dim=-1)).to(torch.long)
    mask = torch.arange(n_frames)[None, :] < lengths[:, None]
    return mask.to(torch.float32)


def select(
    tensor: np.ndarray,
    config: EmbeddingConfig,
    layer_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Slice a cached tensor down to one feature matrix.

    Args:
        tensor: ``(clips, layers, 2, dim)`` from `extract`.
        config: Supplies ``layer`` and ``pooling``.
        layer_weights: Required when ``config.layer == "weighted"``; one
            weight per layer. They are applied to the already-pooled
            statistics.

            For the **mean** channel this is *exact*, because a weighted sum
            and a mean commute:
            ``sum_i w_i * mean(h_i) == mean(sum_i w_i * h_i)``. That identity
            is what lets the cache store per-layer means instead of per-layer
            frame sequences, and it is asserted in
            ``tests/test_embeddings.py``.

            For the **std** channel it is an *approximation*: the result is a
            weighted average of each layer's dispersion, not the dispersion of
            the weighted combination. Those differ whenever layers are
            correlated, which they are. The approximation is usable -- it is
            still a monotone measure of within-clip variability -- but a
            ``weighted`` + ``std``/``mean_std`` result must be reported as
            such, not as an exact layer-weighted statistic. Use
            ``pooling="mean"`` when exactness matters.

    Returns:
        ``(clips, dim)`` for mean or std pooling, ``(clips, 2 * dim)`` for
        ``mean_std``.

    Raises:
        ValueError: On a layer index out of range, or missing weights.
    """
    n_layers = tensor.shape[1]

    if config.layer == "weighted":
        if layer_weights is None:
            raise ValueError("layer='weighted' requires layer_weights")
        weights = np.asarray(layer_weights, dtype=np.float32)
        if weights.shape != (n_layers,):
            raise ValueError(
                f"layer_weights must have shape ({n_layers},), got {weights.shape}"
            )
        # Sums over the layer axis, leaving (clips, [mean,std], dim).
        block = np.tensordot(weights, np.asarray(tensor), axes=([0], [1]))
    else:
        index = n_layers - 1 if config.layer == "last" else int(config.layer)
        if not -n_layers <= index < n_layers:
            raise ValueError(
                f"layer {config.layer} out of range for {n_layers} cached layers"
            )
        block = np.asarray(tensor[:, index])

    mean, std = block[:, 0, :], block[:, 1, :]
    if config.pooling == "mean":
        return np.ascontiguousarray(mean)
    if config.pooling == "std":
        return np.ascontiguousarray(std)
    return np.ascontiguousarray(np.concatenate([mean, std], axis=1))


def extract_frames(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    layer: int = 6,
    progress: bool = True,
):
    """Yield **per-frame** hidden states, for models that attend over time.

    `extract` pools each clip to a mean and a standard deviation, which is what
    a fixed-length classifier needs and is useless to anything with attention:
    pooling is precisely the step attention is meant to replace. This is the
    escape hatch.

    Streaming rather than cached, deliberately. One layer of frame-level states
    for the merged corpus is ``12,162 x ~180 x 768 x 4 bytes`` -- about 6.7 GB,
    and every layer would be 87 GB. RAVDESS alone at one layer is ~820 MB, which
    `cache_frames` will materialise if you want random access; for a training
    loop, iterate this instead and keep nothing.

    Args:
        df: Frame with ``filepath``, in the order batches should arrive.
        config: Embedding configuration. ``config.layer`` is ignored -- pass
            ``layer`` explicitly, since the point here is one layer at a time.
        layer: Which hidden state to return. Defaults to 6, the Phase 13 winner.
        progress: Show a progress bar.

    Yields:
        ``(states, lengths, filenames)`` per batch, where ``states`` is a
        float32 array ``(batch, max_frames, dim)`` **zero-padded on the right**
        and ``lengths`` gives each clip's real frame count.

        **Use the lengths.** Clips differ in duration, so the padding is real
        and an attention layer that does not mask it will happily attend to
        silence that is not in the audio. Build a mask as
        ``torch.arange(max_frames)[None, :] < lengths[:, None]`` and add
        ``-inf`` to the scores where it is False.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    config = config or EmbeddingConfig()
    extractor = AutoFeatureExtractor.from_pretrained(config.model_id)
    if int(extractor.sampling_rate) != int(config.sr):
        raise ValueError(
            f"{config.model_id} expects {extractor.sampling_rate} Hz, config says {config.sr}"
        )
    model = AutoModel.from_pretrained(config.model_id)
    model.eval()

    paths = df["filepath"].tolist()
    names = df["filename"].tolist()
    batches = range(0, len(paths), config.batch_size)
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=f"frames L{layer}", unit="batch")

    for start in batches:
        chunk = paths[start : start + config.batch_size]
        signals = [load_signal(p, config) for p in chunk]
        inputs = extractor(
            signals, sampling_rate=config.sr, return_tensors="pt",
            padding=True, return_attention_mask=True,
        )
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True)

        states = result.hidden_states[layer]
        mask = _frame_mask(model, inputs, states.shape[1])
        yield (
            states.to(torch.float32).numpy(),
            mask.sum(dim=1).to(torch.int32).numpy(),
            names[start : start + config.batch_size],
        )


def cache_frames(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    layer: int = 6,
    path: Path | None = None,
    max_frames: int = 320,
):
    """Materialise one layer of frame-level states to a ``.npy`` for random access.

    Worth it for RAVDESS (~820 MB at layer 6) when a training loop needs to
    shuffle clips across epochs; not worth it for the merged corpus. Written as
    a memmap so a caller can slice a batch without loading the whole file.

    Args:
        df: Frame with ``filepath`` and ``filename``.
        config: Embedding configuration.
        layer: Which hidden state to store.
        path: Destination ``.npy``. Defaults to a hash-derived name beside the
            pooled caches.
        max_frames: Clips are padded or truncated to this many frames. 320 is
            ~6.4 s at wav2vec2's 50 frames/s, comfortably above RAVDESS's 5.3 s
            maximum; anything longer is truncated and reported.

    Returns:
        ``(array, lengths, filenames)`` where ``array`` is a memmap of shape
        ``(len(df), max_frames, dim)`` and ``lengths`` the true frame counts.
    """
    config = config or EmbeddingConfig()
    target = (
        Path(path) if path is not None
        else CACHE_DIR / f"frames_{config.hash}_L{layer}.npy"
    )
    lengths_path = target.with_name(target.stem + "_lengths.npy")

    if target.is_file() and lengths_path.is_file():
        return (
            np.load(target, mmap_mode="r"),
            np.load(lengths_path),
            df["filename"].tolist(),
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array, lengths, cursor, truncated = None, [], 0, 0

    for states, frame_counts, _ in extract_frames(df, config, layer):
        if array is None:
            array = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float32,
                shape=(len(df), max_frames, states.shape[-1]),
            )
        take = min(states.shape[1], max_frames)
        array[cursor : cursor + len(states), :take] = states[:, :take]
        truncated += int((frame_counts > max_frames).sum())
        lengths.extend(np.minimum(frame_counts, max_frames).tolist())
        cursor += len(states)

    array.flush()
    lengths = np.asarray(lengths, dtype=np.int32)
    np.save(lengths_path, lengths)
    if truncated:
        print(f"note: {truncated} clips exceeded {max_frames} frames and were truncated")
    return np.load(target, mmap_mode="r"), lengths, df["filename"].tolist()


def feature_names(dim: int, pooling: str) -> list[str]:
    """Column names for a selected matrix, for frames and feature reports."""
    if pooling == "mean":
        return [f"emb_mean_{i:03d}" for i in range(dim)]
    if pooling == "std":
        return [f"emb_std_{i:03d}" for i in range(dim)]
    half = dim // 2
    return [f"emb_mean_{i:03d}" for i in range(half)] + [
        f"emb_std_{i:03d}" for i in range(half)
    ]


# ==========================================================================
# EVALUATE
# ==========================================================================
#: The shared results log. Appended to, never rewritten.
RESULTS_PATH: Path = Path(__file__).resolve().parent.parent / "results" / "results.csv"

#: Exactly the schema in PLAN.md. Do not add columns -- teammates write
#: readers against this. Extra metrics go in the returned dict instead.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "owner",
    "protocol",
    "fold",
    "feature_config",
    "model",
    "hyperparams",
    "accuracy",
    "macro_f1",
    "notes",
)

#: Fixed label order, so confusion matrices from different runs line up.
LABELS: tuple[str, ...] = tuple(EMOTIONS.values())

#: Metadata columns the slice reports are cut by.
SLICE_COLUMNS: tuple[str, ...] = ("gender", "intensity_label", "emotion_label")


def _resolve_owner(owner: str | None = None) -> str:
    """Determine who a result belongs to.

    Five people append to one file, so an unattributed row is a row nobody
    can chase. Reads ``git config user.name`` so nobody has to remember to
    set anything.

    Args:
        owner: Explicit override. Returned as-is when given.

    Returns:
        The owner name, falling back to the OS username, then ``"unknown"``.
    """
    if owner:
        return owner
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - identity is best-effort, never fatal
        return "unknown"


def _slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series
) -> pd.DataFrame:
    """Accuracy and macro-F1 within each level of ``groups``."""
    rows = []
    for level in sorted(pd.unique(groups.dropna())):
        mask = (groups == level).to_numpy()
        if not mask.any():
            continue
        true_here = y_true[mask]
        # Macro-F1 is only meaningful when the slice contains more than one
        # true class. Slicing by emotion gives single-class subsets, where
        # averaging F1 over the seven absent classes produces a number that
        # looks like a score but is not one. Accuracy there equals recall.
        multiclass = len(set(true_here.tolist())) > 1
        rows.append(
            {
                "level": level,
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(true_here, y_pred[mask])),
                "macro_f1": (
                    float(
                        f1_score(
                            true_here,
                            y_pred[mask],
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if multiclass
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    protocol: str,
    fold: int | str,
    metadata: pd.DataFrame,
    *,
    model: str,
    feature_config: str,
    hyperparams: dict[str, Any] | None = None,
    owner: str | None = None,
    notes: str = "",
    results_path: Path | None = None,
    append: bool = True,
) -> dict[str, Any]:
    """Score one set of predictions and log it.

    Args:
        y_true: True emotion labels, e.g. ``"angry"``. Length must match
            ``y_pred`` and ``metadata``.
        y_pred: Predicted emotion labels, same ordering.
        protocol: Split protocol name, from `ser.splits.PROTOCOLS`.
        fold: Fold number, or the string ``"pooled"`` for a pooled
            out-of-fold score.
        metadata: Rows from `ser.metadata.load` for exactly these clips, in
            the same order as ``y_true``. Supplies the slice columns.
        model: Model identifier, e.g. ``"svm-rbf"``.
        feature_config: Feature configuration identifier or hash.
        hyperparams: Serialised to sorted JSON, so identical configurations
            produce identical strings.
        owner: Defaults to ``git config user.name``.
        notes: Free text for the results log.
        results_path: Override the results file. Tests must pass a temporary
            path so the shared log is never polluted.
        append: Set False to compute metrics without logging.

    Returns:
        ``accuracy``, ``macro_f1``, ``weighted_f1``, ``n``, ``per_class``
        (DataFrame), ``confusion_matrix`` (labelled DataFrame in `LABELS`
        order), ``slices`` (dict of DataFrames), and ``row`` (the dict
        written to CSV).

    Raises:
        ValueError: If the three inputs disagree in length.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} labels but y_pred has {len(y_pred)}"
        )
    if len(metadata) != len(y_true):
        raise ValueError(
            f"metadata has {len(metadata)} rows but y_true has {len(y_true)} "
            f"labels; they must describe the same clips in the same order"
        )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "emotion": list(LABELS),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    # Fixed label order so matrices from different runs are comparable, and
    # so a class the model never predicts still gets a row and a column.
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(LABELS)),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )

    meta = metadata.reset_index(drop=True)
    slices = {
        column: _slice_metrics(y_true, y_pred, meta[column])
        for column in SLICE_COLUMNS
        if column in meta.columns
    }

    row = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "owner": _resolve_owner(owner),
        "protocol": protocol,
        "fold": fold,
        "feature_config": feature_config,
        "model": model,
        "hyperparams": json.dumps(hyperparams or {}, sort_keys=True),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "notes": notes,
    }

    if append:
        _append_row(row, results_path)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": len(y_true),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "slices": slices,
        "row": row,
    }


def _append_row(row: dict[str, Any], results_path: Path | None = None) -> Path:
    """Append one result row, writing the header only if the file is new.

    Opened in append mode per call rather than held open, so five people
    running experiments concurrently do not clobber each other.

    Args:
        row: Keys exactly `RESULTS_COLUMNS`.
        results_path: Destination. Defaults to `RESULTS_PATH`.

    Returns:
        The path written to.
    """
    target = Path(results_path) if results_path is not None else RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0

    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in RESULTS_COLUMNS})
    return target


def evaluate_pooled(
    per_fold: Iterable[tuple[Sequence[str], Sequence[str], pd.DataFrame]],
    protocol: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score pooled out-of-fold predictions as a single set.

    The headline number for unequal folds. Concatenates each fold's held-out
    predictions, then delegates to `evaluate` -- the scoring logic lives in
    exactly one place.

    Args:
        per_fold: One ``(y_true, y_pred, metadata)`` triple per fold.
        protocol: Split protocol name.
        **kwargs: Forwarded to `evaluate`. ``fold`` is set to ``"pooled"``
            and must not be passed.

    Returns:
        Whatever `evaluate` returns, over the concatenated predictions.

    Raises:
        ValueError: If ``per_fold`` is empty or ``fold`` was passed.
    """
    if "fold" in kwargs:
        raise ValueError("evaluate_pooled sets fold='pooled'; do not pass fold")

    trues: list[Any] = []
    preds: list[Any] = []
    metas: list[pd.DataFrame] = []
    for y_true, y_pred, meta in per_fold:
        trues.extend(np.asarray(y_true, dtype=object).tolist())
        preds.extend(np.asarray(y_pred, dtype=object).tolist())
        metas.append(meta)

    if not metas:
        raise ValueError("per_fold is empty; nothing to pool")

    return evaluate(
        trues,
        preds,
        protocol,
        "pooled",
        pd.concat(metas, ignore_index=True),
        **kwargs,
    )


def _demo() -> None:  # pragma: no cover - gate evidence, not library code
    """Score two synthetic models to show comparable rows are produced."""

    meta = load().set_index("filename")
    rng = np.random.default_rng(42)

    for name, accuracy in (("synthetic-good", 0.7), ("synthetic-poor", 0.2)):
        per_fold = []
        for fold in range(5):
            _, test = get_split("speaker_independent", fold)
            truth = meta.loc[test, "emotion_label"].to_numpy()
            keep = rng.random(len(truth)) < accuracy
            pred = np.where(keep, truth, rng.choice(LABELS, size=len(truth)))
            per_fold.append((truth, pred, meta.loc[test].reset_index()))
            evaluate(
                truth, pred, "speaker_independent", fold,
                meta.loc[test].reset_index(),
                model=name, feature_config="synthetic",
                hyperparams={"target_accuracy": accuracy},
                notes="demo row, not a real result",
            )
        pooled = evaluate_pooled(
            per_fold, "speaker_independent",
            model=name, feature_config="synthetic",
            hyperparams={"target_accuracy": accuracy},
            notes="demo row, not a real result",
        )
        print(
            f"{name:16s} pooled acc={pooled['accuracy']:.4f} "
            f"macro_f1={pooled['macro_f1']:.4f} n={pooled['n']}"
        )


# ==========================================================================
# STATS
# ==========================================================================
#: Default resamples. 2,000 is enough for a 95% interval to be stable to about
#: the third decimal, which is finer than any difference this project argues
#: about.
N_BOOT: int = 2000

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1, with absent classes scored 0 rather than raising."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain accuracy."""
    return float(accuracy_score(y_true, y_pred))


METRICS: dict[str, Metric] = {"macro_f1": macro_f1, "accuracy": accuracy}


def _resolve(metric: str | Metric) -> Metric:
    if callable(metric):
        return metric
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    return METRICS[metric]


def _speaker_blocks(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique speakers and the row indices belonging to each."""
    unique = np.unique(groups)
    return unique, [np.flatnonzero(groups == speaker) for speaker in unique]


def cluster_bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    metric: str | Metric = "macro_f1",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Confidence interval for one model's metric, resampling speakers.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, same order.
        groups: Speaker id per clip. The resampling unit.
        metric: ``"macro_f1"``, ``"accuracy"``, or a callable.
        n_boot: Bootstrap resamples.
        alpha: 0.05 gives a 95% interval.
        seed: Seed for the resampling RNG.

    Returns:
        ``point``, ``lo``, ``hi``, ``se`` and ``n_speakers``.

    Raises:
        ValueError: If the three inputs disagree in length.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"groups={len(groups)}"
        )

    score = _resolve(metric)
    unique, blocks = _speaker_blocks(groups)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = score(y_true[rows], y_pred[rows])

    return {
        "point": score(y_true, y_pred),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "se": float(np.std(draws, ddof=1)),
        "n_speakers": int(len(unique)),
    }


def paired_cluster_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    groups: Sequence[str],
    metric: str | Metric = "macro_f1",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Confidence interval on the *difference* between two models.

    Both models are scored on the same resampled speakers each iteration, so
    the shared difficulty of a given speaker cancels. This is strictly sharper
    than comparing two separate intervals, and it is the right test when both
    models were evaluated on the same test set -- which, under the frozen
    folds, they always are.

    Args:
        y_true: True labels.
        pred_a: Predictions from model A (the candidate).
        pred_b: Predictions from model B (the baseline).
        groups: Speaker id per clip.
        metric: ``"macro_f1"``, ``"accuracy"``, or a callable.
        n_boot: Bootstrap resamples.
        alpha: 0.05 gives a 95% interval.
        seed: Seed for the resampling RNG.

    Returns:
        ``delta`` (A - B), ``lo``, ``hi``, ``p_value`` (two-sided, the
        proportion of resamples whose difference falls on the other side of
        zero, doubled), and ``significant`` at ``alpha``.
    """
    y_true = np.asarray(y_true, dtype=object)
    pred_a = np.asarray(pred_a, dtype=object)
    pred_b = np.asarray(pred_b, dtype=object)
    groups = np.asarray(groups)

    score = _resolve(metric)
    unique, blocks = _speaker_blocks(groups)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = score(y_true[rows], pred_a[rows]) - score(y_true[rows], pred_b[rows])

    delta = score(y_true, pred_a) - score(y_true, pred_b)
    # Two-sided bootstrap p: how much of the distribution sits across zero.
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    p_value = float(min(1.0, 2 * tail))
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))

    return {
        "delta": float(delta),
        "lo": lo,
        "hi": hi,
        "p_value": p_value,
        "significant": bool(lo > 0 or hi < 0),
        "n_speakers": int(len(unique)),
    }


def mcnemar(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> dict[str, float]:
    """Exact McNemar test on a paired accuracy comparison.

    Conditions on the clips where exactly one model is right. Clips both get
    right, or both get wrong, carry no information about which is better and
    are discarded -- which is what makes this sharper than comparing two
    accuracies.

    Uses the exact binomial test rather than the chi-square approximation,
    because the discordant counts here are often small enough for the
    approximation to misbehave.

    Note this tests *accuracy*, not macro-F1. Macro-F1 is not a per-clip
    average, so it has no McNemar analogue; use `paired_cluster_bootstrap` for
    that. Note also that McNemar treats clips as independent, so it does not
    account for speaker clustering -- read it alongside the cluster bootstrap,
    not instead of it.

    Args:
        y_true: True labels.
        pred_a: Predictions from model A.
        pred_b: Predictions from model B.

    Returns:
        ``only_a_correct``, ``only_b_correct``, ``p_value`` and
        ``significant`` at 0.05.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true, dtype=object)
    correct_a = np.asarray(pred_a, dtype=object) == y_true
    correct_b = np.asarray(pred_b, dtype=object) == y_true

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "only_a_correct": 0,
            "only_b_correct": 0,
            "p_value": 1.0,
            "significant": False,
        }

    result = binomtest(only_a, discordant, 0.5, alternative="two-sided")
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def format_ci(result: dict[str, float], as_pp: bool = False) -> str:
    """Render a bootstrap result as ``point [lo, hi]`` for a report table.

    Args:
        result: Output of `cluster_bootstrap_ci` or `paired_cluster_bootstrap`.
        as_pp: Express in percentage points rather than raw units.

    Returns:
        A string like ``0.6311 [0.5842, 0.6760]`` or ``-11.11 pp [-15.02, -7.30]``.
    """
    scale = 100.0 if as_pp else 1.0
    unit = " pp" if as_pp else ""
    point = result.get("point", result.get("delta"))
    return (
        f"{scale * point:+.2f}{unit} [{scale * result['lo']:+.2f}, "
        f"{scale * result['hi']:+.2f}]"
        if "delta" in result
        else f"{scale * point:.4f} [{scale * result['lo']:.4f}, "
        f"{scale * result['hi']:.4f}]"
    )


# ==========================================================================
# CLASSICAL
# ==========================================================================
#: Model name -> factory. Every estimator uses library defaults apart from the
#: fixed seed, n_jobs, and max_iter where the default does not converge.
MODELS: dict[str, Callable[[], Any]] = {
    "svm-rbf": lambda: SVC(kernel="rbf", random_state=SEED),
    "random-forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "xgboost": lambda: _xgboost(),
    # lbfgs does not converge on 162 features at the default 100 iterations;
    # raising the cap is a convergence fix, not tuning.
    # n_jobs is deliberately absent: it has had no effect since sklearn 1.8
    "logistic-regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "knn": lambda: KNeighborsClassifier(n_jobs=-1),
    "adaboost": lambda: AdaBoostClassifier(random_state=SEED),
    "gaussian-nb": lambda: GaussianNB(),
}


def _xgboost():
    """XGBoost needs integer labels, so it is wrapped at fit time.

    Imported lazily so the module still imports without xgboost installed.
    """
    from xgboost import XGBClassifier

    return _LabelEncodedXGB(
        XGBClassifier(
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    )


class _LabelEncodedXGB(BaseEstimator, ClassifierMixin):
    """Adapt XGBoost's integer-label requirement to our string labels.

    Every other estimator here accepts ``"angry"`` directly. Rather than
    special-casing the caller, the encoding is hidden behind the usual
    fit/predict interface so all seven models are driven identically.

    Inherits `BaseEstimator` so it satisfies the sklearn estimator protocol
    -- `Pipeline` and `GridSearchCV` both introspect estimators via
    ``__sklearn_tags__`` and ``get_params``, which come from that base.
    """

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    def fit(self, x, y):
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder()
        self.classes_ = np.unique(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, self.encoder_.fit_transform(y))
        return self

    def predict(self, x):
        return self.encoder_.inverse_transform(self.estimator_.predict(x))


#: Grid for the winner only, searched by CV *inside* the training fold.
#: Deliberately small: a wide grid on 1,140 training clips overfits the inner
#: CV and buys a number that will not survive the outer fold.
TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "svm-rbf": {
        "estimator__C": [1, 5, 10, 50],
        "estimator__gamma": ["scale", 0.01, 0.001],
    },
    "random-forest": {
        "estimator__n_estimators": [200, 500],
        "estimator__max_depth": [None, 20],
        "estimator__min_samples_leaf": [1, 3],
    },
    "logistic-regression": {"estimator__C": [0.1, 1, 10]},
    "knn": {
        "estimator__n_neighbors": [3, 5, 11, 21],
        "estimator__weights": ["uniform", "distance"],
    },
}


def build_classifier(name: str) -> Pipeline:
    """Build a scaler + estimator pipeline.

    Wrapping the scaler in a `Pipeline` is what guarantees it is fit on
    training data only: `Pipeline.fit` fits the scaler on the training fold,
    and `Pipeline.predict` merely transforms the test fold with those
    already-fitted statistics. There is no code path that fits on test.

    Args:
        name: A key of `MODELS`.

    Returns:
        An unfitted pipeline.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}")
    return Pipeline([("scaler", StandardScaler()), ("estimator", MODELS[name]())])


def classifier_params(name: str) -> dict[str, object]:
    """The estimator's actual parameters, for the results log."""
    estimator = build_classifier(name).named_steps["estimator"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in sorted(params.items())
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def tune_classifier(
    name: str,
    x_train,
    y_train,
    cv: int = 3,
    scoring: str = "f1_macro",
    groups=None,
):
    """Tune one model by CV **inside** the training fold.

    The held-out fold is never touched here -- that is the whole point. When
    ``groups`` is supplied the inner CV is also speaker-grouped, so the inner
    estimate does not itself leak across speakers and inflate the choice of
    hyperparameters.

    Args:
        name: Model name; must have an entry in `TUNING_GRIDS`.
        x_train: Training-fold features.
        y_train: Training-fold labels.
        cv: Inner CV folds.
        scoring: Optimised metric. Macro-F1, not accuracy, so the search does
            not simply abandon neutral.
        groups: Optional speaker ids for grouped inner CV.

    Returns:
        The fitted `GridSearchCV`.

    Raises:
        ValueError: If no grid is defined for ``name``.
    """
    from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

    if name not in TUNING_GRIDS:
        raise ValueError(
            f"No tuning grid for {name!r}; defined for {sorted(TUNING_GRIDS)}"
        )

    splitter = (
        GroupKFold(n_splits=cv)
        if groups is not None
        else StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    )
    search = GridSearchCV(
        build_classifier(name),
        TUNING_GRIDS[name],
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups)
    return search


# ==========================================================================
# CORPORA
# ==========================================================================
RAVDESS_EMOTIONS = EMOTIONS
load_ravdess_metadata = load





#: Repository root. External audio lands beside, never inside, ``data/``.
PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

#: Where CREMA-D / TESS / SAVEE are downloaded to. Gitignored (rule 7).
EXTERNAL_DIR: Path = PROJECT_DIR / "data_external"

#: The built index. Regenerable, therefore gitignored.
INDEX_PATH: Path = EXTERNAL_DIR / "dataset_index.csv"

#: Corpus names, in the order they are concatenated and reported.
SOURCES: tuple[str, ...] = ("RAVDESS", "CREMA-D", "TESS", "SAVEE")

#: The eight canonical labels, in RAVDESS code order (01..08).
CANONICAL: tuple[str, ...] = tuple(RAVDESS_EMOTIONS[f"{i:02d}"] for i in range(1, 9))

#: emotion name -> the two-digit RAVDESS code. External corpora are mapped
#: onto these codes so ``emotion`` means the same thing in every row.
EMOTION_TO_CODE: dict[str, str] = {v: k for k, v in RAVDESS_EMOTIONS.items()}

#: Published clip counts, asserted against what actually lands on disk.
PUBLISHED_COUNTS: dict[str, int] = {
    "RAVDESS": 1440,
    "CREMA-D": 7442,
    "TESS": 2800,
    "SAVEE": 480,
}

# --------------------------------------------------------------------------
# raw code -> canonical emotion, one explicit table per corpus
# --------------------------------------------------------------------------

CREMAD_EMOTION: dict[str, str] = {
    "NEU": "neutral", "HAP": "happy", "SAD": "sad",
    "ANG": "angry", "FEA": "fearful", "DIS": "disgust",
}

TESS_EMOTION: dict[str, str] = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
    "fear": "fearful", "disgust": "disgust",
    "ps": "surprised",  # ps = pleasant surprise
}

SAVEE_EMOTION: dict[str, str] = {
    "n": "neutral", "h": "happy", "sa": "sad", "a": "angry",
    "f": "fearful", "d": "disgust", "su": "surprised",
}

CREMAD_INTENSITY: dict[str, str] = {
    "XX": "unspecified", "LO": "low", "MD": "medium", "HI": "high",
}

#: Every external mapping must land inside the frozen eight. Checked at import
#: so a typo in a table above is a startup failure, not a silent relabel.
for _name, _mapping in (
    ("CREMA-D", CREMAD_EMOTION), ("TESS", TESS_EMOTION), ("SAVEE", SAVEE_EMOTION)
):
    _unknown = set(_mapping.values()) - set(CANONICAL)
    if _unknown:
        raise ValueError(f"{_name} would introduce non-canonical labels: {_unknown}")

#: The index schema. Anything not listed here is a feature column -- pass this
#: to `ser.features.feature_columns` as ``meta`` when working on merged
#: frames, otherwise it defaults to RAVDESS's columns and mistakes the extra
#: index columns for features.
INDEX_COLUMNS: tuple[str, ...] = (
    "filename", "filepath", "source", "speaker_id", "actor", "gender",
    "emotion", "intensity", "statement", "repetition",
    "emotion_label", "intensity_label", "statement_label", "repetition_label",
    "label", "label_idx",
)


def _row(
    *,
    path: Path,
    source: str,
    speaker_id: str,
    gender: str,
    emotion_label: str,
    intensity: str = "NA",
    intensity_label: str = "unspecified",
    statement: str = "NA",
    repetition: str = "01",
    actor: object = pd.NA,
) -> dict[str, object]:
    """Build one index row in the canonical schema.

    Every parser funnels through here so the four corpora cannot drift into
    slightly different column semantics.

    Args:
        path: Absolute path to the clip.
        source: One of `SOURCES`.
        speaker_id: Globally unique, corpus-prefixed.
        gender: ``"male"``, ``"female"`` or ``"unknown"``.
        emotion_label: One of `CANONICAL`.
        intensity: Raw intensity code as the corpus writes it.
        intensity_label: Human-readable intensity.
        statement: Raw sentence/word/take identifier.
        repetition: Take number; ``"01"`` where a corpus has no repetitions.
        actor: RAVDESS actor number, or ``pd.NA`` for external corpora.

    Returns:
        A dict with exactly the keys in `INDEX_COLUMNS`.
    """
    code = EMOTION_TO_CODE[emotion_label]
    return {
        "filename": path.name,
        "filepath": str(path),
        "source": source,
        "speaker_id": speaker_id,
        "actor": actor,
        "gender": gender,
        "emotion": code,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "emotion_label": emotion_label,
        "intensity_label": intensity_label,
        "statement_label": statement,
        "repetition_label": repetition,
        "label": int(code),
        "label_idx": int(code) - 1,
    }


# ==========================================================================
# Fetching
# ==========================================================================

CREMAD_TREE = (
    "https://api.github.com/repos/CheyneyComputerScience/CREMA-D"
    "/git/trees/master?recursive=1"
)
CREMAD_MEDIA = (
    "https://media.githubusercontent.com/media/"
    "CheyneyComputerScience/CREMA-D/master/"
)
CREMAD_DEMO = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D"
    "/master/VideoDemographics.csv"
)

TESS_DOI = "doi:10.5683/SP2/E8H2MF"
TESS_DATASET = (
    "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=" + TESS_DOI
)
TESS_DATAFILE = "https://borealisdata.ca/api/access/datafiles/"

SAVEE_KAGGLE = "ejlok1/surrey-audiovisual-expressed-emotion-savee"


def _session():
    """An HTTP session that retries the transient failures these hosts throw.

    Built lazily so importing this module costs nothing when no download is
    happening, and so ``requests`` stays optional for anyone who only ever
    works with the local RAVDESS clips.
    """
    import requests

    retry = requests.adapters.Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=32),
    )
    return session


def _is_wav(blob: bytes) -> bool:
    """True if ``blob`` really is a RIFF/WAVE payload.

    The guard that stops a Git-LFS pointer file or an HTML error page being
    written out with a ``.wav`` extension, where it would surface much later
    as an inexplicable decode failure.
    """
    return blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"


def fetch_cremad(
    dest: Path | None = None, limit: int | None = None, workers: int = 8
) -> Path:
    """Download CREMA-D's AudioWAV directory and speaker demographics.

    Resumable: files already on disk are skipped, so re-running after an
    interruption downloads only what is missing.

    Args:
        dest: Destination. Defaults to ``data_external/CREMA-D``.
        limit: Cap the number of files, for a smoke test. ``None`` fetches all.
        workers: Parallel download threads.

    Returns:
        The destination directory.

    Raises:
        RuntimeError: If GitHub truncates the tree listing.
    """
    from tqdm.auto import tqdm

    dest = Path(dest) if dest is not None else EXTERNAL_DIR / "CREMA-D"
    audio = dest / "AudioWAV"
    audio.mkdir(parents=True, exist_ok=True)
    session = _session()

    tree = session.get(CREMAD_TREE, timeout=90).json()
    if tree.get("truncated"):
        raise RuntimeError(
            "GitHub truncated the CREMA-D tree listing; use the Kaggle mirror"
        )
    wavs = sorted(
        entry["path"]
        for entry in tree["tree"]
        if entry["path"].startswith("AudioWAV/")
        and entry["path"].lower().endswith(".wav")
    )
    if limit:
        wavs = wavs[:limit]

    demo = dest / "VideoDemographics.csv"
    if not demo.exists():
        demo.write_bytes(session.get(CREMAD_DEMO, timeout=60).content)

    todo = [p for p in wavs if not (audio / Path(p).name).exists()]
    print(
        f"CREMA-D: {len(wavs):,} listed, {len(wavs) - len(todo):,} on disk, "
        f"{len(todo):,} to download"
    )

    def grab(path: str) -> None:
        blob = session.get(CREMAD_MEDIA + path, timeout=180).content
        if not _is_wav(blob):
            raise RuntimeError(
                f"{path}: {len(blob)} bytes that are not a WAV "
                f"(an LFS pointer or an error page)"
            )
        (audio / Path(path).name).write_bytes(blob)

    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(grab, p): p for p in todo}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="CREMA-D", unit="file"
        ):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect, report, continue
                errors.append((futures[future], repr(exc)))

    if errors:
        print(f"  {len(errors)} failed (re-run to retry): {errors[:3]}")
    print(f"  -> {len(list(audio.glob('*.wav'))):,} wav in {audio}")
    return dest


def fetch_tess(
    dest: Path | None = None, limit: int | None = None, chunk: int = 200
) -> Path:
    """Download TESS from Borealis Dataverse as batched zips.

    Args:
        dest: Destination. Defaults to ``data_external/TESS``.
        limit: Cap the number of files, for a smoke test.
        chunk: Datafile ids per zip request.

    Returns:
        The destination directory.
    """
    from tqdm.auto import tqdm

    dest = Path(dest) if dest is not None else EXTERNAL_DIR / "TESS"
    dest.mkdir(parents=True, exist_ok=True)
    session = _session()

    meta = session.get(TESS_DATASET, timeout=90).json()
    files = [
        (entry["dataFile"]["id"], entry["dataFile"]["filename"])
        for entry in meta["data"]["latestVersion"]["files"]
        if entry["dataFile"]["filename"].lower().endswith(".wav")
    ]
    if limit:
        files = files[:limit]

    todo = [(i, n) for i, n in files if not (dest / n).exists()]
    print(
        f"TESS: {len(files):,} listed, {len(files) - len(todo):,} on disk, "
        f"{len(todo):,} to download"
    )

    written = 0
    for start in tqdm(range(0, len(todo), chunk), desc="TESS", unit="batch"):
        ids = ",".join(str(i) for i, _ in todo[start : start + chunk])
        blob = session.get(TESS_DATAFILE + ids, timeout=600).content
        # Raises BadZipFile if Dataverse returned an error document instead.
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".wav"):
                    (dest / Path(name).name).write_bytes(archive.read(name))
                    written += 1

    print(f"  -> {written:,} new, {len(list(dest.glob('*.wav'))):,} wav in {dest}")
    return dest


def fetch_savee(dest: Path | None = None) -> Path:
    """Fetch SAVEE, which has no unauthenticated public mirror.

    Surrey requires registration, so three routes are tried in order:

    1. ``kagglehub``, using credentials the user has **already** authorised
       under ``~/.kaggle/``. This function never prompts for, reads or
       transmits credentials itself.
    2. Any ``*.zip`` the user has dropped into ``data_external/``.
    3. Wavs already extracted under the destination.

    Args:
        dest: Destination. Defaults to ``data_external/SAVEE``.

    Returns:
        The destination directory, which may be empty if no route succeeded.
        SAVEE is 3.9% of the merge; the rest of the pipeline runs without it.
    """
    dest = Path(dest) if dest is not None else EXTERNAL_DIR / "SAVEE"
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.rglob("*.wav"))
    if existing:
        print(f"SAVEE: {len(existing)} wav already in {dest}")
        return dest

    try:
        import kagglehub

        src = Path(kagglehub.dataset_download(SAVEE_KAGGLE))
        wavs = list(src.rglob("*.wav"))
        for wav in wavs:
            target = (
                dest / wav.name
                if wav.parent == src
                else dest / wav.parent.name / wav.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wav.read_bytes())
        print(f"SAVEE: {len(wavs)} wav copied from the Kaggle cache at {src}")
        return dest
    except ImportError:
        print("SAVEE: kagglehub not installed -> pip install kagglehub")
    except Exception as exc:  # noqa: BLE001 - fall through to the manual routes
        print(f"SAVEE: kagglehub route failed ({exc!r})")

    for archive_path in sorted(EXTERNAL_DIR.rglob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".wav")]
            if not names:
                continue
            for name in names:
                rel = Path(name)
                target = (
                    dest / rel.parent.name / rel.name
                    if rel.parent.name
                    else dest / rel.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            print(f"SAVEE: {len(names)} wav extracted from {archive_path.name}")
            return dest

    print(
        "SAVEE: nothing to extract. Either\n"
        f"  a) authorise kagglehub for {SAVEE_KAGGLE} and re-run, or\n"
        "  b) register at http://kahlan.eps.surrey.ac.uk/savee/Download.html,\n"
        f"     drop AudioData.zip into {EXTERNAL_DIR}/ and re-run.\n"
        "The pipeline runs without it; SAVEE is simply skipped."
    )
    return dest


# ==========================================================================
# Parsing -- one function per corpus, all returning `INDEX_COLUMNS`
# ==========================================================================


def parse_ravdess(root: Path | None = None) -> pd.DataFrame:
    """Index the base corpus by delegating to `ser.metadata.load`.

    Deliberately *not* a second RAVDESS parser. `ser.metadata` is strict about
    the five-field schema, checks the actor field against the ``PersonNN``
    directory, and validates the dataset-level invariants; duplicating that
    here would create two sources of truth that could drift apart.

    Args:
        root: Directory holding the ``PersonNN`` folders. Defaults to
            `ser.metadata.DATA_ROOT`.

    Returns:
        One row per clip, with the columns in `INDEX_COLUMNS`.
    """
    meta = load_ravdess_metadata(root=root)
    out = meta.drop(columns=["modality", "vocal_channel"]).copy()
    out["source"] = "RAVDESS"
    out["speaker_id"] = out["actor"].map(lambda a: f"RAVDESS_{int(a):02d}")
    out["label"] = out["emotion"].astype(int)
    out["label_idx"] = out["label"] - 1
    return out[list(INDEX_COLUMNS)]


def parse_cremad(root: Path | None = None) -> pd.DataFrame:
    """Index CREMA-D from ``actor_sentence_EMOTION_intensity.wav`` filenames.

    Speaker sex comes from the corpus's own ``VideoDemographics.csv`` rather
    than being inferred; where that file is absent the sex is recorded as
    ``"unknown"`` instead of guessed, so the gender slice stays honest.

    Args:
        root: Corpus directory. Defaults to ``data_external/CREMA-D``.

    Returns:
        One row per clip, or an empty frame if the corpus is not present.

    Raises:
        ValueError: On a filename that is not four underscore-separated fields.
        KeyError: On an emotion or intensity code outside the mapping -- by
            design, so an unexpected code is loud rather than dropped.
    """
    root = Path(root) if root is not None else EXTERNAL_DIR / "CREMA-D"
    audio = root / "AudioWAV"
    if not audio.is_dir():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    sex_of: dict[str, str] = {}
    demo = root / "VideoDemographics.csv"
    if demo.exists():
        frame = pd.read_csv(demo)
        sex_of = {
            str(r.ActorID): ("male" if r.Sex == "Male" else "female")
            for r in frame.itertuples()
        }

    rows = []
    for path in sorted(audio.glob("*.wav")):
        fields = path.stem.split("_")
        if len(fields) != 4:
            raise ValueError(f"unexpected CREMA-D filename: {path.name}")
        actor, sentence, emotion, intensity = fields
        rows.append(
            _row(
                path=path,
                source="CREMA-D",
                speaker_id=f"CREMA-D_{actor}",
                gender=sex_of.get(actor, "unknown"),
                emotion_label=CREMAD_EMOTION[emotion],
                intensity=intensity,
                intensity_label=CREMAD_INTENSITY[intensity],
                statement=sentence,
            )
        )
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


def parse_tess(root: Path | None = None) -> pd.DataFrame:
    """Index TESS from ``speaker_word_emotion.wav`` filenames.

    One corpus quirk is handled explicitly: a single file ships as
    ``OA_bite_neutral.wav``, missing the ``F`` of ``OAF``. Both TESS speakers
    are female, so ``gender`` is constant.

    Args:
        root: Corpus directory. Defaults to ``data_external/TESS``.

    Returns:
        One row per clip, or an empty frame if the corpus is not present.

    Raises:
        ValueError: On a filename that is not three underscore-separated fields.
        KeyError: On an emotion code outside `TESS_EMOTION`.
    """
    root = Path(root) if root is not None else EXTERNAL_DIR / "TESS"
    if not root.is_dir():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    rows = []
    for path in sorted(root.rglob("*.wav")):
        fields = path.stem.split("_")
        if len(fields) != 3:
            raise ValueError(f"unexpected TESS filename: {path.name}")
        speaker, word, emotion = fields[0].upper(), fields[1], fields[2].lower()
        if speaker == "OA":  # OA_bite_neutral.wav -- a typo in the corpus
            speaker = "OAF"
        rows.append(
            _row(
                path=path,
                source="TESS",
                speaker_id=f"TESS_{speaker}",
                gender="female",  # both TESS speakers are female
                emotion_label=TESS_EMOTION[emotion],
                statement=word,
            )
        )
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


#: Longest-first alternation, so ``sa``/``su`` are matched before ``a``/``s``.
#: With a naive character class, every ``sad`` clip would be read as ``angry``.
SAVEE_RE = re.compile(r"^(?:([A-Za-z]{2})[_-])?(sa|su|a|d|f|h|n)(\d+)$", re.IGNORECASE)

#: The four SAVEE speakers, all male.
SAVEE_SPEAKERS: frozenset[str] = frozenset({"DC", "JE", "JK", "KL"})


def parse_savee(root: Path | None = None) -> pd.DataFrame:
    """Index SAVEE, handling both the flat and per-speaker-folder layouts.

    The speaker is either the filename prefix (``DC_a01.wav``) or the parent
    directory (``DC/a01.wav``), depending on which distribution was obtained.

    Args:
        root: Corpus directory. Defaults to ``data_external/SAVEE``.

    Returns:
        One row per clip, or an empty frame if the corpus is not present.

    Raises:
        ValueError: On an unparseable filename or an unresolvable speaker.
    """
    root = Path(root) if root is not None else EXTERNAL_DIR / "SAVEE"
    if not root.is_dir():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    rows = []
    for path in sorted(root.rglob("*.wav")):
        match = SAVEE_RE.match(path.stem)
        if not match:
            raise ValueError(f"unexpected SAVEE filename: {path}")
        prefix, code, take = match.groups()
        speaker = (prefix or path.parent.name).upper()
        if speaker not in SAVEE_SPEAKERS:
            raise ValueError(f"cannot resolve SAVEE speaker for {path}")
        rows.append(
            _row(
                path=path,
                source="SAVEE",
                speaker_id=f"SAVEE_{speaker}",
                gender="male",  # all four SAVEE speakers are male
                emotion_label=SAVEE_EMOTION[code.lower()],
                statement=take,
            )
        )
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


_PARSERS = {
    "RAVDESS": parse_ravdess,
    "CREMA-D": parse_cremad,
    "TESS": parse_tess,
    "SAVEE": parse_savee,
}


# ==========================================================================
# Building and validating the index
# ==========================================================================


def build_index(
    sources: tuple[str, ...] | list[str] = SOURCES,
    write: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Concatenate the requested corpora into one index.

    ``sources`` is a parameter rather than a constant because *which* external
    corpora to train on is itself an experiment: CREMA-D contributes no
    ``calm`` and no ``surprised``, so adding it may shift the priors against
    those two classes. That is measured, not assumed.

    Rows are sorted by ``(source, speaker_id, filename)`` and the index reset,
    so row order is identical on every machine regardless of glob order.

    Args:
        sources: Which corpora to include. Order is ignored; `SOURCES` order
            is always used.
        write: Persist to `INDEX_PATH`.
        validate: Run `validate_index` and raise on violation.

    Returns:
        One row per clip, with the columns in `INDEX_COLUMNS`.

    Raises:
        ValueError: On an unknown source name, or from `validate_index`.
    """
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)}; expected {SOURCES}")

    frames = [_PARSERS[name]() for name in SOURCES if name in sources]
    present = [f for f in frames if not f.empty]
    if not present:
        raise ValueError(f"no clips found for any of {sorted(sources)}")

    index = pd.concat(present, ignore_index=True)
    index["source"] = pd.Categorical(index["source"], categories=list(SOURCES))
    index = (
        index.sort_values(["source", "speaker_id", "filename"], kind="mergesort")
        .reset_index(drop=True)
    )

    if validate:
        validate_index(index)

    if write:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(INDEX_PATH, index=False, lineterminator="\n")
    return index


def validate_index(index: pd.DataFrame) -> None:
    """Assert the merge's structural contract.

    Every failure is collected before raising, so one run reports the whole
    picture rather than the first problem it meets.

    The checks are chosen to catch the ways a multi-corpus merge goes wrong
    silently: a label outside the frozen eight, a clip counted twice, a
    ``speaker_id`` that spans two corpora (which would leak across any
    grouped split), and an indexed file that is not actually on disk.

    Args:
        index: A frame in the shape `build_index` returns.

    Raises:
        ValueError: Listing every invariant that failed.
    """
    problems: list[str] = []

    stray = set(index["emotion_label"].unique()) - set(CANONICAL)
    if stray:
        problems.append(f"non-canonical emotion label(s): {sorted(stray)}")

    if not index["label"].between(1, 8).all():
        problems.append("label outside 1..8")

    # Computed defensively: a non-canonical label maps to NaN, and coercing
    # that to int would raise here -- turning the validator into a crash
    # instead of the full report it exists to produce.
    mapped = pd.to_numeric(index["emotion_label"].map(EMOTION_TO_CODE), errors="coerce")
    known = mapped.notna()
    if known.any() and not (index.loc[known, "label_idx"] == mapped[known] - 1).all():
        problems.append("label_idx out of sync with emotion_label")

    if index["filepath"].duplicated().any():
        problems.append(
            f"{int(index['filepath'].duplicated().sum())} duplicate filepath rows"
        )

    # Fold files key on ``filename``, not path, so a basename shared by two
    # corpora would silently merge two different clips into one fold entry.
    # The four naming schemes happen not to collide -- this asserts it rather
    # than relying on it.
    if index["filename"].duplicated().any():
        clashes = sorted(index.loc[index["filename"].duplicated(), "filename"].unique())
        problems.append(
            f"{len(clashes)} filename(s) shared across corpora, e.g. {clashes[:3]} -- "
            f"fold files key on filename, so these would collide"
        )

    missing_cols = [c for c in INDEX_COLUMNS if c not in index.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")

    for column in ("emotion_label", "speaker_id", "source", "gender"):
        if column in index.columns and index[column].isna().any():
            problems.append(f"null values in {column}")

    spanning = (
        index.groupby("speaker_id", observed=True)["source"].nunique().pipe(
            lambda s: s[s > 1]
        )
    )
    if not spanning.empty:
        problems.append(
            f"speaker_id spans two corpora: {sorted(spanning.index)[:5]} -- "
            f"a grouped split would leak across them"
        )

    absent = [p for p in index["filepath"] if not Path(p).is_file()]
    if absent:
        problems.append(f"{len(absent)} indexed files do not exist, e.g. {absent[:3]}")

    # Only corpora that actually contributed rows. ``source`` is categorical,
    # so value_counts() otherwise reports the absent ones as present-but-empty
    # and every partial merge would fail this check.
    for source, count in index["source"].value_counts().items():
        if int(count) == 0:
            continue
        expected = PUBLISHED_COUNTS.get(str(source))
        if expected is not None and int(count) != expected:
            problems.append(
                f"{source}: {int(count)} clips on disk, published size is {expected}"
            )

    if problems:
        raise ValueError("Merged index invariants failed:\n  - " + "\n  - ".join(problems))


def load_index(path: Path | None = None) -> pd.DataFrame:
    """Read a previously built index.

    Args:
        path: Source. Defaults to `INDEX_PATH`.

    Returns:
        The index frame, with ``source`` restored as an ordered categorical.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    source_path = Path(path) if path is not None else INDEX_PATH
    if not source_path.is_file():
        raise FileNotFoundError(
            f"No index at {source_path}. Build it once with "
            f"ser.corpora.build_index()."
        )
    index = pd.read_csv(source_path, dtype={"emotion": str, "intensity": str})
    index["source"] = pd.Categorical(index["source"], categories=list(SOURCES))
    return index


def summarise_corpora(index: pd.DataFrame) -> pd.DataFrame:
    """Per-corpus shape of the merge: clips, speakers, emotions, gender.

    Args:
        index: A frame in the shape `build_index` returns.

    Returns:
        One row per corpus present, plus a TOTAL row.
    """
    rows = []
    for source, group in index.groupby("source", observed=True):
        rows.append(
            {
                "source": source,
                "clips": len(group),
                "speakers": group["speaker_id"].nunique(),
                "emotions": group["emotion_label"].nunique(),
                "male": int((group["gender"] == "male").sum()),
                "female": int((group["gender"] == "female").sum()),
                "unknown_sex": int((group["gender"] == "unknown").sum()),
            }
        )
    out = pd.DataFrame(rows)
    total = {
        "source": "TOTAL",
        "clips": len(index),
        "speakers": index["speaker_id"].nunique(),
        "emotions": index["emotion_label"].nunique(),
        "male": int((index["gender"] == "male").sum()),
        "female": int((index["gender"] == "female").sum()),
        "unknown_sex": int((index["gender"] == "unknown").sum()),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


# ==========================================================================
# SPLITS MERGED
# ==========================================================================
#: The merged fold assignments. Frozen on first write, like ``folds.csv``.
MERGED_FOLDS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "splits" / "folds_merged.csv"
)

MERGED_PROTOCOLS: tuple[str, ...] = (
    "speaker_independent_merged",
    "corpus_holdout",
    "random_stratified_merged",
)

_MergedSplit = tuple[np.ndarray, np.ndarray]


# ==========================================================================
# Architecture A -- RAVDESS test side, external clips added to training
# ==========================================================================


def augmented_train(
    fold: int,
    sources: tuple[str, ...] | list[str] = (),
    index: pd.DataFrame | None = None,
    protocol: str = "speaker_independent",
) -> tuple[list[str], list[str]]:
    """Frozen RAVDESS fold, with external corpora added to the training side.

    The test side is returned exactly as ``splits/folds.csv`` defines it. The
    training side is that fold's RAVDESS training clips plus every clip from
    the requested external corpora.

    ``sources`` is a parameter because *which* corpora help is an experiment,
    not a setting: CREMA-D contributes no ``calm`` and no ``surprised``, so it
    may shift the priors against exactly the two classes RAVDESS is weakest on.

    Args:
        fold: Fold number, 0-based, in the frozen RAVDESS protocol.
        sources: External corpora to add to training. ``()`` reproduces the
            RAVDESS-only baseline exactly.
        index: Merged index from `ser.corpora.build_index`. Defaults to
            reading the built index from disk.
        protocol: Which frozen RAVDESS protocol supplies the fold.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If ``sources`` names RAVDESS (it is always present), an
            unknown corpus, or if the resulting test side is not RAVDESS-only.
    """
    requested = tuple(sources)
    if "RAVDESS" in requested:
        raise ValueError(
            "RAVDESS is the test corpus and is always in training; "
            "list only external corpora in `sources`"
        )
    unknown = set(requested) - set(corpora.SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)}")

    train, test = get_split(protocol, fold)

    if requested:
        if index is None:
            index = corpora.load_index()
        extra = index.loc[
            index["source"].isin(requested), "filename"
        ].tolist()
        train = list(train) + extra

    # The property this architecture rests on. Assert it rather than trust it:
    # a single external clip on the test side would reintroduce the corpus cue
    # that holding the test set to RAVDESS exists to eliminate.
    if index is not None:
        lookup = index.set_index("filename")["source"]
        leaked = {
            str(lookup[name])
            for name in test
            if name in lookup.index and str(lookup[name]) != "RAVDESS"
        }
        if leaked:
            raise ValueError(
                f"test side contains non-RAVDESS clips from {sorted(leaked)}; "
                f"Architecture A requires a RAVDESS-only test set"
            )

    return train, test


# ==========================================================================
# Architecture B -- merged test folds
# ==========================================================================


def speaker_independent_merged(
    index: pd.DataFrame, n_splits: int = N_SPLITS
) -> list[_MergedSplit]:
    """5-fold GroupKFold on ``speaker_id`` across every corpus.

    No speaker appears on both sides of a fold. Note that this does **not**
    hold out a corpus: all four corpora appear in both training and test, so a
    model may still exploit recording conditions. `corpus_holdout` is the
    protocol that removes that.

    Args:
        index: Merged index frame.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return list(
        splitter.split(index, index["emotion_label"], groups=index["speaker_id"])
    )


def corpus_holdout(index: pd.DataFrame) -> list[_MergedSplit]:
    """Leave-one-corpus-out — the honest cross-corpus number.

    One fold per corpus present: train on the other three, test on this one.
    Both the speakers and the recording conditions differ between the sides,
    which is the situation a deployed model actually faces.

    Expect a large drop against `speaker_independent_merged`. That drop is the
    result, not a failure: it is the part of the merged-test score that was
    coming from shared recording conditions.

    A corpus missing a label cannot be scored on it -- CREMA-D has no ``calm``
    and no ``surprised`` -- so per-class metrics on those folds are reported
    over the labels actually present.

    Args:
        index: Merged index frame.

    Returns:
        ``(train_idx, test_idx)`` pairs, one per corpus, in `SOURCES` order.
    """
    positions = np.arange(len(index))
    source = index["source"].astype(str).to_numpy()
    out: list[_MergedSplit] = []
    for name in corpora.SOURCES:
        mask = source == name
        if not mask.any():
            continue
        out.append((positions[~mask], positions[mask]))
    return out


def random_stratified_merged(
    index: pd.DataFrame, n_splits: int = N_SPLITS
) -> list[_MergedSplit]:
    """5-fold StratifiedKFold on emotion — the leakage comparison only.

    The same speaker, and the same corpus, appear on both sides of every fold.
    That is the defect this protocol exists to quantify by comparison against
    `speaker_independent_merged`. Never report this number alone.

    Args:
        index: Merged index frame.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(index, index["emotion_label"]))


_MERGED_BUILDERS = {
    "speaker_independent_merged": speaker_independent_merged,
    "corpus_holdout": corpus_holdout,
    "random_stratified_merged": random_stratified_merged,
}


def build_merged_folds(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the long-format fold table for every merged protocol.

    Args:
        index: Merged index frame. Defaults to the built index on disk.

    Returns:
        Columns ``protocol, fold, filename, split``, deterministically sorted
        so the CSV is byte-stable across machines.
    """
    if index is None:
        index = corpora.load_index()

    filenames = index["filename"].to_numpy()
    records: list[pd.DataFrame] = []

    for protocol in MERGED_PROTOCOLS:
        for fold, (train_idx, test_idx) in enumerate(_MERGED_BUILDERS[protocol](index)):
            for side, idx in (("train", train_idx), ("test", test_idx)):
                records.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "filename": filenames[idx],
                            "split": side,
                        }
                    )
                )

    folds = pd.concat(records, ignore_index=True)
    folds["_order"] = folds["protocol"].map(
        {name: i for i, name in enumerate(MERGED_PROTOCOLS)}
    )
    folds = (
        folds.sort_values(["_order", "fold", "split", "filename"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return folds[list(FOLDS_COLUMNS)]


def write_merged_folds(
    path: Path | None = None,
    force: bool = False,
    index: pd.DataFrame | None = None,
) -> Path:
    """Write the merged fold assignments to CSV, once.

    Refuses to overwrite, for the same reason `ser.splits.write_folds` does:
    the moment a result is logged against these assignments, regenerating them
    silently invalidates it.

    Args:
        path: Destination. Defaults to `MERGED_FOLDS_PATH`.
        force: Overwrite an existing file. Do not use.
        index: Merged index to build from. Defaults to the built index on
            disk. Passed explicitly by tests, which must not depend on a
            corpus having been downloaded.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    target = Path(path) if path is not None else MERGED_FOLDS_PATH
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists and merged folds are frozen. Every "
            f"Architecture B result assumes these exact assignments."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    build_merged_folds(index).to_csv(target, index=False, lineterminator="\n")
    return target


def load_merged_folds(path: Path | None = None) -> pd.DataFrame:
    """Read the frozen merged fold assignments.

    Args:
        path: Source. Defaults to `MERGED_FOLDS_PATH`.

    Returns:
        Columns ``protocol, fold, filename, split``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(path) if path is not None else MERGED_FOLDS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"No merged fold assignments at {source}. Generate them once with "
            f"ser.splits_merged.write_merged_folds(), then commit the file."
        )
    return pd.read_csv(source, dtype={"protocol": str, "filename": str, "split": str})


def get_merged_split(
    protocol: str, fold: int, folds: pd.DataFrame | None = None
) -> tuple[list[str], list[str]]:
    """Return the train and test filenames for one merged (protocol, fold).

    Args:
        protocol: One of `MERGED_PROTOCOLS`.
        fold: Fold number, 0-based.
        folds: Fold table. Defaults to reading the frozen CSV.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If the protocol is unknown or the fold does not exist.
    """
    if protocol not in MERGED_PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {MERGED_PROTOCOLS}"
        )
    table = load_merged_folds() if folds is None else folds
    subset = table[(table["protocol"] == protocol) & (table["fold"] == fold)]
    if subset.empty:
        available = sorted(table.loc[table["protocol"] == protocol, "fold"].unique())
        raise ValueError(
            f"No fold {fold} for protocol {protocol!r}; available: {available}"
        )
    train = subset.loc[subset["split"] == "train", "filename"].tolist()
    test = subset.loc[subset["split"] == "test", "filename"].tolist()
    return train, test


def summarise_merged_folds(
    folds: pd.DataFrame | None = None, index: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-fold composition of every merged protocol's test side.

    The table to read before trusting any per-fold number, because merged
    folds are necessarily uneven.

    Args:
        folds: Fold table. Defaults to the frozen CSV.
        index: Merged index frame. Defaults to the built index.

    Returns:
        One row per (protocol, fold) with sizes, speaker and corpus counts,
        overlap checks, gender split and the two sparse classes' counts.
    """
    table = load_merged_folds() if folds is None else folds
    frame = corpora.load_index() if index is None else index
    lookup = frame.set_index("filename")

    rows = []
    for (protocol, fold), group in table.groupby(["protocol", "fold"], sort=False):
        train = lookup.loc[group.loc[group["split"] == "train", "filename"]]
        test = lookup.loc[group.loc[group["split"] == "test", "filename"]]
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_speakers": train["speaker_id"].nunique(),
                "test_speakers": test["speaker_id"].nunique(),
                "speaker_overlap": len(
                    set(train["speaker_id"]) & set(test["speaker_id"])
                ),
                "corpus_overlap": len(
                    set(train["source"].astype(str)) & set(test["source"].astype(str))
                ),
                "test_corpora": ",".join(sorted(set(test["source"].astype(str)))),
                "test_pct_male": round(100 * (test["gender"] == "male").mean(), 1),
                "test_emotions": test["emotion_label"].nunique(),
                "test_calm": int((test["emotion_label"] == "calm").sum()),
                "test_neutral": int((test["emotion_label"] == "neutral").sum()),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# ARMS
# ==========================================================================
ARMS: tuple[str, ...] = ("ravdess_only", "pooled", "two_stage", "two_stage_frozen")

#: Arms whose second stage must not update the learned representation.
FREEZE_BACKBONE: frozenset[str] = frozenset({"two_stage_frozen"})


@dataclass(frozen=True)
class ArmSplit:
    """The data a single arm needs, for one fold.

    Attributes:
        arm: Which arm this is.
        fold: Fold number, 0-based.
        stage1: Filenames to fit on first, or ``None`` when the arm is
            single-stage. For the two-stage arms this is every clip from all
            four corpora that is not in the test fold.
        stage2: Filenames for the second fit. For single-stage arms this is the
            only fit. Always RAVDESS-only except under ``pooled``.
        test: The held-out filenames. **Always RAVDESS-only**, and identical
            across all four arms.
        freeze_backbone: Whether stage 2 should update only the final
            classification layer and leave the representation fixed.
    """

    arm: str
    fold: int
    stage1: list[str] | None
    stage2: list[str]
    test: list[str]
    freeze_backbone: bool

    def describe(self) -> str:
        """One-line summary, for logging what an experiment actually ran on."""
        first = f"{len(self.stage1):,}" if self.stage1 else "-"
        note = " (frozen backbone)" if self.freeze_backbone else ""
        return (
            f"{self.arm:<17s} fold {self.fold}  "
            f"stage1={first:>7s}  stage2={len(self.stage2):>6,}{note}  "
            f"test={len(self.test):>4,}"
        )


def arm_split(
    arm: str,
    fold: int,
    index: pd.DataFrame | None = None,
    protocol: str = "speaker_independent",
) -> ArmSplit:
    """Build one arm's data for one fold.

    Args:
        arm: One of `ARMS`.
        fold: Fold number, 0-based.
        index: Merged index from `ser.corpora.build_index`. Defaults to the
            built index on disk. Only needed for the arms that use external
            data.
        protocol: Which frozen RAVDESS protocol supplies the fold. Leave as the
            default unless you know why you are changing it.

    Returns:
        An `ArmSplit`.

    Raises:
        ValueError: On an unknown arm, or if an external clip would reach the
            test side -- which would silently invalidate the comparison.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    ravdess_train, test = get_split(protocol, fold)

    if arm == "ravdess_only":
        return ArmSplit(arm, fold, None, list(ravdess_train), list(test), False)

    if index is None:
        index = corpora.load_index()
    source = index.set_index("filename")["source"].astype(str)
    external = source.index[source != "RAVDESS"].tolist()

    leaked = {str(source[n]) for n in test if n in source.index and source[n] != "RAVDESS"}
    if leaked:
        raise ValueError(
            f"test side contains clips from {sorted(leaked)}; every arm must be "
            f"scored on RAVDESS alone or the comparison is meaningless"
        )

    everything = list(ravdess_train) + external

    if arm == "pooled":
        return ArmSplit(arm, fold, None, everything, list(test), False)

    # both two-stage arms share the same data; they differ only in whether the
    # representation is allowed to move during stage 2
    return ArmSplit(
        arm, fold, everything, list(ravdess_train), list(test),
        arm in FREEZE_BACKBONE,
    )


def all_arms(
    fold: int, index: pd.DataFrame | None = None
) -> dict[str, ArmSplit]:
    """Every arm for one fold, so a caller can loop over the comparison."""
    if index is None:
        index = corpora.load_index()
    return {arm: arm_split(arm, fold, index=index) for arm in ARMS}


def summarise_arms(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """The four arms across all five folds, as a table.

    Reproduces the design table in `docs/two_stage_findings.md` from the actual
    fold definitions rather than from prose, so the two cannot drift apart.
    """
    if index is None:
        index = corpora.load_index()
    rows = []
    for fold in range(5):
        for arm, split in all_arms(fold, index=index).items():
            rows.append({
                "arm": arm,
                "fold": fold,
                "stage1": len(split.stage1) if split.stage1 else 0,
                "stage2": len(split.stage2),
                "test": len(split.test),
                "freeze_backbone": split.freeze_backbone,
            })
    return pd.DataFrame(rows)


# ==========================================================================
# EXPERIMENT CONSTANTS
# ==========================================================================
# The winning configurations, as selected by the sweeps. They live here as
# named constants rather than as literals inside a notebook so that every
# notebook that reports a number is demonstrably reporting the same setup.

#: Best handcrafted configuration (selected on the training folds).
HANDCRAFTED_WINNER = FeatureConfig(
    normalisation="per_speaker",
    n_mfcc=20,
    extras=("chroma", "spectral_contrast", "spectral_centroid", "zcr"),
)

#: Best wav2vec2 configuration: mid-depth layer, mean pooling, CMVN per speaker.
EMBEDDING_WINNER = EmbeddingConfig(
    layer=6, pooling="mean", normalisation="per_speaker"
)


def load_cached_embeddings(
    config: EmbeddingConfig = EMBEDDING_WINNER,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the cached per-layer tensor and the merged index it aligns to."""
    index = load_index()
    tensor, frame = extract_embeddings(config, index, use_cache=True, progress=False)
    return tensor, frame


def ravdess_view(
    tensor: np.ndarray, index: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Restrict the cached tensor to RAVDESS rows.

    One extraction pass covers the whole merged corpus; the RAVDESS-test
    architecture needs only its RAVDESS subset, and slicing is free.
    """
    mask = (index["source"].astype(str) == "RAVDESS").to_numpy()
    return np.asarray(tensor[mask]), index.loc[mask].reset_index(drop=True)


def build_matrix(
    tensor: np.ndarray,
    index: pd.DataFrame,
    config: EmbeddingConfig,
    layer_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice the cached tensor into a feature frame, ready for a classifier.

    ``per_speaker`` normalisation is applied here because it is not a
    fold-level fit at all -- it uses only each speaker's own audio and no
    labels; see `per_speaker_normalise` for the full argument. ``global``
    normalisation IS a train-fold fit and is therefore applied inside the fold
    loop, never here.
    """
    matrix = select(tensor, config, layer_weights=layer_weights)
    columns = feature_names(matrix.shape[1], config.pooling)

    if config.normalisation == "per_speaker":
        matrix = per_speaker_normalise(matrix, index["speaker_id"].to_numpy())

    frame = pd.DataFrame(matrix, index=index["filename"], columns=columns)
    target = pd.Series(
        index["emotion_label"].to_numpy(),
        index=index["filename"],
        name="emotion_label",
    )
    return frame, target


# ==========================================================================
# IMPORT COMPATIBILITY
# ==========================================================================
# During development this library was a package of one module per concern
# (ser.metadata, ser.splits, ser.features, ...). The submission file budget
# does not allow twenty files, so those modules are concatenated into this
# one. `install()` re-creates the original import paths as aliases onto this
# module, so the notebooks' `from ser.splits import get_split` keeps working
# and nothing in them had to be rewritten to fit the packaging.
#
# This is packaging only. No behaviour differs between the two layouts.

#: Original module path -> any names that were renamed on the way in, so a
#: caller using the old name still reaches the same function object.
_LAYOUT: dict[str, dict[str, str]] = {
    "ser": {},
    "ser.metadata": {},
    "ser.preprocess": {},
    "ser.splits": {},
    "ser.features": {"extract": "extract_handcrafted"},
    "ser.embeddings": {"extract": "extract_embeddings"},
    "ser.evaluate": {},
    "ser.stats": {},
    "ser.models": {},
    "ser.models.classical": {
        "build": "build_classifier",
        "tune": "tune_classifier",
        "default_hyperparams": "classifier_params",
    },
    "ser.corpora": {"summarise": "summarise_corpora"},
    "ser.splits_merged": {},
    "ser.arms": {"summarise": "summarise_arms"},
    "ser.run_phase5": {"WINNER": "HANDCRAFTED_WINNER", "prepare": "prepare_handcrafted"},
    "ser.run_wav2vec2": {"load_cached": "load_cached_embeddings"},
    "ser.run_wav2vec2_transfer": {"WINNER": "EMBEDDING_WINNER"},
}


def install() -> None:
    """Register the original module paths as aliases onto this module.

    Call once, before any ``from ser... import ...``. Idempotent.
    """
    import sys
    import types

    me = sys.modules[__name__]

    for path, renames in _LAYOUT.items():
        module = sys.modules.get(path)
        if module is None or getattr(module, "_ser_lib_alias", False) is False:
            module = types.ModuleType(path)
            module._ser_lib_alias = True
            module.__doc__ = f"Alias of {__name__} (originally {path})."
            sys.modules[path] = module

        for name, value in vars(me).items():
            if not name.startswith("_"):
                setattr(module, name, value)
        for old, new in renames.items():
            if hasattr(me, new):
                setattr(module, old, getattr(me, new))

    # make `ser.splits` reachable as an attribute of `ser`, not only via import
    for path in _LAYOUT:
        if "." in path:
            parent, _, leaf = path.rpartition(".")
            setattr(sys.modules[parent], leaf, sys.modules[path])
