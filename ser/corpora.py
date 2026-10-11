"""Unified index over RAVDESS and the three external emotion corpora.

Origin
------
This module began as a teammate's ``speech_emotion_dataset_builder.ipynb``.
The fetchers and the external-corpus parsers are his work and are kept close
to verbatim -- they solve three real problems well (see below). What changed
is everything that touched *this* repo's contracts:

* RAVDESS is parsed by `ser.metadata.load`, not re-implemented here. That
  parser is strict, tested, and already the single source of truth for which
  clips exist.
* Audio is never opened here. Nothing in this module calls ``librosa.load``;
  `ser.preprocess.load_audio` remains the one entry point, and it is the
  caller's job to use it.
* Column names are this project's (``emotion_label``, ``gender``,
  ``intensity_label``), because `ser.evaluate.SLICE_COLUMNS` reads exactly
  those three to produce the required slices. His names survive as aliases.

Three fetching problems his notebook solved, kept as-is
-------------------------------------------------------
1. **CREMA-D wavs are Git-LFS objects.** ``raw.githubusercontent.com`` serves
   a ~130-byte pointer file, not audio. The bytes come from
   ``media.githubusercontent.com/media/...``, which resolves LFS over plain
   HTTP -- so no ``git-lfs`` install is needed. Every response is checked for
   the ``RIFF``/``WAVE`` magic so a pointer or an HTML error page can never be
   written out as a ``.wav``.
2. **TESS lives on Borealis Dataverse**, which serves batched zips by datafile
   id rather than individual files.
3. **SAVEE is not freely redistributable.** Surrey requires registration, so
   there is no unauthenticated link; ``kagglehub`` or a manual drop-in.

All three fetchers are idempotent and resumable: files already on disk are
skipped, so an interrupted download is repaired by re-running.

The eight labels are frozen
---------------------------
They are RAVDESS's, taken from `ser.metadata.EMOTIONS`. External corpora only
ever contribute rows whose emotion is already one of the eight. An unmapped
code raises rather than being dropped or guessed, because a silently discarded
clip is a change to the dataset that nobody would see.

Two facts to carry into any model trained on this index
--------------------------------------------------------
* **``calm`` exists only in RAVDESS.** In the full merge it is ~1.6% of clips
  and 100% of them carry RAVDESS recording conditions, so a model can reach it
  by detecting the corpus rather than the emotion.
* **``surprised`` is absent from CREMA-D**, and TESS records it as *pleasant*
  surprise (``ps``) where RAVDESS and SAVEE leave it unvalenced. That is the
  standard mapping in the literature but it is a real semantic stretch.

Both are measured in ``docs/dataset_merge.md`` rather than assumed away.
"""

from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ser.metadata import EMOTIONS as RAVDESS_EMOTIONS
from ser.metadata import load as load_ravdess_metadata

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


def summarise(index: pd.DataFrame) -> pd.DataFrame:
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
