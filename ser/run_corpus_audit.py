"""Phase 9.4 driver: audit the merged corpus, write ``docs/dataset_merge.md``.

This is the external-corpus counterpart to `ser.audit`, and it asks the same
question: *is the data what we think it is?* Nothing here is a modelling
decision; every number is a property of the files on disk.

Five checks earn their place because each catches a failure that would
otherwise surface much later as an inexplicable result:

* **Counts against published sizes.** A partially-downloaded corpus looks
  exactly like a smaller corpus until someone compares totals.
* **Content hashes across corpora.** TESS and RAVDESS both say "say the word
  ...", and public mirrors of these datasets repackage each other. A clip
  present in two corpora would be counted twice and could straddle a
  train/test boundary.
* **Native sample rate per corpus.** The four corpora were recorded at four
  different rates. All are resampled to 16 kHz by `ser.preprocess.load_audio`,
  but CREMA-D is *already* 16 kHz and so passes through untouched while the
  others are downsampled. Anti-alias filtering leaves a spectral signature, so
  "which corpus" may be audible to a model as a band-limit artefact rather
  than as anything to do with emotion.
* **Internal uniformity of each corpus.** A corpus is supposed to have one
  native rate. A file that does not is a repackaging error or a different
  source recording, and either way it should be named rather than quietly
  absorbed by the resampler.
* **Decode check.** Every file is opened; zero-length and undecodable files
  are listed rather than discovered mid-training.

Scope note: this measures file provenance across *all* clips, not emotion
structure. That is the same licence `ser.audit` takes in Phase 1 and it is not
EDA -- signal analysis that informs modelling choices is restricted to
training folds in `ser.run_merged_eda`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ser import SEED, set_seeds
from ser import corpora
from ser.preprocess import TARGET_SR, load_audio

DOC_PATH: Path = corpora.PROJECT_DIR / "docs" / "dataset_merge.md"

#: Clips per corpus used for the spectral-rolloff probe. Decoding every file
#: would add minutes for a number that is stable well before then.
ROLLOFF_SAMPLE = 120


def probe_headers(index: pd.DataFrame) -> pd.DataFrame:
    """Read every file's header without decoding it.

    Args:
        index: A frame in the shape `ser.corpora.build_index` returns.

    Returns:
        ``filename``, ``source``, ``native_sr``, ``channels``, ``subtype``,
        ``duration_s``, ``frames``, one row per clip. ``native_sr`` is ``-1``
        and ``subtype`` ``"UNREADABLE"`` where the header could not be read.
    """
    rows = []
    for source, path in zip(index["source"], index["filepath"], strict=True):
        try:
            info = sf.info(str(path))
            rows.append(
                {
                    "filename": Path(path).name,
                    "source": source,
                    "native_sr": int(info.samplerate),
                    "channels": int(info.channels),
                    "subtype": str(info.subtype),
                    "duration_s": float(info.duration),
                    "frames": int(info.frames),
                }
            )
        except Exception as exc:  # noqa: BLE001 - a bad file is a finding
            rows.append(
                {
                    "filename": Path(path).name,
                    "source": source,
                    "native_sr": -1,
                    "channels": -1,
                    "subtype": f"UNREADABLE: {exc!r}"[:60],
                    "duration_s": 0.0,
                    "frames": 0,
                }
            )
    return pd.DataFrame(rows)


def summarise_headers(headers: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-file probe into one row per corpus."""
    return (
        headers.groupby("source", observed=True)
        .agg(
            clips=("filename", "size"),
            native_sr=("native_sr", lambda s: sorted(set(int(x) for x in s))),
            channels=("channels", lambda s: sorted(set(int(x) for x in s))),
            subtypes=("subtype", lambda s: sorted(set(s))[:3]),
            dur_min=("duration_s", "min"),
            dur_med=("duration_s", "median"),
            dur_max=("duration_s", "max"),
            total_hours=("duration_s", lambda s: round(s.sum() / 3600, 2)),
        )
        .reset_index()
    )


def find_duplicates(index: pd.DataFrame) -> pd.DataFrame:
    """Hash every clip's bytes and report any content shared across corpora.

    Public mirrors of these datasets repackage one another, so a clip
    appearing in two corpora is a real possibility rather than a paranoid
    check. A duplicate would be counted twice in the class balance and could
    land on both sides of a grouped split.

    Args:
        index: A frame in the shape `ser.corpora.build_index` returns.

    Returns:
        One row per duplicated hash, with the corpora and filenames involved.
        Empty when every clip is unique.
    """
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, path in zip(index["source"], index["filepath"], strict=True):
        try:
            digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
        except OSError:
            continue
        by_hash[digest].append((str(source), Path(path).name))

    rows = []
    for digest, members in by_hash.items():
        if len(members) > 1:
            rows.append(
                {
                    "md5": digest[:12],
                    "n": len(members),
                    "sources": ", ".join(sorted({m[0] for m in members})),
                    "cross_corpus": len({m[0] for m in members}) > 1,
                    "files": ", ".join(name for _, name in members[:4]),
                }
            )
    return pd.DataFrame(rows)


def probe_rolloff(index: pd.DataFrame, n: int = ROLLOFF_SAMPLE) -> pd.DataFrame:
    """Measure post-resampling spectral rolloff per corpus **and gender**.

    The point is not the absolute value but the *contrast between corpora*.
    Every clip reaches the model at 16 kHz via `load_audio`, so if one corpus
    carries systematically different high-frequency energy from another, a
    model can identify the corpus from band-limiting alone -- a shortcut with
    nothing to do with emotion.

    Gender is a necessary control, not decoration. TESS is entirely female and
    SAVEE entirely male, so a raw per-corpus comparison confounds the recording
    chain with the voices in it: male speech is generally lower in spectral
    centroid, and without the split a gender effect would be misread as a
    corpus effect. RAVDESS and CREMA-D carry both genders and so provide the
    within-corpus contrast that separates the two.

    Args:
        index: A frame in the shape `ser.corpora.build_index` returns.
        n: Clips sampled per (corpus, gender) cell.

    Returns:
        Mean and sd of the 95th-percentile rolloff frequency in Hz, per
        (source, gender) cell.
    """
    import librosa

    rng = np.random.default_rng(SEED)
    rows = []
    grouped = index.groupby([index["source"].astype(str), "gender"], observed=True)
    for (source, gender), group in grouped:
        paths = group["filepath"].to_numpy()
        take = rng.choice(paths, size=min(n, len(paths)), replace=False)
        values = []
        for path in take:
            try:
                signal, sr = load_audio(path)
            except Exception:  # noqa: BLE001 - already reported by the header probe
                continue
            if signal.size < 2048:
                continue
            rolloff = librosa.feature.spectral_rolloff(
                y=signal, sr=sr, roll_percent=0.95, n_fft=2048, hop_length=512
            )
            values.append(float(np.mean(rolloff)))
        if values:
            rows.append(
                {
                    "source": source,
                    "gender": gender,
                    "n_probed": len(values),
                    "rolloff_hz_mean": round(float(np.mean(values)), 1),
                    "rolloff_hz_sd": round(float(np.std(values)), 1),
                }
            )
    return pd.DataFrame(rows)


def sample_rate_anomalies(headers: pd.DataFrame) -> pd.DataFrame:
    """Files whose native rate differs from the rest of their own corpus.

    A corpus is *supposed* to be internally uniform. A file that is not is
    either a repackaging error or a different source recording, and either way
    it is worth naming rather than letting the resampler quietly absorb it.

    Args:
        headers: Per-file frame from `probe_headers`.

    Returns:
        One row per anomalous file, with its corpus's dominant rate for
        comparison. Empty when every corpus is internally uniform.
    """
    rows = []
    for source, group in headers.groupby("source", observed=True):
        usable = group[group["native_sr"] > 0]
        if usable.empty:
            continue
        dominant = int(usable["native_sr"].mode().iloc[0])
        odd = usable[usable["native_sr"] != dominant]
        for record in odd.itertuples():
            rows.append(
                {
                    "source": source,
                    "filename": record.filename,
                    "native_sr": record.native_sr,
                    "corpus_dominant_sr": dominant,
                    "duration_s": round(record.duration_s, 3),
                }
            )
    return pd.DataFrame(rows)


def label_coverage(index: pd.DataFrame) -> pd.DataFrame:
    """Which corpus contributes which of the eight canonical labels."""
    table = pd.crosstab(index["emotion_label"], index["source"])
    table = table.reindex(list(corpora.CANONICAL)).fillna(0).astype(int)
    table["TOTAL"] = table.sum(axis=1)
    table["share"] = (table["TOTAL"] / table["TOTAL"].sum()).map("{:.1%}".format)
    table["corpora"] = [
        ", ".join(c for c in corpora.SOURCES if c in table.columns and table.loc[e, c] > 0)
        for e in table.index
    ]
    return table.reset_index()


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def main() -> None:  # pragma: no cover - driver, not library code
    set_seeds()
    print("building index (validation deferred so a partial merge still reports)")
    index = corpora.build_index(validate=False, write=True)
    print(f"  {len(index):,} clips across {index['source'].nunique()} corpora")

    print("probing headers ...")
    headers = probe_headers(index)
    header_summary = summarise_headers(headers)
    unreadable = headers[headers["native_sr"] < 0]
    empty = headers[(headers["frames"] == 0) & (headers["native_sr"] >= 0)]

    print("hashing for cross-corpus duplicates ...")
    duplicates = find_duplicates(index)

    print("probing spectral rolloff after the 16 kHz resample ...")
    rolloff = probe_rolloff(index)
    anomalies = sample_rate_anomalies(headers)

    coverage = label_coverage(index)
    shape = corpora.summarise(index)

    counts_ok = []
    for source, count in index["source"].value_counts().items():
        if int(count) == 0:
            continue
        expected = corpora.PUBLISHED_COUNTS.get(str(source))
        counts_ok.append(
            {
                "source": source,
                "on_disk": int(count),
                "published": expected,
                "match": "yes" if expected == int(count) else "NO",
            }
        )
    counts_frame = pd.DataFrame(counts_ok)

    cross = duplicates[duplicates["cross_corpus"]] if not duplicates.empty else pd.DataFrame()

    doc = f"""# Merged corpus audit

Generated by `ser/run_corpus_audit.py`. Every number here is a property of the
files on disk, measured rather than taken from documentation.

Audio is read through `ser.preprocess.load_audio`, which downmixes to mono and
resamples to **{TARGET_SR:,} Hz** -- the project rate, and the rate wav2vec2
requires.

## 1. Shape of the merge

{_md_table(shape)}

## 2. Counts against published corpus sizes

{_md_table(counts_frame)}

## 3. Native audio formats

{_md_table(header_summary)}

Unreadable files: **{len(unreadable)}**. Zero-length files: **{len(empty)}**.

## 4. Cross-corpus duplicate check

{len(duplicates)} duplicated content hash(es) found, of which
**{len(cross)}** span more than one corpus.

{_md_table(cross) if len(cross) else "No clip appears in two corpora. Nothing is double-counted."}

## 5. Sample-rate anomalies within a corpus

{_md_table(anomalies) if len(anomalies) else "Every corpus is internally uniform."}

## 6. Resampling signature, controlled for gender

All four corpora arrive at the model at {TARGET_SR:,} Hz, but only CREMA-D was
recorded there; the others are downsampled, and anti-alias filtering leaves a
spectral signature. If the rolloff figures separate cleanly by corpus, then
"which corpus" is partly audible as a band-limit artefact -- a cue with nothing
to do with emotion.

Gender is split out because TESS is entirely female and SAVEE entirely male, so
a raw per-corpus comparison would confound the recording chain with the voices
in it.

{_md_table(rolloff)}

Read the male column across corpora: that is the comparison gender cannot
explain.

## 7. Label coverage and class balance

{_md_table(coverage)}

## 8. What this means for modelling

**`calm` is RAVDESS-only.** Every `calm` clip therefore carries RAVDESS
recording conditions, so a model can reach that class by detecting the corpus.
This is why the headline protocol holds the test set to RAVDESS alone, where
corpus identity is constant and carries no information.

**`surprised` is absent from CREMA-D**, and TESS records it as *pleasant*
surprise (`ps`) where RAVDESS and SAVEE leave it unvalenced. That is the
standard mapping in the literature, but it is a genuine semantic stretch and
the per-class metrics for `surprised` should be read with it in mind.

**Speaker counts are lopsided.** Grouped folds will land unevenly, so per-fold
composition is reported rather than assumed.
"""

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(doc, encoding="utf-8", newline="\n")
    print(f"\nwrote {DOC_PATH}")

    print("\n" + shape.to_string(index=False))
    print("\n" + counts_frame.to_string(index=False))
    print("\n" + header_summary.to_string(index=False))
    print("\n" + rolloff.to_string(index=False))
    print(f"\ncross-corpus duplicates: {len(cross)}")
    print(f"unreadable: {len(unreadable)} | zero-length: {len(empty)}")


if __name__ == "__main__":  # pragma: no cover
    main()
