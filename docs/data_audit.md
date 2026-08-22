# Data audit — RAVDESS speech subset

**Phase 1 deliverable.** Every number below was measured from the files on disk
by `ser/audit.py`, at the native sample rate. Reproduce with:

```bash
.venv/Scripts/python.exe -m ser.audit
```

Audit date 2026-08-22 · 1,440 files · 0 failed to load · `features/audit.parquet` (gitignored, ~60 s to rebuild)

---

## 1. What is clean

| Property | Finding |
|---|---|
| File count | **1,440**, exactly as specified |
| Load integrity | **0** corrupt, **0** zero-length, 0 read errors |
| Sample rate | **48 000 Hz** — uniform, all 1,440 |
| Bit depth / encoding | **PCM_16** — uniform, all 1,440 |
| Clips per actor | **60** for every one of the 24 actors |
| Class balance | 96 neutral, 192 × each of the other seven |
| Strong-intensity neutral | **0 files** — confirms the documented asymmetry |
| Statement / repetition | 720 / 720 both, perfectly balanced |
| Duration | 2.936 – 5.272 s (median 3.670, mean 3.701, sd 0.337) |

No file needs to be dropped or repaired. This is an unusually clean corpus —
which means the interesting findings are about *structure*, not damage.

## 2. What is not

### 2.1 The filename schema is not the canonical one

Files carry **5** hyphen-separated fields, not RAVDESS's 7:
`emotion-intensity-statement-repetition-actor`, e.g. `06-02-01-02-12.wav`.
The constant `modality` (03) and `vocal-channel` (01) fields were stripped when
this copy was repackaged, and actors live in `data/data/PersonNN/`, not
`Actor_NN/`.

All documented counts still hold exactly, so the data is intact RAVDESS — only
the encoding differs. `ser/metadata.py` re-adds the two constant fields as
columns and **rejects** 7-field names outright, so a canonical file dropped in
later fails loudly instead of shifting every field by two positions.

### 2.2 Five files are dual-channel

| File |
|---|
| `Person01/02-01-01-02-01.wav` |
| `Person01/08-01-02-02-01.wav` |
| `Person05/02-01-02-02-05.wav` |
| `Person20/03-01-02-01-20.wav` |
| `Person20/06-01-01-02-20.wav` |

Both channels are **bit-identical** (`max|L−R| = 0.0`) — duplicated mono, not
true stereo. Harmless, but code that assumes a 1-D array would break on 5 of
1,440 files, which is exactly the kind of defect that surfaces as an
inexplicable error 200 clips into a feature extraction run.

### 2.3 Half of the average clip is silence

| Measure | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|
| Total duration (s) | 2.936 | 3.470 | 3.670 | 3.701 | 3.871 | 5.272 |
| Speech only, trimmed (s) | 1.067 | 1.547 | 1.739 | 1.840 | 2.059 | 4.256 |
| **Silence fraction** | 0.004 | 0.479 | **0.530** | 0.506 | 0.562 | **0.646** |

Leading silence is ~0.95 s and trailing ~0.91 s, both remarkably consistent
(sd 0.18 / 0.21) — a fixed recording protocol, not random variation. But the
*fraction* still ranges from 0.4% to 64.6%: **92 files are over 60% silence and
13 are under 20%.**

### 2.4 A ~44 dB loudness range, and one clipped sample

| Measure | min | median | mean | max |
|---|---|---|---|---|
| Peak amplitude | 0.0061 | 0.0993 | 0.1783 | 0.9991 |
| RMS | 0.0007 | 0.0099 | 0.0180 | 0.1522 |

Peak spans **0.006 to 0.999 — about 44 dB**. 124 files peak above 0.5, and 39
above 0.9. **Exactly one sample in one file clips**
(`Person10/03-02-02-01-10.wav`, peak 0.999146) — negligible, no action needed,
but recorded so it is not rediscovered later.

One artifact worth noting: **15 files share the peak value 0.998810 to six
decimal places**, nearly all angry-strong. Identical peaks across different
actors indicate a hard limiter in the recording chain rather than coincidence.
Those clips are ceiling-bound, so their peak amplitude understates how loud
they actually were.

---

## 3. The finding that changes the modelling plan

**Loudness is emotion, not gain.** Mean peak amplitude by emotion × intensity:

| Emotion | normal | strong |
|---|---|---|
| **angry** | 0.185 | **0.750** |
| fearful | 0.107 | 0.430 |
| happy | 0.098 | 0.304 |
| surprised | 0.098 | 0.172 |
| disgust | 0.080 | 0.145 |
| sad | 0.048 | 0.123 |
| neutral | 0.054 | — |
| **calm** | 0.041 | 0.039 |

Angry-strong is **19× louder than calm-strong** (~26 dB). By contrast, mean RMS
across the 24 actors spans only **2.9× (9.2 dB)**.

So the amplitude variation is dominated by *what is being expressed*, not by
*who recorded it* or at what gain. The emotion-driven spread is roughly three
times the actor-driven spread.

This inverts the obvious preprocessing instinct. Per-clip peak normalisation is
the reflexive default for audio pipelines, and here it would **destroy the single
most discriminative cue in the dataset** — that angry is loud and calm is not.
Note also that calm barely changes between normal and strong (0.041 → 0.039),
which is itself a sensible sanity check on the label semantics.

---

## 4. Preprocessing decisions

Each follows from a measured number above, not from convention.

### Decision 1 — Trim leading and trailing silence, with `top_db` as a swept parameter

*Evidence:* mean silence fraction **50.6%**, range 0.4%–64.6% (§2.3).

Half of the average clip is protocol padding. Worse than the mean is the
*spread*: raw duration measures how long the engineer left the recorder running,
not how long the actor spoke. Any duration- or rate-derived feature computed on
untrimmed audio is therefore part signal, part recording artifact. Aggregations
like mean-MFCC are also diluted — averaging over a window that is half silence
pulls every clip's statistics toward the same near-zero background.

Trimming is on. `top_db` is **not** fixed at 30; it enters `ser/features.py` as a
Phase 4 parameter to sweep, because 30 dB is a librosa default rather than a
measured choice, and 13 files sit under 20% silence where an aggressive
threshold would start eating speech.

### Decision 2 — Do NOT peak-normalise per clip. Compare per-speaker feature normalisation instead

*Evidence:* emotion spread ~26 dB vs actor spread 9.2 dB (§3).

This reverses the default. Because loudness carries emotion far more than it
carries recording gain, per-clip peak normalisation would remove real signal to
suppress a smaller nuisance. Keep absolute amplitude; let energy and RMS
features see it.

The per-actor gain component is real but secondary, and it belongs at the
*feature* level where it can be measured rather than assumed: **per-speaker
z-scoring is already a Phase 4 axis** and must be compared head-to-head against
global z-scoring, fit on training folds only. Phase 7's actor-identification
probe is what will tell us how much identity the features still encode.

If a later experiment does want loudness-invariant features, the correct move is
to add a normalised variant as an *ablation arm* — never to bake it into
preprocessing where no one can measure its cost.

### Decision 3 — Resample 48 kHz → 16 kHz at load, and always pass `sr=` explicitly

*Evidence:* uniform 48 kHz across all 1,440 (§1).

Speech energy is essentially all below 8 kHz, so 16 kHz is Nyquist-sufficient and
cuts decode and feature-extraction cost about 3×. It is also the rate
wav2vec2/WavLM/Whisper require in Phase 6, so one choice serves the classical and
pretrained tracks alike and avoids two incompatible feature caches.

**Non-negotiable corollary:** `librosa.load()` silently defaults to `sr=22050`.
Any call that omits `sr=` produces a resampled artifact that matches neither the
source nor our target. Every load in `ser/` passes `sr=` explicitly; `ser/audit.py`
uses `sr=None` precisely so this document describes the true files.

### Decision 4 — Downmix to mono at load; never edit the raw files

*Evidence:* 5 dual-channel files, channels bit-identical (§2.2).

`librosa.load(..., mono=True)` averages identical channels — a mathematical no-op
here — and is idempotent for the other 1,435. One code path, no special-casing,
and `data/` stays pristine and gitignored. Rewriting 5 files on disk would make
every teammate's copy silently differ from the source.

---

## 5. Secondary observations for Phase 2

Not preprocessing decisions, but they shape what EDA should look for.

- **Duration separates emotions even after trimming.** Trimmed speech runs
  disgust 2.11 s and calm 2.01 s at the long end, surprised 1.58 s and neutral
  1.59 s at the short end — a ~34% spread that survives silence removal, so it is
  speaking-rate signal rather than padding.
- **Strong intensity is longer**: 1.97 s vs 1.72 s trimmed. Intensity is
  partly encoded as duration, not only as loudness.
- **Gender barely affects duration** (female 3.75 s, male 3.65 s), so duration
  features should not be a large gender-bias vector. Phase 7's fairness slice
  should still verify this on model outputs.
- **Neutral has both the shortest trimmed duration and among the highest silence
  fractions**, on top of having half the clips. It is the class most likely to
  fail, which is exactly why macro-F1 is reported alongside accuracy.
- The 15 limiter-ceiling files mean peak amplitude is *censored* at the top end.
  Prefer RMS or percentile-based energy features over raw peak.

---

## 6. Gate

Phase 1 requires the audit to name at least three concrete preprocessing
decisions justified by evidence rather than convention. Four are given in §4,
each tied to a measured statistic — and Decision 2 explicitly *contradicts* the
conventional choice on the strength of the measurement, which is the point of
auditing rather than assuming.
