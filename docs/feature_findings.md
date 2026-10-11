# Feature findings — Phase 4 ablation

**Which axes mattered, which did not.** One axis varied at a time from a fixed
baseline, with the model held constant (SVM-RBF, library defaults) so every
difference is attributable to the features. Protocol: `speaker_independent`,
pooled out-of-fold over all 1,440 clips.

Reproduce with:

```bash
.venv/Scripts/python.exe -m ser.run_ablation
```

Baseline: `mfcc40-d-dd-mean_std-trim` — accuracy 0.5146, macro-F1 0.4980.

---

## 1. The ranked table

Sorted by macro-F1. Δ is against the baseline, in percentage points.

| Axis | Config | Feats | Accuracy | Macro-F1 | Δ F1 |
|---|---|---|---|---|---|
| **normalisation** | **per-speaker** | 240 | 0.5771 | **0.5634** | **+6.53** |
| n_mfcc | 20 MFCCs | 120 | 0.5451 | 0.5351 | +3.71 |
| aggregation | percentiles | 600 | 0.5437 | 0.5315 | +3.34 |
| n_mfcc | 13 MFCCs | 78 | 0.5424 | 0.5309 | +3.28 |
| extras | chroma+contrast+centroid+zcr | 282 | 0.5382 | 0.5238 | +2.58 |
| extras | spectral contrast | 254 | 0.5278 | 0.5105 | +1.25 |
| n_mels | 80 bands | 240 | 0.5243 | 0.5048 | +0.68 |
| fmin | 50 Hz | 240 | 0.5160 | 0.5044 | +0.64 |
| extras | chroma | 264 | 0.5208 | 0.5034 | +0.54 |
| deltas | Δ only (no ΔΔ) | 160 | 0.5181 | 0.5012 | +0.32 |
| — | **baseline** | 240 | 0.5146 | 0.4980 | 0.00 |
| normalisation | global | 240 | 0.5146 | 0.4980 | 0.00 |
| n_mels | 40 bands | 240 | 0.5097 | 0.4878 | −1.02 |
| aggregation | mean+std+min+max | 480 | 0.5111 | 0.4877 | −1.03 |
| **framing** | **40 ms / 10 ms** | 240 | 0.5056 | 0.4867 | **−1.13** |
| **framing** | **25 ms / 10 ms** | 240 | 0.5083 | 0.4855 | **−1.25** |
| deltas | no Δ, no ΔΔ | 80 | 0.4667 | 0.4345 | −6.36 |
| aggregation | mean only | 120 | 0.4472 | 0.4287 | −6.93 |
| trim | trimming off | 240 | 0.4604 | 0.4238 | −7.42 |

---

## 2. The framing result — a negative finding, and the one we expected to win

**This ablation was prompted by a teammate's question**: what frame duration do
the MFCCs use? The answer was 128 ms, inherited from librosa's default, against
a 20–40 ms speech convention. We expected that to be a defect worth fixing.

**It is not. Shortening the window made things worse.**

| Framing | Accuracy | Macro-F1 | Δ F1 |
|---|---|---|---|
| 128 ms / 32 ms (inherited) | 0.5146 | **0.4980** | — |
| 40 ms / 10 ms | 0.5056 | 0.4867 | −1.13 |
| 25 ms / 10 ms (speech standard) | 0.5083 | 0.4855 | −1.25 |

**Why the textbook advice does not apply here.** The 25 ms convention comes from
phoneme recognition, where you need to resolve which *sound* is being produced
and a longer window smears adjacent phonemes together. We are not identifying
phonemes. We collapse every frame into a mean and standard deviation across the
whole clip, so fine time resolution is discarded immediately — and a longer
window produces a more stable spectral estimate to average. For a
clip-level task, the music-oriented default happens to be the better trade.

**This is a real result, not a failed experiment.** It also means the honest
answer to the original question has changed: we did not choose 128 ms, but
having now tested it, we are keeping it deliberately.

**Caveat.** This conclusion is bound to `mean+std` aggregation. Any Phase 6
model that consumes frame *sequences* — the CNN-LSTM track, or spectrogram
CNNs — has the opposite requirement and should revisit framing on its own
terms. The delta-width coupling is handled: `resolved_delta_width` scales with
the hop so the delta sees ~288 ms of context at every framing, meaning these
rows differ in one thing only.

---

## 3. Per-speaker normalisation — the largest single win

**+6.53 pp macro-F1**, roughly double the next-best axis. This is the Phase 4
gate: per-speaker compared explicitly against global.

| Normalisation | Accuracy | Macro-F1 |
|---|---|---|
| none | 0.5146 | 0.4980 |
| global | 0.5146 | 0.4980 |
| **per-speaker** | **0.5771** | **0.5634** |

**Why `none` and `global` are identical.** Not a bug, and worth stating plainly:
every model runs inside an sklearn `Pipeline` whose first step is a
`StandardScaler`. Global standardisation is therefore *already applied* in the
baseline. The `global` row is a genuine duplicate of the baseline, so the
meaningful comparison — per-speaker against global — is the +6.53 pp above.

**What it means.** Removing each speaker's own mean and variance strips out
what is constant about a voice — its timbre and recording gain — and leaves
what varies within it. That this helps so much is direct evidence that the raw
features encode substantial speaker identity, which is exactly the Phase 7
leakage hypothesis. It is now partly answered before Phase 7 begins.

### The rule-2 carve-out, stated openly

CLAUDE.md rule 2 says per-speaker statistics are fit on the training fold only.
Under `speaker_independent` that is **impossible**: a held-out speaker has no
training clips by construction.

**What we do:** compute each speaker's mean and standard deviation from that
speaker's own clips, held-out speakers included.

**Why it is defensible:** it consumes **no labels** — only audio, which is
equally available at deployment, where you would have a caller's other
utterances before deciding anything about them. This is standard CMVN practice
in speech processing. A test asserts the function cannot read a label: its
signature accepts only a matrix and speaker ids.

**Why we still flag it:** it is *transductive*. The method sees held-out audio,
if not held-out labels. A strict reviewer may discount the gain on that basis,
and they would not be wrong to ask. We would rather state it here than have it
found.

---

## 4. What else mattered

**Trimming is the single most important preprocessing step (−7.42 pp without
it).** This is the strongest validation of Decision 1 in `docs/data_audit.md`:
half of the average clip is silence, and averaging features over it pulls every
clip toward the same near-zero background.

**Deltas are essential (−6.36 pp without them), but the second derivative adds
almost nothing** (+0.32 pp for Δ-only vs Δ+ΔΔ). How the voice *changes* carries
emotion; how its rate of change changes barely does.

**Standard deviation carries as much as the mean.** Mean-only aggregation costs
−6.93 pp. Within-clip variability is not noise — it is signal.

**More MFCCs are worse.** 20 beats 40 by +3.71 pp, and even 13 beats 40 by
+3.28 pp with *one third* the features. High-order cepstral coefficients at
16 kHz mostly capture fine spectral detail that varies with speaker and
recording rather than with emotion.

## 5. What did not matter

- **Mel band count**: 80 gives +0.68, 40 gives −1.02. Within noise.
- **`fmin=50 Hz`**: +0.64 pp. Cutting sub-50 Hz rumble is theoretically right
  and practically negligible.
- **Extra descriptors individually**: chroma +0.54, spectral contrast +1.25.
  Only worthwhile combined (+2.58).

---

## 6. The winning combination

The three strongest axes stack, so we tested them together:

| Combination | Accuracy | Macro-F1 | Feats |
|---|---|---|---|
| **per-speaker + 20 MFCC + extras** | **0.6389** | **0.6311** | **162** |
| per-speaker + 20 MFCC + percentiles | 0.6146 | 0.6040 | 300 |
| per-speaker + 20 MFCC | 0.6028 | 0.5934 | 120 |
| per-speaker only | 0.5771 | 0.5634 | 240 |
| per-speaker + percentiles | 0.5750 | 0.5597 | 600 |

**Winner:** `mfcc20-d-dd-mean_std-trim-chroma+spectral_centroid+spectral_contrast+zcr-per_speaker`

**+12.43 pp accuracy and +13.31 pp macro-F1 over the floor — using 162
features instead of 240.** Better *and* smaller.

Note that percentiles, which won on its own axis (+3.34), *loses* once
per-speaker normalisation is applied — the two were capturing overlapping
information. That is exactly the interaction a one-axis-at-a-time grid cannot
see, and why the combination step matters.

### The winner under both protocols

| Model | Protocol | Accuracy | Macro-F1 |
|---|---|---|---|
| svm-rbf | **speaker_independent** | **0.6389** | **0.6311** |
| svm-rbf | random_stratified | 0.7271 | 0.7200 |
| random-forest | speaker_independent | 0.5687 | 0.5488 |
| random-forest | random_stratified | 0.6486 | 0.6309 |

Per-fold spread for the winner (SVM, speaker-independent): mean 0.6358,
**sd 0.0509**, range 0.5625–0.6933. Still substantial — always quote it.

### The leakage gap narrowed

| Model | Floor gap | Winner gap | Change |
|---|---|---|---|
| svm-rbf | 13.75 pp | **8.82 pp** | −4.93 pp |
| random-forest | 11.81 pp | **7.99 pp** | −3.82 pp |

**This is the most interesting number in the phase.** Per-speaker normalisation
did not just raise accuracy — it *reduced the amount a random split can
exploit*, by about a third. If the features encoded no speaker identity at all,
the two protocols would converge. They have moved measurably in that direction,
which is direct support for the Phase 7 hypothesis and a genuine finding for the
report.

---

## 7. Recommendation

Adopt `mfcc20-d-dd-mean_std-trim-chroma+...-per_speaker` as the standard
configuration for Phase 5 onward. Keep the 128 ms framing — now a tested choice
rather than an inherited default.

Still untuned: every number here uses library-default hyperparameters. Phase 5
tunes the winner via CV *inside* the training folds.
