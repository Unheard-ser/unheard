# wav2vec2: frozen embeddings clear the floor by 16 points

**Headline: 0.7993 accuracy / 0.7944 macro-F1**, pooled out-of-fold on the
frozen RAVDESS `speaker_independent` folds — **+16.33 pp macro-F1** over the
handcrafted floor of 0.6311, and comfortably past the 0.6333 bar Phase 6d set
for a deep track to justify its complexity.

Winning configuration: `facebook/wav2vec2-base`, **hidden layer 6**, mean
pooling over frames, **per-speaker normalisation**, logistic-regression head.

Everything is frozen — no fine-tuning. One forward pass over 12,162 clips
(60 min, 295 ms/clip on 16 CPU threads) caches the pooled statistics of all 13
hidden states, and the entire four-axis sweep below is slices of that 971 MB
file. No axis costs a second forward pass.

## 1. The sweep

One axis at a time, in the Phase 4 style, so each result is attributable.

### Axis 1 — layer: the middle of the network, decisively

| Layer | Accuracy | Macro-F1 | vs winner |
|---|---|---|---|
| 0 (CNN feature projection) | 0.5097 | 0.4994 | −29.50 pp |
| 3 | 0.6549 | 0.6417 | −15.26 pp |
| **6** | **0.6882** | **0.6758** | — |
| 9 | 0.6431 | 0.6331 | −16.13 pp |
| 12 (last) | 0.5083 | 0.4999 | −29.45 pp |

A clean inverted U. **Layer 6 beats the final layer by 17.6 pp**, and the final
layer is barely better than the raw convolutional output.

This is the single most useful practical finding in the phase. The obvious
thing to do with a pretrained encoder — take `last_hidden_state`, which is what
every quickstart shows — would have thrown away 17.6 points. wav2vec2's
pretraining objective rewards predicting masked *phonetic* content, so its
upper layers specialise toward what was said. Emotion lives in the middle,
where the representation still carries how it was said.

#### The complete curve

Five points locate a shape, not a maximum, and the cache holds all thirteen
layers, so the rest cost nothing (`ser/run_wav2vec2_probes.py`). Macro-F1 by
depth, under both normalisations — figure `18_wav2vec2_layer_curve.png`:

| Layer | global | per-speaker |
|---|---|---|
| 0 | 0.4994 | 0.6173 |
| 1 | 0.5645 | 0.6731 |
| 2 | 0.6096 | 0.7017 |
| 3 | 0.6417 | 0.7503 |
| 4 | 0.6569 | 0.7466 |
| 5 | 0.6628 | 0.7703 |
| **6** | **0.6758** | **0.7856** |
| 7 | 0.6751 | 0.7740 |
| 8 | 0.6559 | 0.7603 |
| 9 | 0.6331 | 0.7453 |
| 10 | 0.6003 | 0.7157 |
| 11 | 0.5059 | 0.6212 |
| 12 | 0.4999 | 0.5942 |

**Layer 6 is the true maximum under both normalisations**, so the five-point
probe found the right answer rather than getting lucky between samples. Layer 7
is effectively tied; the peak is a plateau over 5-7 rather than a spike.

The two curves are near-parallel, which is worth noting: per-speaker
normalisation adds a roughly constant 10-12 pp at every depth rather than
interacting with the layer choice. The two axes are independent, which is why
sweeping them one at a time was legitimate here.

Note also where the curves cross the handcrafted floor. Under global
normalisation only layers 3-9 clear it. Under per-speaker normalisation every
layer except 11 and 12 does — **the final two layers of wav2vec2 are worse for
emotion than MFCCs**, even with the best normalisation applied.

### Axis 2 — pooling: the mean is enough

| Pooling | Macro-F1 | Dim |
|---|---|---|
| **mean** | **0.6758** | 768 |
| mean + std | 0.6655 | 1536 |
| std | 0.6280 | 768 |

Concatenating the standard deviation doubles the dimensionality and loses 1.0
pp. Within-clip variation does carry signal on its own (std alone reaches
0.6280, close to the entire handcrafted pipeline) but adds nothing the mean
does not already have.

### Axis 3 — normalisation: the largest single effect

| Normalisation | Macro-F1 | vs none |
|---|---|---|
| none (raw embeddings) | 0.6255 | — |
| global (train-fold StandardScaler) | 0.6758 | +5.03 pp |
| **per-speaker** | **0.7856** | **+16.01 pp** |

This answers the open question the phase was written around. Per-speaker
z-scoring bought +6.53 pp on handcrafted features; on learned features it buys
**+16.01 pp** — more than twice as much, and more than any other axis here.

A self-supervised encoder has *not* factored the speaker out. If anything it
encodes identity more strongly than MFCCs do, which is consistent with wav2vec2
being a strong speaker-verification backbone in the literature. The Phase 7
actor-identity probe should be re-run on these embeddings; the prediction is
that it scores higher than the 92.7% handcrafted figure, not lower.

Note also that this is the project's one stated leakage exception (CLAUDE.md
rule 2) doing very heavy lifting. It is transductive — it sees held-out audio,
though never held-out labels. The size of the gain makes that caveat more
important to state, not less, and it is restated in §4.

### Axis 4 — head: the features are doing the work

| Head | Accuracy | Macro-F1 |
|---|---|---|
| **logistic-regression** | **0.7993** | **0.7944** |
| svm-rbf | 0.7896 | 0.7856 |
| xgboost | 0.6688 | 0.6606 |
| random-forest | 0.6681 | 0.6560 |
| gaussian-nb | 0.6111 | 0.6078 |
| knn | 0.6028 | 0.5812 |
| adaboost | 0.4799 | 0.4746 |

A **linear** classifier on frozen embeddings is the best model in this project.
That is the point: layer-6 wav2vec2 features are close to linearly separable by
emotion, which no MFCC representation here ever was.

## 2. Significance

With 24 test speakers, point estimates need intervals. All of these are cluster
bootstraps resampling **speakers, not clips** (see `ser/stats.py`).

**Winner: 0.7944 macro-F1, 95% CI [0.7479, 0.8355].** The handcrafted floor of
0.6311 sits far outside that interval.

Of the 17 other cells in the sweep, **15 are significantly worse than the
winner** and 2 are not distinguishable from it:

| Comparison | Δ | 95% CI | p | Verdict |
|---|---|---|---|---|
| logistic-regression vs svm-rbf | +0.88 pp | [−1.46, +3.10] | 0.442 | **not distinguishable** |
| vs layer 6 + global norm | +11.86 pp | [+7.99, +15.61] | <0.001 | significant |
| vs layer 12 (last) | +29.45 pp | [+24.69, +34.11] | <0.001 | significant |
| vs layer 0 | +29.50 pp | [+25.19, +33.63] | <0.001 | significant |

**Do not report logistic-regression as beating SVM-RBF.** The 0.88 pp gap is
inside the noise. The honest statement is that a linear head and an RBF head
are indistinguishable here, and the linear one is preferred for being simpler
and cheaper, not for being better.

## 3. The actor-identity probe: the prediction was right, and it is worse than expected

Axis 3 predicted that wav2vec2 encodes speaker identity *more* strongly than
MFCCs, because per-speaker normalisation was worth more than twice as much on
it. The Phase 7 probe, re-run unchanged on these embeddings — same estimator,
same stratified 5-fold, same 1/24 chance baseline — confirms it:

| Features | Actor-ID accuracy | × chance |
|---|---|---|
| handcrafted, raw | 0.927 | 22.3× |
| **wav2vec2 L6, raw** | **0.949** | **22.8×** |
| handcrafted, per-speaker normalised | 0.212 | 5.1× |
| **wav2vec2 L6, per-speaker normalised** | **0.349** | **8.4×** |

Figure `19_actor_identity_handcrafted_vs_wav2vec2.png`.

**The representation that is best at emotion is also best at identifying the
speaker.** wav2vec2 recovers the actor from a 1.5-second average with 94.9%
accuracy, above MFCCs' 92.7%, and after per-speaker normalisation it retains
*more* residual identity than MFCCs do (34.9% vs 21.2%, both far above the 4.2%
chance rate).

This sharpens Phase 7's headline rather than contradicting it. The project's
finding was that these features encode who is speaking, and that per-speaker
normalisation reduces but does not remove it. Switching to a large
self-supervised encoder does not solve that problem — it makes it slightly
worse, while simultaneously improving emotion recognition by 16 points. Speaker
identity and emotional prosody are entangled in the signal, and better features
capture more of both.

Two consequences worth stating in the report:

- **The +16.33 pp emotion gain and the higher speaker leakage are the same
  phenomenon.** Anyone deploying this in a contact centre inherits a model that
  can identify the caller, which is a privacy property the business case has to
  acknowledge, not a footnote.
- **It motivates Phase 14's adversarial track directly.** If per-speaker
  normalisation leaves 8.4× chance identifiability, a gradient-reversal head
  that explicitly penalises actor predictability is the obvious next attempt,
  and there is now a clear before/after number to move.

## 4. Against the rest of the project

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| **wav2vec2 L6 + per-speaker + linear** | **0.7993** | **0.7944** |
| Handcrafted MFCC-20 + extras, SVM-RBF tuned | 0.6389 | 0.6333 |
| Handcrafted, untuned (the floor) | 0.6389 | 0.6311 |
| Handcrafted + all external corpora pooled | 0.5236 | 0.5200 |

For scale: **Phase 4's entire feature-engineering programme moved the number
+13.31 pp. Swapping to a frozen pretrained encoder moved it +16.33 pp**, using a
simpler classifier and no feature design at all.

## 5. What this does not claim

- **No fine-tuning was done.** These are frozen features plus a linear head.
  Fine-tuning the encoder is documented as future work; it is not feasible on
  16 CPU threads.
- **Per-speaker normalisation is transductive.** It computes a held-out
  speaker's statistics from that speaker's own clips. It consumes no labels and
  is standard CMVN practice, but it does see held-out audio, and it accounts for
  +16.01 pp of this result. A reviewer entitled to discount it would put the
  number at 0.6758, which still clears the floor by 4.47 pp.
- **One encoder, one corpus.** WavLM and HuBERT are untested here (Phase 14),
  and everything is RAVDESS-only test data under Architecture A.
- **Layer 6 is confirmed as the maximum over all thirteen layers** (§1), so this
  caveat from the first draft no longer applies.

## 6. What follows

1. **Phase 14**: learnable layer-weighted sum, speaker-adversarial training,
   prosody fusion, WavLM/HuBERT.
2. **Phase 13b** asks whether the merged corpus helps at this feature
   resolution — the hypothesis `docs/merged_baseline_findings.md` left open.

All runs are in `results/results.csv` under notes beginning `phase-13`.
