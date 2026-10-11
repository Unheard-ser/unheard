# Leave-one-corpus-out: the headline halves

**0.794 within RAVDESS. 0.380 on a corpus the model has never heard.**

Every other protocol in this project holds out *speakers* recorded in the same
studio, on the same equipment, reading the same two sentences. This one holds
out a whole corpus: train on three, test on the fourth — different people,
different microphones, different room, different prompts. It is the only test
here that resembles deployment.

## Results

Head is `LinearSVC` for both feature sets, so the two columns are comparable.
Macro-F1 is computed over the classes actually present in each test fold (see
§3).

| Held out | n train | n test | Classes | Handcrafted | wav2vec2 L6 |
|---|---|---|---|---|---|
| RAVDESS | 10,722 | 1,440 | 8 | 0.315 | **0.380** |
| CREMA-D | 4,720 | 7,442 | 6 | **0.346** | 0.315 |
| TESS | 9,362 | 2,800 | 7 | **0.425** | 0.395 |
| SAVEE | 11,682 | 480 | 7 | **0.365** | 0.338 |

Chance is 0.125–0.167 depending on the fold's class count, so every cell is
clearly above chance — and every cell is far below usable.

## 1. The headline more than halves

The RAVDESS row is the one to quote, because it is the same test set as the
project's headline:

| Protocol | What is held out | Macro-F1 |
|---|---|---|
| `speaker_independent` | 4–5 RAVDESS actors | **0.794** |
| `corpus_holdout` | all of RAVDESS | **0.380** |

Same 1,440 clips scored either way. The difference is whether the model saw
*other recordings made in the same studio*.

**So more than half of 0.794 is corpus-specific rather than transferable
emotion recognition.** Speaker-independent evaluation within one corpus is a
much weaker guarantee than it appears: holding out the voice still leaves the
microphone, the room, the prompt set and the direction style shared between
training and test.

This is the most consequential measurement in the project, and it reframes
everything above it. It is also a criticism of the standard protocol in this
literature — RAVDESS numbers are almost always reported speaker-independent
within RAVDESS, and this shows what that omits.

## 2. wav2vec2's advantage does not transfer

Within RAVDESS, wav2vec2 beats handcrafted features by 16.3 pp. Across corpora
it wins **one fold of four**:

| Held out | wav2vec2 − handcrafted |
|---|---|
| RAVDESS | **+6.5 pp** |
| CREMA-D | −3.1 pp |
| TESS | −3.0 pp |
| SAVEE | −2.7 pp |

The one win is the fold where RAVDESS is the target — which is the fold most
like the conditions wav2vec2 was tuned on here. On genuinely foreign audio it is
no better than MFCCs, and slightly worse.

That is a real qualification on the project's headline finding. "Frozen
pretrained embeddings beat hand-designed features by 16 points" is true, and
true only within a corpus. Stated without that clause it over-promises.

## 3. A metric detail that would otherwise mislead

CREMA-D contains no `calm` and no `surprised`; TESS and SAVEE contain no `calm`.
Macro-F1 over the project's fixed 8 labels scores those absent classes 0 and
divides by 8, penalising a fold for classes that cannot appear in it — the
CREMA-D fold reads 0.236 that way versus 0.315 over the six classes it actually
has. Both are logged (`macro_f1_all8`, `macro_f1_present`); the table above uses
the honest one.

The same artefact bit the intensity slice in `docs/slice_findings.md`, where it
reversed the conclusion. Worth checking wherever a slice changes the label set.

## 4. Why this makes the merge result coherent

`docs/merged_baseline_findings.md` found that adding external corpora to
training *hurts*, and that the damage scales with volume (r = −0.98). That was
measured but unexplained.

This is the explanation. If a model trained on three corpora only reaches 0.32–0.43
on the fourth, the corpora are close to different tasks as far as the model is
concerned. Pooling them does not add examples of the same problem; it adds
examples of adjacent problems, and the model splits its capacity between them.

The two results are the same fact seen from two directions.

## 5. What to say in the report

- **Headline stays 0.794**, correctly labelled as speaker-independent *within
  RAVDESS*.
- **Add 0.380 as the cross-corpus figure**, and say plainly that it is the
  better estimate of deployment behaviour.
- Qualify the wav2vec2 win as within-corpus.
- Note that the acted-speech caveat compounds with this: 0.681 on
  normal-intensity delivery (`docs/slice_findings.md`), and 0.380 on unseen
  recording conditions. A real contact centre has both at once, and nothing
  here measures that combination — no corpus available to us would.

All runs in `results/results.csv` under `phase-15`.
