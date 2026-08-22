# Meeting Brief — everything explained

## The 30-second version

We built the measuring setup and got the first real numbers.
SVM gets **51.5%** holding out whole speakers, **65.2%** on a random split.
Same model, same code. The random split inflates by ~27%.
**That gap is our finding.**

---

# PART 1 — The words, explained

## What are "features"?

A computer can't read a sound file. A `.wav` is just a huge list of numbers
describing air pressure over time — 48,000 numbers per second. Useless directly.

So you **summarise** each clip into a smaller, meaningful list of numbers. Those
are features. Things like: average pitch, how loud, how fast, what the tone
quality is.

Every model in this project eats features, not audio.

## What is an MFCC?

**Mel-Frequency Cepstral Coefficient.** It's the standard feature for speech.

Plain version: it measures the *shape* of a sound — which frequencies are strong
at each moment. It's basically a fingerprint of voice quality.

Why "mel": human ears don't hear all frequencies equally. We're sensitive to
changes in low pitch, less so in high pitch. The mel scale squashes frequencies
the way a human ear does, so the numbers reflect what a person actually hears.

**What we use:** 40 MFCCs, plus "deltas" (how each one changes over time) and
"delta-deltas" (how fast that change is changing). Averaged over the clip and
combined, each 3-second clip becomes **240 numbers**.

**If asked:** MFCCs are the standard audio feature for speech tasks. Almost every
published paper on this dataset uses them.

## What is a mel-spectrogram?

A picture of the sound. Time along the bottom, frequency up the side, brightness
showing loudness. Same underlying information as MFCCs, kept as an image instead
of a summary. Useful if you want to use image-style models (CNNs).

## What is an SVM?

**Support Vector Machine.** A classic model. It draws a dividing boundary between
classes and positions it as far as possible from the nearest examples on each
side — which makes it robust.

**Why I picked it first:** SVMs are unusually good when you have *many features
and few examples*, which is exactly us — 240 features, 1,440 clips. It also
trains in seconds, so it was the fastest way to check nothing was structurally
broken. Historically it's also the strongest classical performer on speech
emotion tasks.

"RBF kernel" just means the dividing boundary is allowed to curve instead of
being a straight line.

## What is Random Forest?

Hundreds of decision trees, each asking yes/no questions ("is energy above X?"),
each seeing a random subset of features. They vote. Handles curved patterns,
doesn't care about feature scaling.

We ran it alongside SVM. It scored lower (46.7% vs 51.5%).

## What is an RNN?

**Recurrent Neural Network.** Reads a clip moment by moment, carrying a memory of
what came before. Good at sequences — it can notice the voice *rose then cracked*,
not just the average. This is Rishab's track.

## What is a fold?

We split the data into 5 chunks. Each chunk takes a turn being the test set while
the other 4 train. Each chunk is a "fold". Averaging across all 5 gives a more
reliable estimate than one single split.

## Speaker-independent vs random split

**Random split:** shuffle all 1,440 clips, take 20% as the test set. The same
actor's voice ends up in both training and testing.

**Speaker-independent:** hold out entire *actors*. If actor 7 is in the test set,
none of his clips are in training.

**Why it matters:** we only have 24 actors. With a random split, the model can
learn "this is actor 7's voice, and when actor 7 sounds like this he's angry."
That works on actor 7 and fails on a stranger. It looks like high accuracy but
it's recognising people, not emotions.

## What is "leakage"?

Information about the test set sneaking into training, so the score is
flattering and fake. Speaker overlap is one kind. Fitting a scaler on all the
data before splitting is another. Tuning settings against test performance is a
third.

## Accuracy vs Macro-F1

**Accuracy:** % of clips it got right overall.

**Macro-F1:** average performance across all 8 classes, weighting each class
equally regardless of size.

**Why both:** neutral has half the clips of everything else. A model could fail
completely on neutral and still show decent accuracy, because neutral is only 7%
of the data. Macro-F1 exposes that. Ours: accuracy 0.515, macro-F1 0.498 — the
gap is neutral failing.

## What is a baseline?

The simple result everything else must beat to justify itself. If the RNN scores
55%, that's good — it beat 51.5%. If it scores 48%, the complexity didn't pay
off. Without a baseline, a number on its own tells you nothing.

## What is a confusion matrix?

A grid showing what got predicted as what. Rows = true emotion, columns = what
the model guessed. The diagonal is correct answers; everything off-diagonal is a
mistake. It tells you *which* mistakes, not just how many.

---

# PART 2 — What I did, and why

## 1. Data audit

Opened the actual audio and measured it. 48 kHz, ~3.3 seconds, mono.

**Found 5 stereo files** where the other 1,435 are mono. Left unhandled, those 5
get processed differently and silently corrupt results.

**Found we should NOT normalise volume.** Standard preprocessing rescales every
clip to the same loudness so quiet and loud recordings are comparable. We
measured: loudness varies ~26 dB *across emotions*, but only ~9.2 dB *across
speakers*. Angry is loud, sad is quiet — **loudness IS the emotion here.**
Normalising would delete 26 dB of real signal to remove 9.2 dB of noise. Bad
trade. So we don't.

**Why this matters for marks:** "we tested the standard step and found it
destroys signal on this dataset" is evidence-driven, not convention-driven.
That's what the Data Understanding rubric line rewards.

## 2. Froze the splits

Decided which actors go in which test fold, wrote it to `splits/folds.csv`,
committed it. Nobody edits it.

**Why:** if five of us each pick our own split and our own metric, we come back
with five numbers that can't be compared. Three weeks of work, nothing to put in
a table. Frozen folds turn parallel work into one experiment.

Three protocols exist:
- **speaker_independent** — the honest one, our primary
- **random_stratified** — run only to measure the inflation
- **statement_holdout** — train on sentence 1, test on sentence 2 (tests whether
  the model learned tone rather than specific words)

## 3. One scoring function

`evaluate.py`. Everyone calls it, so "accuracy" means the same thing for all of
us. It returns accuracy, macro-F1, per-class scores, a confusion matrix, and
slices by gender/intensity/emotion. Every run writes one row to `results.csv`.

## 4. Extracted features

1,440 clips → 240 numbers each. Cached, so nobody re-extracts.

## 5. Ran the models

SVM and Random Forest, both protocols, library defaults, no tuning.

**Why no tuning:** a baseline is a floor. Tuning before comparing makes the
comparison unfair.

---

# PART 3 — The results

| Model | Setup | Accuracy | Macro-F1 |
|---|---|---|---|
| **SVM** | **speaker held out (honest)** | **51.5%** | 0.498 |
| SVM | random split (inflated) | 65.2% | 0.637 |
| Random Forest | speaker held out | 46.7% | 0.423 |
| Random Forest | random split | 58.5% | 0.554 |

**Gap = 13.75 points, or 27% relative inflation.**

Random guessing = 12.5%. Majority class = 13.3%. So 51.5% is ~4× chance — a real
result, and it sits inside the 40–70% range published speaker-independent work
reports. Nothing looks too good to be true.

**Always quote the spread too.** Speaker-independent varies ±6.1 points across
folds, versus ±3.0 for random. Which five actors you hold out matters a lot.
51% could reasonably have been 44% or 59%.

---

# PART 4 — Four findings

**1. The leakage gap.** 13.75 points, 27% relative. Most published 92% results
use random splits. This measures how much that inflates things. This is our
headline.

**2. Neutral is our worst class** — 23% correct, and it usually gets predicted as
"calm". Why: neutral has 96 clips where everything else has 192, because there's
no such thing as "strongly neutral" so it was only recorded at one intensity.

**3. Errors cluster by energy level, not randomly.** sad→calm (47 clips),
neutral→calm (22), happy→fearful (47). Quiet emotions collapse into each other;
loud ones collapse into each other. That's a pattern, and it justifies trying a
two-stage model later (predict loud/quiet first, then the emotion within).

**4. 7.9 point gap between female and male accuracy** (female better). Real
fairness finding — and it ties straight into the demographic-bias risk already
sitting in our business slides.

**Also:** strong-intensity clips score 17 points better than normal-intensity
ones. Real call-centre audio is normal, not theatrical — so **~44% is the honest
deployment estimate**, not 51.5%. (Caveat: part of that gap is because all the
neutral clips are normal-intensity, and neutral is our worst class. Adjusted, the
gap is ~14 points. Quote the adjusted one.)

---

# PART 5 — Rishab's RNN at 15%

**15% is basically chance.** 8 classes → random guessing is 12.5%. The model has
learned essentially nothing. **That's a bug, not a bad architecture.**

Likely causes:
- labels not lining up with clips after loading
- clips not padded to the same length (raw audio varies in duration; if it's
  silently truncating you'd see exactly this)
- loss not actually decreasing during training — worth checking it's going down
  at all

**He's right that MFCCs beat raw audio.** Raw waveforms are enormous and mostly
irrelevant detail; MFCCs are the compressed, meaningful version.

**One catch to tell him:** my extraction *averages over time* — one row of 240
numbers per clip. That's correct for an SVM and useless for an RNN, which needs
the frames **in order** to have anything to be recurrent about. If he uses the
pooled version he gets a sequence of length 1 and the RNN is pointless. I'll add
a flag for the sequence version.

**Say gently:** the RNN may still lose to the SVM even once fixed. 1,440 clips is
small and RNNs overfit badly at that size. If it comes in at 45%, that's a
finding — "we tested whether complexity paid off at this dataset size and it
didn't" — not a failure of his work. Worth saying in advance so it doesn't land
as a verdict.

**He also messaged agreeing** that random splits inflate results because speaker
characteristics get learned, and that speaker holdout is the better protocol.
That's him confirming our methodology unprompted. Let him say it in the meeting
rather than repeating it yourself.

---

# PART 6 — What's next

- Run the rest of the models on the same folds — XGBoost, logistic regression,
  KNN, AdaBoost, Naive Bayes → proper comparison table, then tune the winner only
- **Test whether normalising per speaker closes the gap.** Subtract each actor's
  own average from their features, so the numbers describe "how far from this
  person's normal" rather than "how this person sounds". If the gap narrows,
  we've both diagnosed the leak and partly fixed it.
- Keep the gender and intensity slices going
- Add the sequence-feature flag for the RNN track

**Parallel tracks, all scoring through the same harness:** EDA (training folds
only), RNN, CNN over spectrograms, pretrained embeddings (Whisper/wav2vec2),
business deck.

---

# PART 7 — If they ask

**"Why only 51% when papers say 92%?"**
Those use random splits with the same voices in train and test. Our comparable
number is 65%. Published speaker-independent results run 40–70%. We're in range.

**"Why not just push accuracy up?"**
The benchmark moved 1.7 points in four years — we won't win that in three weeks.
And most of our marks are EDA, visualisation, storytelling and business framing,
not accuracy.

**"Why SVM?"**
Best on small datasets with many features, which is exactly what we have. Trains
in seconds. Random Forest ran alongside and scored lower.

**"Isn't this just setup?"**
The setup is how the experiment runs. Deciding how to normalise IS deciding how
much speaker identity leaks. Same question.

**"Does this limit what I can try?"**
No. Any model, any features — it just gets scored on the same folds so we can
compare. Adding a model later takes an afternoon.

**"Why does it matter that neutral is bad?"**
It's why we report macro-F1 as well as accuracy. Accuracy alone would hide a
class we get right less than a quarter of the time.

---

# Still open — raise these

- **Nobody has opened the mid-review instructions doc.** Deadline 6 Sep.
- **Nobody has checked what honor code 4N allows** — relevant, we're using
  Claude Code.
- GitHub default branch is still `master`, needs flipping to `main`.
