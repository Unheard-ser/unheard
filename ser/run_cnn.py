"""Phase 19 — a 1-D CNN over wav2vec2 frames, instead of pooling them.

Every model in this project so far collapses a clip to one vector before
classifying it: `ser.embeddings.extract` averages wav2vec2's output across all
frames. That discards *when* things happen. A rise in pitch followed by a fall
and a fall followed by a rise produce the same mean.

This keeps the time axis. Frame-level states (~180 frames x 768) go into a small
1-D convolutional network, so filters slide along time and can respond to
ordered patterns rather than aggregate statistics.

It is the natural test of whether pooling was throwing anything away. If the CNN
matches the pooled linear model, temporal structure was not carrying extra
signal at this scale — which is itself worth knowing, and is the more likely
outcome given that a *linear* head beat XGBoost by 13 pp on the pooled features.

Three implementation points that are easy to get wrong
-----------------------------------------------------
**Masked pooling.** Clips run 2.94-5.27 s, so frames are padded to a common
length. The masked mean divides by each clip's *true* length; without it a short
clip's average is diluted by padding the network would otherwise read as real
silence.

**Deliberately small.** 1,140 training clips. Two conv blocks and a 128-unit
width is already generous, and `ser/run_feature_augment.py` established that the
model is not short of examples — adding synthetic ones changed nothing — so
capacity is the thing to keep down, not data the thing to manufacture.

**Per-speaker normalisation, across frames.** Worth +16 pp on the pooled
features. Applied here over each speaker's own frames, under the same stated
exception in CLAUDE.md rule 2.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ser import SEED, set_seeds
from ser.embeddings import EmbeddingConfig, extract_frames
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR
from ser.metadata import load as load_metadata
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

LAYER = 6
MAX_FRAMES = 280
WIDTH = 128
DROPOUT = 0.3
EPOCHS = 40
LR = 1e-3
BATCH = 32

FRAMES_CACHE = CACHE_DIR / f"frames_L{LAYER}_RAVDESS.npy"
LENGTHS_CACHE = CACHE_DIR / f"frames_L{LAYER}_RAVDESS_lengths.npy"


def build_cache(meta: pd.DataFrame):
    """Extract and cache frame-level states for every RAVDESS clip."""
    if FRAMES_CACHE.is_file() and LENGTHS_CACHE.is_file():
        return np.load(FRAMES_CACHE, mmap_mode="r"), np.load(LENGTHS_CACHE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x = np.zeros((len(meta), MAX_FRAMES, 768), dtype=np.float32)
    lengths = np.zeros(len(meta), dtype=np.int32)

    cursor, started = 0, time.time()
    for states, counts, _ in extract_frames(
        meta, EmbeddingConfig(batch_size=8), layer=LAYER
    ):
        for i in range(len(states)):
            n = int(min(counts[i], MAX_FRAMES))
            x[cursor + i, :n] = states[i, :n]
            lengths[cursor + i] = n
        cursor += len(states)

    print(f"  extracted in {(time.time() - started) / 60:.1f} min", flush=True)
    np.save(FRAMES_CACHE, x)
    np.save(LENGTHS_CACHE, lengths)
    return np.load(FRAMES_CACHE, mmap_mode="r"), lengths


def normalise_per_speaker(x: np.ndarray, lengths, actors) -> np.ndarray:
    """Z-score each speaker's frames against that speaker's own statistics."""
    out = np.array(x, dtype=np.float32)
    for actor in np.unique(actors):
        rows = np.flatnonzero(actors == actor)
        stacked = np.concatenate([out[i, : lengths[i]] for i in rows])
        mean, sd = stacked.mean(0), stacked.std(0) + 1e-8
        for i in rows:
            out[i, : lengths[i]] = (out[i, : lengths[i]] - mean) / sd
    return out


def make_net(n_classes: int):
    import torch.nn as nn

    class EmotionCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(768, WIDTH, 5, padding=2),
                nn.BatchNorm1d(WIDTH), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Conv1d(WIDTH, WIDTH, 5, padding=2),
                nn.BatchNorm1d(WIDTH), nn.ReLU(), nn.Dropout(DROPOUT),
            )
            self.head = nn.Linear(WIDTH, n_classes)

        def forward(self, x, lengths):
            import torch

            h = self.conv(x.transpose(1, 2))
            mask = (
                torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
            ).float().unsqueeze(1)
            h = (h * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return self.head(h)

    return EmotionCNN()


def run(results_path: Path | None = None):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset

    set_seeds()
    torch.set_num_threads(16)

    meta = load_metadata()
    print(f"extracting frame-level features for {len(meta):,} clips ...", flush=True)
    frames, lengths = build_cache(meta)
    print(f"  frames {frames.shape} | per clip {lengths.min()}-{lengths.max()}\n", flush=True)

    actors = meta["actor"].to_numpy()
    x = normalise_per_speaker(frames, lengths, actors)
    encoder = LabelEncoder().fit(meta["emotion_label"])
    y = encoder.transform(meta["emotion_label"])
    position = {name: i for i, name in enumerate(meta["filename"])}
    indexed = meta.set_index("filename")

    folds = load_folds()
    per_fold, collected = [], []

    for fold in range(5):
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        tr = [position[n] for n in train_files]
        te = [position[n] for n in test_files]

        torch.manual_seed(SEED)
        net = make_net(len(encoder.classes_))
        optimiser = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(x[tr]),
                          torch.tensor(lengths[tr], dtype=torch.long),
                          torch.tensor(y[tr])),
            batch_size=BATCH, shuffle=True,
        )

        started = time.time()
        net.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for xb, lb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(net(xb, lb), yb)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(xb)
            if (epoch + 1) % 20 == 0:
                print(f"  fold {fold} epoch {epoch+1:>2}  loss {total/len(tr):.4f}",
                      flush=True)

        net.eval()
        with torch.inference_mode():
            logits = net(torch.tensor(x[te]),
                         torch.tensor(lengths[te], dtype=torch.long))
        predicted = encoder.inverse_transform(logits.argmax(1).numpy())
        truth = indexed.loc[test_files, "emotion_label"].to_numpy()
        test_meta = indexed.loc[test_files].reset_index()

        per_fold.append((truth, predicted, test_meta))
        collected.append(pd.DataFrame({
            "y_true": truth, "y_pred": predicted,
            "speaker": test_meta["actor"].astype(str).to_numpy()}))
        evaluate(truth, predicted, "speaker_independent", fold, test_meta,
                 model="cnn1d-wav2vec2", feature_config=f"w2v-L{LAYER}-frames-per_speaker",
                 hyperparams={"epochs": EPOCHS, "width": WIDTH, "lr": LR},
                 notes="phase-19 1D CNN over wav2vec2 frames",
                 results_path=results_path)
        print(f"  fold {fold}: acc {(truth == predicted).mean():.4f} "
              f"({time.time() - started:.0f}s)\n", flush=True)

    pooled = evaluate_pooled(
        per_fold, "speaker_independent", model="cnn1d-wav2vec2",
        feature_config=f"w2v-L{LAYER}-frames-per_speaker",
        hyperparams={"epochs": EPOCHS, "width": WIDTH, "lr": LR},
        notes="phase-19 1D CNN over wav2vec2 frames (pooled out-of-fold)",
        results_path=results_path)

    allp = pd.concat(collected, ignore_index=True)
    ci = cluster_bootstrap_ci(allp.y_true, allp.y_pred, allp.speaker, "macro_f1")

    print(f"\nCNN over frames : acc {pooled['accuracy']:.4f}  "
          f"macro-F1 {pooled['macro_f1']:.4f}  [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"pooled + linear : acc 0.8243  macro-F1 0.8220   (Phase 17, nested CV)")
    print(f"handcrafted     : acc 0.6389  macro-F1 0.6311   (Phase 5)")

    return pooled, allp


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    pooled, allp = run(results_path=args.results)
    allp.to_parquet(CACHE_DIR / "phase19_cnn_predictions.parquet", index=False)
    print("\n=== per class ===")
    print(pooled["per_class"].round(3).to_string())
