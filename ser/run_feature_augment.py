"""Phase 18 — augmentation in feature space, where torch is not required.

`PLAN.md` has listed augmentation since Phase 6 and nobody has run it. The
obvious form — perturbing the waveform and re-extracting — needs a working torch
to push the altered audio back through wav2vec2, which is unavailable on this
machine (see `docs/environment.md`). Feature-space augmentation needs none of
that: it operates on the cached embeddings directly.

It also attacks the right problem. The model has 1,140 training clips against 768
(or 1,536) feature dimensions. That ratio, not the classifier, is what limits it
— Phase 14a found that regularisation and dimensionality reduction both failed to
help, which is what a data-limited rather than capacity-limited problem looks
like.

Three methods, each training-fold only (CLAUDE.md rule 3 — no augmented example
ever reaches a test fold):

``noise``
    Add Gaussian noise scaled to each feature's own standard deviation. The
    cheapest regulariser there is: it tells the model that small perturbations
    should not change the label.

``mixup``
    Convex combinations of **two clips of the same emotion from different
    speakers**. Standard mixup blends labels too, which sklearn's classifiers
    cannot consume; restricting to within-class pairs keeps the label
    unambiguous while still producing a genuinely new point. Mixing across
    speakers is deliberate — the synthetic example sits between two voices, which
    is exactly the direction the model needs to generalise in, given that
    speaker identity is recoverable from these features with 94.9% accuracy.

``both``
    Mixup first, then noise on everything.

Every variant uses the Phase 17 winning configuration so the comparison isolates
the augmentation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ser import SEED, set_seeds
from ser.embeddings import EmbeddingConfig, select
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR, per_speaker_normalise
from ser.run_wav2vec2 import load_cached, ravdess_view
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

#: The Phase 17 winner, chosen by nested CV.
WINNER = EmbeddingConfig(layer=6, pooling="mean_std", normalisation="per_speaker")
C = 0.01

NOISE_SIGMAS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
#: Synthetic examples as a multiple of the real training set.
MIX_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: Beta parameter. 0.4 keeps blends close to one parent; 1.0 is uniform.
MIX_ALPHA = 0.4


def add_noise(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Gaussian noise scaled per feature, so every dimension is perturbed equally."""
    scale = x.std(axis=0, keepdims=True)
    return x + rng.normal(0.0, sigma, size=x.shape) * scale


def within_class_mixup(
    x: np.ndarray, y: np.ndarray, speakers: np.ndarray, ratio: float, rng
):
    """Blend pairs from the same class but different speakers.

    Returns only the synthetic rows; the caller concatenates them onto the real
    ones. Pairs are drawn within a class so the blend's label is unambiguous,
    and across speakers so the synthetic point lies between two voices rather
    than describing one of them more thickly.
    """
    new_x, new_y = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if len(idx) < 2:
            continue
        n = int(round(len(idx) * ratio))
        for _ in range(n):
            a, b = rng.choice(idx, size=2, replace=False)
            # resample until the two come from different speakers, but give up
            # rather than loop forever on a class held by one speaker
            for _ in range(8):
                if speakers[a] != speakers[b]:
                    break
                a, b = rng.choice(idx, size=2, replace=False)
            lam = rng.beta(MIX_ALPHA, MIX_ALPHA)
            new_x.append(lam * x[a] + (1 - lam) * x[b])
            new_y.append(label)
    if not new_x:
        return np.empty((0, x.shape[1])), np.empty(0, dtype=y.dtype)
    return np.vstack(new_x), np.asarray(new_y, dtype=y.dtype)


def run(results_path: Path | None = None):
    set_seeds()
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    tensor, index = ravdess_view(tensor, index)
    indexed = index.set_index("filename")

    matrix = select(tensor, WINNER)
    matrix = per_speaker_normalise(matrix, index["speaker_id"].to_numpy())
    features = pd.DataFrame(matrix, index=index["filename"])
    labels = pd.Series(index["emotion_label"].to_numpy(), index=index["filename"])
    print(f"features {features.shape}  (Phase 17 winner: {WINNER.name})\n", flush=True)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique())

    kept, rows = {}, []

    def cell(name: str, augment, notes: str):
        started, per_fold, frames = time.time(), [], []
        for fold in fold_numbers:
            tr, te = get_split("speaker_independent", fold, folds=folds)
            x_tr = features.loc[tr].to_numpy()
            y_tr = labels.loc[tr].to_numpy()
            spk = indexed.loc[tr, "speaker_id"].to_numpy()

            if augment is not None:
                # Training fold only. The test fold is never touched.
                rng = np.random.default_rng(SEED + fold)
                x_tr, y_tr = augment(x_tr, y_tr, spk, rng)

            model = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            model.fit(x_tr, y_tr)
            pred = model.predict(features.loc[te].to_numpy())

            truth = labels.loc[te].to_numpy()
            meta = indexed.loc[te].reset_index()
            per_fold.append((truth, pred, meta))
            frames.append(pd.DataFrame({
                "y_true": truth, "y_pred": pred,
                "speaker": meta["speaker_id"].to_numpy()}))
            evaluate(truth, pred, "speaker_independent", fold, meta,
                     model="logistic-regression", feature_config=name,
                     hyperparams={"augmentation": name}, notes=notes,
                     results_path=results_path)

        pooled = evaluate_pooled(
            per_fold, "speaker_independent", model="logistic-regression",
            feature_config=name, hyperparams={"augmentation": name},
            notes=notes + " (pooled out-of-fold)", results_path=results_path)
        kept[name] = pd.concat(frames, ignore_index=True)
        rows.append({"variant": name, "accuracy": pooled["accuracy"],
                     "macro_f1": pooled["macro_f1"],
                     "seconds": round(time.time() - started, 1)})
        print(f"  {name:<28s} acc={pooled['accuracy']:.4f} "
              f"f1={pooled['macro_f1']:.4f}  ({rows[-1]['seconds']}s)", flush=True)

    print("baseline")
    cell("no augmentation", None, "phase-18 baseline, no augmentation")

    print("\nGaussian noise on the training fold")
    for s in NOISE_SIGMAS:
        cell(f"noise sigma={s:g}",
             (lambda x, y, sp, rng, s=s: (add_noise(x, s, rng), y)),
             f"phase-18 gaussian noise sigma={s:g}")

    print("\nwithin-class, cross-speaker mixup")
    for r in MIX_RATIOS:
        def mix(x, y, sp, rng, r=r):
            nx, ny = within_class_mixup(x, y, sp, r, rng)
            return np.vstack([x, nx]), np.concatenate([y, ny])
        cell(f"mixup x{r:g}", mix, f"phase-18 within-class mixup ratio={r:g}")

    print("\nboth")
    for r, s in ((1.0, 0.1), (2.0, 0.1)):
        def both(x, y, sp, rng, r=r, s=s):
            nx, ny = within_class_mixup(x, y, sp, r, rng)
            X = np.vstack([x, nx]); Y = np.concatenate([y, ny])
            return add_noise(X, s, rng), Y
        cell(f"mixup x{r:g} + noise {s:g}", both,
             f"phase-18 mixup ratio={r:g} with noise sigma={s:g}")

    table = pd.DataFrame(rows)
    base = float(table.loc[table.variant == "no augmentation", "macro_f1"].iloc[0])
    table["vs_baseline_pp"] = ((table["macro_f1"] - base) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\npaired vs no augmentation")
    ctrl = kept["no augmentation"]
    sig = []
    for name, frame in kept.items():
        if name == "no augmentation":
            continue
        b = paired_cluster_bootstrap(frame["y_true"], frame["y_pred"],
                                     ctrl["y_pred"], frame["speaker"], "macro_f1")
        sig.append({"variant": name, "delta_pp": round(100*b["delta"], 2),
                    "lo_pp": round(100*b["lo"], 2), "hi_pp": round(100*b["hi"], 2),
                    "p": round(b["p_value"], 4), "significant": b["significant"]})
        mark = "SIGNIFICANT" if b["significant"] else "not distinguishable"
        print(f"  {name:<28s} {100*b['delta']:+5.2f} pp "
              f"[{100*b['lo']:+5.2f},{100*b['hi']:+5.2f}] p={b['p_value']:.3f} -> {mark}")

    best = table.iloc[0]
    ci = cluster_bootstrap_ci(kept[best["variant"]]["y_true"],
                              kept[best["variant"]]["y_pred"],
                              kept[best["variant"]]["speaker"], "macro_f1")
    print(f"\n  best: {best['variant']} = {best['macro_f1']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    wins = [s for s in sig if s["significant"] and s["delta_pp"] > 0]
    print(f"  variants SIGNIFICANTLY better than no augmentation: {len(wins)}")
    return table, pd.DataFrame(sig)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    table, sig = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase18_augmentation.parquet", index=False)
    sig.to_parquet(CACHE_DIR / "phase18_significance.parquet", index=False)
    print("\n=== Phase 18: feature-space augmentation ===")
    print(table.to_string(index=False))
