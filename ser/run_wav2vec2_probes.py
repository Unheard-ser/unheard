"""Phase 13c — the full layer curve, and what the embeddings know about you.

Two loose ends from `docs/wav2vec2_findings.md`, both cheap because the cache
already holds every layer.

**The complete layer curve.** Phase 13 probed five of thirteen layers and found
an inverted U peaking at layer 6. Five points is enough to see a shape and not
enough to locate a maximum, and the cache holds all thirteen, so there is no
reason to leave it interpolated.

**The actor-identity probe, re-run on learned features.** Phase 7's headline is
that handcrafted features identify the speaker with 92.7% accuracy -- 22.3x
chance -- and that per-speaker normalisation cuts that to 21.2%. Phase 13 found
per-speaker normalisation worth **+16.01 pp** on wav2vec2 embeddings, more than
twice what it was worth on MFCCs, which predicts that these embeddings encode
identity *more* strongly, not less. That prediction is testable with the exact
Phase 7 design, and it is the kind of prediction worth being wrong about in
public.

The design is deliberately identical to `ser.run_phase7.experiment_leakage_diagnosis`:
the same estimator, the same `StratifiedKFold(5, shuffle, seed)`, the same
chance baseline. Actor identity cannot be tested speaker-independently -- a
held-out actor has no training examples of themselves -- so a stratified split
over actors is the appropriate design for this question, as it was there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ser import SEED, set_seeds
from ser.embeddings import EmbeddingConfig
from ser.features import CACHE_DIR, per_speaker_normalise
from ser.models.classical import build
from ser.run_wav2vec2 import FLOOR_F1, build_matrix, load_cached, ravdess_view, score

FIGURES = Path(__file__).resolve().parent.parent / "figures"

#: Phase 7 used svm-rbf for the probe; matched so the numbers are comparable.
PROBE_MODEL = "svm-rbf"

ALL_LAYERS: tuple[int, ...] = tuple(range(13))


def layer_curve(
    tensor: np.ndarray,
    index: pd.DataFrame,
    normalisation: str,
    results_path: Path | None = None,
) -> pd.DataFrame:
    """Macro-F1 for every one of the 13 hidden states.

    Args:
        tensor: Cached ``(clips, layers, [mean,std], dim)`` array.
        index: The matching index frame.
        normalisation: ``global`` or ``per_speaker``.
        results_path: Override the results log.

    Returns:
        One row per layer.
    """
    rows = []
    for layer in ALL_LAYERS:
        config = EmbeddingConfig(
            layer=layer, pooling="mean", normalisation=normalisation
        )
        features, labels = build_matrix(tensor, index, config)
        result = score(
            features, labels, index, config, model="svm-rbf",
            notes=f"phase-13c full layer curve ({normalisation}): layer {layer}",
            results_path=results_path,
        )
        result.pop("predictions", None)
        rows.append({"layer": layer, "normalisation": normalisation, **result})
        print(
            f"  layer {layer:>2d}  {normalisation:<12s} acc={result['accuracy']:.4f} "
            f"f1={result['macro_f1']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def identity_probe(
    tensor: np.ndarray, index: pd.DataFrame, layer: int = 6
) -> pd.DataFrame:
    """Can a classifier recover WHO is speaking from the embeddings?

    Mirrors `ser.run_phase7.experiment_leakage_diagnosis` exactly so the two
    numbers sit in the same table: same estimator, same stratified 5-fold, same
    1/24 chance baseline.

    Args:
        tensor: Cached per-layer tensor, RAVDESS rows only.
        index: The matching index frame.
        layer: Which hidden state to probe. Defaults to the Phase 13 winner.

    Returns:
        One row for raw embeddings and one for per-speaker normalised.
    """
    actors = index["speaker_id"].astype(str).to_numpy()
    n_actors = len(np.unique(actors))
    chance = 1 / n_actors

    rows = []
    for normalise in (False, True):
        config = EmbeddingConfig(
            layer=layer, pooling="mean",
            normalisation="per_speaker" if normalise else "none",
        )
        features, _ = build_matrix(tensor, index, config)
        scores = cross_val_score(
            build(PROBE_MODEL), features.to_numpy(), actors,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="accuracy", n_jobs=-1,
        )
        rows.append({
            "features": f"wav2vec2 L{layer}"
            + (" per-speaker normalised" if normalise else " raw"),
            "actor_id_accuracy": float(scores.mean()),
            "sd": float(scores.std(ddof=1)),
            "chance": chance,
        })
        print(
            f"  {rows[-1]['features']:<44s} {rows[-1]['actor_id_accuracy']:.3f} "
            f"({rows[-1]['actor_id_accuracy']/chance:.1f}x chance)",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table["x_chance"] = (table["actor_id_accuracy"] / table["chance"]).round(1)
    return table


def plot_layer_curve(curves: pd.DataFrame):
    """Macro-F1 against transformer depth, one line per normalisation.

    Returns a Figure; the caller saves it (project convention).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for normalisation, group in curves.groupby("normalisation"):
        group = group.sort_values("layer")
        ax.plot(
            group["layer"], group["macro_f1"],
            marker="o", label=f"{normalisation} normalisation",
        )
    ax.axhline(
        FLOOR_F1, linestyle="--", linewidth=1, color="grey",
        label=f"handcrafted floor ({FLOOR_F1:.3f})",
    )
    ax.set_xlabel("wav2vec2 hidden state (0 = CNN output, 12 = final layer)")
    ax.set_ylabel("macro-F1, pooled out-of-fold")
    ax.set_title("Emotion lives in the middle of wav2vec2, not at its output")
    ax.set_xticks(list(ALL_LAYERS))
    ax.legend(loc="lower center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_identity_probe(handcrafted: pd.DataFrame, learned: pd.DataFrame):
    """Actor identifiability, handcrafted vs learned, raw vs normalised."""
    import matplotlib.pyplot as plt

    labels, values = [], []
    for frame in (handcrafted, learned):
        for row in frame.itertuples():
            labels.append(row.features)
            values.append(row.actor_id_accuracy)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(values))
    ax.barh(positions, values)
    ax.axvline(
        1 / 24, linestyle="--", linewidth=1, color="grey", label="chance (1/24)"
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("actor-identification accuracy")
    ax.set_title("What the features know about the speaker")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def run(results_path: Path | None = None):
    set_seeds()
    base = EmbeddingConfig(batch_size=8)
    tensor, index = load_cached(base)
    tensor, index = ravdess_view(tensor, index)
    print(f"RAVDESS view: {tensor.shape}\n", flush=True)

    print("full layer curve, per-speaker normalisation")
    per_speaker = layer_curve(tensor, index, "per_speaker", results_path)
    print("\nfull layer curve, global normalisation")
    global_norm = layer_curve(tensor, index, "global", results_path)
    curves = pd.concat([per_speaker, global_norm], ignore_index=True)

    best = curves.loc[curves["macro_f1"].idxmax()]
    print(
        f"\n  best: layer {int(best['layer'])} / {best['normalisation']} "
        f"= {best['macro_f1']:.4f}\n",
        flush=True,
    )

    print("actor-identity probe on wav2vec2 embeddings")
    learned = identity_probe(tensor, index)

    # Phase 7's handcrafted numbers, for the side-by-side figure.
    handcrafted = pd.DataFrame([
        {"features": "handcrafted raw", "actor_id_accuracy": 0.927,
         "sd": float("nan"), "chance": 1 / 24, "x_chance": 22.3},
        {"features": "handcrafted per-speaker normalised",
         "actor_id_accuracy": 0.212, "sd": float("nan"), "chance": 1 / 24,
         "x_chance": 5.1},
    ])

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_layer_curve(curves).savefig(
        FIGURES / "18_wav2vec2_layer_curve.png", dpi=150
    )
    plot_identity_probe(handcrafted, learned).savefig(
        FIGURES / "19_actor_identity_handcrafted_vs_wav2vec2.png", dpi=150
    )
    print(f"\nwrote figures 18 and 19 to {FIGURES}")

    return curves, learned, handcrafted


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    curves, learned, handcrafted = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    curves.to_parquet(CACHE_DIR / "phase13c_layer_curve.parquet", index=False)
    learned.to_parquet(CACHE_DIR / "phase13c_identity_probe.parquet", index=False)

    print("\n=== full layer curve ===")
    print(curves.pivot(index="layer", columns="normalisation", values="macro_f1")
          .round(4).to_string())
    print("\n=== actor identity ===")
    print(pd.concat([handcrafted, learned], ignore_index=True).to_string(index=False))
