"""Phase 14b — is a different encoder the headroom Phase 14a could not find?

Phase 14a swept six levers around the wav2vec2 layer-6 configuration and moved
the number by less than a percentage point in either direction. That rules out
the head, the regularisation, the layer choice, the dimensionality and the
class weighting. What it leaves is the representation itself.

WavLM-base-plus is the strongest candidate at the same size. Its pretraining
adds two things to wav2vec2's masked-prediction objective: denoising, and
simulated overlapping speech, which forces the encoder to separate speakers
rather than blend them. It leads wav2vec2 on SUPERB's paralinguistic tasks.
Same 12 layers, same 768 dimensions, so every piece of Phase 13's apparatus
applies without modification.

Both encoders get the **full 13-layer curve**, not a single layer, because
Phase 13 showed the optimum is mid-network and there is no reason to assume two
differently-pretrained models peak in the same place. Comparing wav2vec2's best
layer against WavLM's last layer would be a rigged comparison; comparing best
against best is the fair one.

A second question this settles, given Phase 13c: WavLM's speaker-simulation
objective might make it encode identity *more* strongly still. The actor probe
is re-run on whichever encoder wins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser.embeddings import EmbeddingConfig
from ser.run_wav2vec2 import FLOOR_F1, build_matrix, load_cached, ravdess_view, score
from ser.run_wav2vec2_probes import ALL_LAYERS, FIGURES, identity_probe
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

ENCODERS: tuple[str, ...] = (
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base-plus",
)

#: Phase 13's winner, for reference in the printed comparison.
WAV2VEC2_BEST_F1 = 0.785612  # layer 6, per-speaker, svm-rbf head


def curve_for(
    model_id: str, results_path: Path | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Full 13-layer curve for one encoder, per-speaker normalised.

    Returns the table and the pooled predictions per layer, so the winner can
    be compared against another encoder's winner with a paired test.
    """
    base = EmbeddingConfig(model_id=model_id, batch_size=8)
    tensor, index = load_cached(base)
    tensor, index = ravdess_view(tensor, index)

    rows, kept = [], {}
    for layer in ALL_LAYERS:
        config = EmbeddingConfig(
            model_id=model_id, layer=layer, pooling="mean",
            normalisation="per_speaker",
        )
        features, labels = build_matrix(tensor, index, config)
        result = score(
            features, labels, index, config, model="svm-rbf",
            notes=f"phase-14b encoder comparison: {model_id} layer {layer}",
            results_path=results_path,
        )
        kept[f"L{layer}"] = result.pop("predictions")
        rows.append({
            "encoder": model_id.split("/")[-1], "layer": layer,
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
        })
        print(
            f"  layer {layer:>2d}  acc={result['accuracy']:.4f} "
            f"f1={result['macro_f1']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows), kept


def plot_encoder_curves(curves: pd.DataFrame):
    """Both encoders' layer curves on one axis. Returns a Figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for encoder, group in curves.groupby("encoder"):
        group = group.sort_values("layer")
        ax.plot(group["layer"], group["macro_f1"], marker="o", label=encoder)
    ax.axhline(
        FLOOR_F1, linestyle="--", linewidth=1, color="grey",
        label=f"handcrafted floor ({FLOOR_F1:.3f})",
    )
    ax.set_xlabel("hidden state (0 = CNN output, 12 = final layer)")
    ax.set_ylabel("macro-F1, pooled out-of-fold")
    ax.set_title("Where each encoder keeps its emotion information")
    ax.set_xticks(list(ALL_LAYERS))
    ax.legend(loc="lower center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def run(results_path: Path | None = None):
    set_seeds()
    all_curves, all_preds = [], {}

    for model_id in ENCODERS:
        print(f"\n{model_id}")
        try:
            curve, kept = curve_for(model_id, results_path)
        except FileNotFoundError as exc:
            print(f"  skipped -- no cached embeddings ({exc})", flush=True)
            continue
        all_curves.append(curve)
        all_preds[model_id] = kept
        best = curve.loc[curve["macro_f1"].idxmax()]
        print(
            f"  -> best layer {int(best['layer'])} = {best['macro_f1']:.4f}",
            flush=True,
        )

    if len(all_curves) < 2:
        print("\nneed both encoders cached to compare; run the extractions first")
        return pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame(), None

    curves = pd.concat(all_curves, ignore_index=True)

    # --- best vs best, paired ----------------------------------------------
    winners = {}
    for model_id in ENCODERS:
        sub = curves[curves["encoder"] == model_id.split("/")[-1]]
        best_layer = int(sub.loc[sub["macro_f1"].idxmax(), "layer"])
        winners[model_id] = (best_layer, all_preds[model_id][f"L{best_layer}"])

    (w2v_layer, w2v_pred) = winners["facebook/wav2vec2-base"]
    (wavlm_layer, wavlm_pred) = winners["microsoft/wavlm-base-plus"]

    boot = paired_cluster_bootstrap(
        wavlm_pred["y_true"], wavlm_pred["y_pred"], w2v_pred["y_pred"],
        wavlm_pred["speaker"], "macro_f1",
    )
    wavlm_ci = cluster_bootstrap_ci(
        wavlm_pred["y_true"], wavlm_pred["y_pred"], wavlm_pred["speaker"], "macro_f1"
    )

    print("\n=== best vs best ===")
    print(f"  wav2vec2-base    layer {w2v_layer}")
    print(f"  wavlm-base-plus  layer {wavlm_layer}  "
          f"{wavlm_ci['point']:.4f} [{wavlm_ci['lo']:.4f}, {wavlm_ci['hi']:.4f}]")
    verdict = "SIGNIFICANT" if boot["significant"] else "not distinguishable"
    print(
        f"  WavLM - wav2vec2 = {100*boot['delta']:+.2f} pp "
        f"[{100*boot['lo']:+.2f}, {100*boot['hi']:+.2f}] "
        f"p={boot['p_value']:.4f} -> {verdict}"
    )

    # --- does WavLM leak identity harder? ----------------------------------
    print("\nactor-identity probe on the winning encoder")
    winning_id = (
        "microsoft/wavlm-base-plus" if boot["delta"] > 0 else "facebook/wav2vec2-base"
    )
    best_layer = winners[winning_id][0]
    tensor, index = load_cached(EmbeddingConfig(model_id=winning_id, batch_size=8))
    tensor, index = ravdess_view(tensor, index)
    probe = identity_probe(tensor, index, layer=best_layer)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_encoder_curves(curves).savefig(
        FIGURES / "20_encoder_layer_curves.png", dpi=150
    )
    print(f"\nwrote figure 20 to {FIGURES}")

    summary = pd.DataFrame([{
        "wav2vec2_best_layer": w2v_layer,
        "wavlm_best_layer": wavlm_layer,
        "delta_pp": round(100 * boot["delta"], 2),
        "lo_pp": round(100 * boot["lo"], 2),
        "hi_pp": round(100 * boot["hi"], 2),
        "p": round(boot["p_value"], 4),
        "significant": boot["significant"],
    }])
    return curves, (summary, probe)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    curves, extra = run(results_path=args.results)
    out = Path("features")
    out.mkdir(exist_ok=True)
    curves.to_parquet(out / "phase14b_encoder_curves.parquet", index=False)

    print("\n=== layer curves ===")
    if not curves.empty:
        print(curves.pivot(index="layer", columns="encoder", values="macro_f1")
              .round(4).to_string())
    if extra is not None:
        summary, probe = extra
        summary.to_parquet(out / "phase14b_summary.parquet", index=False)
        probe.to_parquet(out / "phase14b_identity_probe.parquet", index=False)
        print("\n=== identity probe on the winner ===")
        print(probe.to_string(index=False))
