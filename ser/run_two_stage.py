"""Phase 16 — can we use all four corpora *without* losing accuracy?

Every earlier attempt pooled the external clips into one training set and fitted
once. That lost 11 pp on handcrafted features and 2.5 pp on wav2vec2, and four
covariate-shift remedies failed to beat using none of it.

**Pooling is not the only way to use extra data, and it is not the one transfer
learning normally uses.** The standard recipe is sequential:

    stage 1 — learn general structure from the large, mixed set
    stage 2 — specialise on the small set that matches the test distribution

The difference matters. Pooling asks one decision boundary to serve 12,162
clips from four recording setups at once, so the 10,722 foreign clips outvote
the 1,140 that match the test set. Two-stage lets the model use the foreign data
to find a good starting point and then move to where RAVDESS actually is.

This has never been tested on this project, and it is the only remaining route
by which the merged corpus could earn its place in the final model.

Four arms, identical frozen RAVDESS test folds throughout:

``ravdess_only``      the control -- trained on 1,140 RAVDESS clips
``pooled``            everything in one training set (what failed before)
``two_stage``         pretrain on all 11,862, then fine-tune on RAVDESS
``two_stage_frozen``  pretrain on all, then retrain only the output layer

An MLP is used rather than logistic regression because two-stage needs a model
that can be *continued* — sklearn's `warm_start` resumes from existing weights.
So the control here is an MLP too, not the 0.7944 headline; the comparison is
internal.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from ser import SEED, set_seeds
from ser import corpora
from ser.embeddings import EmbeddingConfig
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR
from ser.run_wav2vec2 import build_matrix, load_cached
from ser.run_wav2vec2_transfer import WINNER
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

HIDDEN = (256,)
#: Stage-1 budget. Kept modest: the point is a good starting point, not a
#: converged model on the wrong distribution.
PRETRAIN_ITER = 60
#: Stage-2 budget, on 1,140 clips.
FINETUNE_ITER = 120
#: Single-stage budget, so the control is not handicapped by iteration count.
SINGLE_ITER = 180


def mlp(max_iter: int, warm: bool = False) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN, max_iter=max_iter, random_state=SEED,
        early_stopping=False, warm_start=warm, learning_rate_init=1e-3,
    )


def run(results_path: Path | None = None):
    set_seeds()
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    features, labels = build_matrix(tensor, index, WINNER)
    indexed = index.set_index("filename")
    is_external = indexed["source"].astype(str) != "RAVDESS"
    external = indexed.index[is_external].tolist()
    print(f"features {features.shape} | external clips {len(external):,}\n", flush=True)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique())

    kept, rows = {}, []

    def cell(name: str, fit_fn, notes: str):
        started, per_fold, frames = time.time(), [], []
        for fold in fold_numbers:
            tr, te = get_split("speaker_independent", fold, folds=folds)
            model = fit_fn(tr)
            pred = model.predict(features.loc[te].to_numpy())
            truth = labels.loc[te].to_numpy()
            meta = indexed.loc[te].reset_index()
            per_fold.append((truth, pred, meta))
            frames.append(pd.DataFrame({
                "y_true": truth, "y_pred": pred,
                "speaker": meta["speaker_id"].to_numpy()}))
            evaluate(truth, pred, "speaker_independent", fold, meta,
                     model="mlp", feature_config=name,
                     hyperparams={"arm": name, "hidden": list(HIDDEN)},
                     notes=notes, results_path=results_path)

        pooled = evaluate_pooled(
            per_fold, "speaker_independent", model="mlp", feature_config=name,
            hyperparams={"arm": name, "hidden": list(HIDDEN)},
            notes=notes + " (pooled out-of-fold)", results_path=results_path)
        kept[name] = pd.concat(frames, ignore_index=True)
        rows.append({"arm": name, "accuracy": pooled["accuracy"],
                     "macro_f1": pooled["macro_f1"],
                     "seconds": round(time.time() - started, 1)})
        print(f"  {name:<20s} acc={pooled['accuracy']:.4f} "
              f"f1={pooled['macro_f1']:.4f}  ({rows[-1]['seconds']}s)", flush=True)

    # --- arm 1: control, RAVDESS only -------------------------------------
    def ravdess_only(tr):
        m = mlp(SINGLE_ITER)
        m.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
        return m

    # --- arm 2: pooled, everything at once --------------------------------
    def pooled_all(tr):
        full = tr + external
        m = mlp(SINGLE_ITER)
        m.fit(features.loc[full].to_numpy(), labels.loc[full].to_numpy())
        return m

    # --- arm 3: two-stage, full fine-tune ---------------------------------
    def two_stage(tr):
        full = tr + external
        m = mlp(PRETRAIN_ITER, warm=True)
        m.fit(features.loc[full].to_numpy(), labels.loc[full].to_numpy())
        # Continue from the pretrained weights on RAVDESS alone. warm_start
        # keeps the coefficients; max_iter is reset for the second stage.
        m.set_params(max_iter=FINETUNE_ITER)
        m.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
        return m

    # --- arm 4: two-stage, output layer only ------------------------------
    def two_stage_frozen(tr):
        """Pretrain, then relearn only the output layer on RAVDESS.

        The hidden layer keeps what the 12,162 clips taught it; only the final
        decision layer moves to RAVDESS. This is the 'frozen backbone' version
        of transfer learning, and it is the arm most likely to keep the benefit
        of the extra data without being dragged by it.
        """
        from sklearn.linear_model import LogisticRegression

        full = tr + external
        m = mlp(PRETRAIN_ITER)
        m.fit(features.loc[full].to_numpy(), labels.loc[full].to_numpy())

        hidden = np.maximum(
            features.loc[tr].to_numpy() @ m.coefs_[0] + m.intercepts_[0], 0)
        head = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED)
        head.fit(hidden, labels.loc[tr].to_numpy())

        class Wrapped:
            def predict(self, X):
                h = np.maximum(X @ m.coefs_[0] + m.intercepts_[0], 0)
                return head.predict(h)

        return Wrapped()

    print("four ways to use the data")
    cell("ravdess_only", ravdess_only, "phase-16 control: MLP on RAVDESS only")
    cell("pooled", pooled_all, "phase-16 pooled: MLP on all 4 corpora at once")
    cell("two_stage", two_stage, "phase-16 two-stage: pretrain merged, fine-tune RAVDESS")
    cell("two_stage_frozen", two_stage_frozen,
         "phase-16 two-stage frozen backbone, head relearned on RAVDESS")

    table = pd.DataFrame(rows)
    base = float(table.loc[table.arm == "ravdess_only", "macro_f1"].iloc[0])
    table["vs_ravdess_only_pp"] = ((table["macro_f1"] - base) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\npaired vs the RAVDESS-only control")
    ctrl = kept["ravdess_only"]
    sig = []
    for name, frame in kept.items():
        if name == "ravdess_only":
            continue
        b = paired_cluster_bootstrap(frame["y_true"], frame["y_pred"],
                                     ctrl["y_pred"], frame["speaker"], "macro_f1")
        sig.append({"arm": name, "delta_pp": round(100*b["delta"], 2),
                    "lo_pp": round(100*b["lo"], 2), "hi_pp": round(100*b["hi"], 2),
                    "p": round(b["p_value"], 4), "significant": b["significant"]})
        mark = "SIGNIFICANT" if b["significant"] else "not distinguishable"
        print(f"  {name:<20s} {100*b['delta']:+6.2f} pp "
              f"[{100*b['lo']:+6.2f},{100*b['hi']:+6.2f}] p={b['p_value']:.3f} -> {mark}")

    best = table.iloc[0]
    ci = cluster_bootstrap_ci(kept[best["arm"]]["y_true"], kept[best["arm"]]["y_pred"],
                              kept[best["arm"]]["speaker"], "macro_f1")
    print(f"\n  best arm: {best['arm']} = {best['macro_f1']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    return table, pd.DataFrame(sig)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    table, sig = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase16_two_stage.parquet", index=False)
    sig.to_parquet(CACHE_DIR / "phase16_significance.parquet", index=False)
    print("\n=== Phase 16: four ways to use four corpora ===")
    print(table.to_string(index=False))
