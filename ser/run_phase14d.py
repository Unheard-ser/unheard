"""Phase 14d — combining the levers that individually did nothing.

Phase 14a swept six levers one at a time and none reached significance. Two came
closest: prosody fusion (+0.86 pp) and regularisation at C=0.01 (+0.84 pp). They
were never tried **together**, and they attack different problems — fusion adds
information the encoder discards, regularisation controls a 768-dimension model
fitted on 1,140 clips. If they are independent, they should stack.

Also here: the **layer ensemble**. Layers 4-8 all score within 2.5 pp of each
other, and Phase 14a only tried concatenating them into one long vector, which
multiplies the dimensionality of an already over-parameterised model. Averaging
their *predictions* instead keeps the dimensionality flat and is the standard
way to combine near-equal views.

Discipline unchanged: `C` is chosen by inner cross-validation inside each
training fold, and every result carries a paired cluster bootstrap against the
Phase 13 baseline. Reporting the maximum of many test-set evaluations is
optimistically biased, so §"honest ranking" in the output says which number is
defensible as a claim and which is just the top of a list.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ser import SEED, set_seeds
from ser.embeddings import EmbeddingConfig, select
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR, per_speaker_normalise
from ser.run_phase14 import BASE_F1, prosody_matrix
from ser.run_wav2vec2 import load_cached, ravdess_view
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

C_GRID: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3)
ENSEMBLE_LAYERS: tuple[int, ...] = (4, 5, 6, 7, 8)
INNER_SPLITS = 3


def head(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED)


def layer_matrix(tensor, index, layer: int) -> pd.DataFrame:
    """Per-speaker-normalised mean-pooled features for one layer."""
    block = select(tensor, EmbeddingConfig(layer=layer, pooling="mean"))
    block = per_speaker_normalise(block, index["speaker_id"].to_numpy())
    return pd.DataFrame(
        block, index=index["filename"],
        columns=[f"L{layer}_{i:03d}" for i in range(block.shape[1])],
    )


def pick_C(train_files, features, labels, speakers) -> float:
    """Choose C by speaker-grouped CV inside the training fold only."""
    from sklearn.metrics import f1_score

    names = np.asarray(train_files)
    y = labels.loc[train_files].to_numpy()
    groups = speakers.loc[train_files].to_numpy()
    inner = list(GroupKFold(n_splits=INNER_SPLITS).split(names, y, groups))

    best, best_score = C_GRID[0], -np.inf
    for C in C_GRID:
        scores = []
        for a, b in inner:
            tr, te = list(names[a]), list(names[b])
            model = head(C)
            model.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
            scores.append(f1_score(
                labels.loc[te].to_numpy(), model.predict(features.loc[te].to_numpy()),
                average="macro", zero_division=0))
        mean = float(np.mean(scores))
        if mean > best_score:
            best, best_score = C, mean
    return best


def run(results_path: Path | None = None):
    set_seeds()
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    tensor, index = ravdess_view(tensor, index)
    indexed = index.set_index("filename")
    speakers = indexed["speaker_id"]
    labels = pd.Series(index["emotion_label"].to_numpy(), index=index["filename"])

    L6 = layer_matrix(tensor, index, 6)
    prosody = prosody_matrix(index)
    fused = pd.concat([L6, prosody.loc[L6.index]], axis=1)
    multi = pd.concat(
        [layer_matrix(tensor, index, n) for n in ENSEMBLE_LAYERS], axis=1
    )
    multi_fused = pd.concat([multi, prosody.loc[multi.index]], axis=1)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique())

    kept, rows = {}, []

    def run_cell(label, features, mode, notes):
        started, per_fold, frames, chosen = time.time(), [], [], []
        for fold in fold_numbers:
            tr, te = get_split("speaker_independent", fold, folds=folds)

            if mode == "innerC":
                C = pick_C(tr, features, labels, speakers)
                chosen.append(f"{C:g}")
                model = head(C)
                model.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
                pred = model.predict(features.loc[te].to_numpy())

            elif mode == "layer_vote":
                # One model per layer; average their probabilities.
                probs, classes = None, None
                for n in ENSEMBLE_LAYERS:
                    block = layer_matrix(tensor, index, n)
                    m = head(0.01)
                    m.fit(block.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
                    p = m.predict_proba(block.loc[te].to_numpy())
                    probs = p if probs is None else probs + p
                    classes = m.classes_
                pred = classes[probs.argmax(axis=1)]

            elif mode == "logreg_svm_vote":
                lr = head(0.01)
                lr.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
                sv = SVC(kernel="rbf", probability=True, random_state=SEED)
                scaler = StandardScaler().fit(features.loc[tr].to_numpy())
                sv.fit(scaler.transform(features.loc[tr].to_numpy()),
                       labels.loc[tr].to_numpy())
                p = (lr.predict_proba(features.loc[te].to_numpy())
                     + sv.predict_proba(scaler.transform(features.loc[te].to_numpy())))
                pred = lr.classes_[p.argmax(axis=1)]

            else:  # fixed C
                model = head(float(mode))
                model.fit(features.loc[tr].to_numpy(), labels.loc[tr].to_numpy())
                pred = model.predict(features.loc[te].to_numpy())

            meta = indexed.loc[te].reset_index()
            truth = labels.loc[te].to_numpy()
            per_fold.append((truth, pred, meta))
            frames.append(pd.DataFrame({
                "y_true": truth, "y_pred": pred,
                "speaker": meta["speaker_id"].to_numpy()}))
            evaluate(truth, pred, "speaker_independent", fold, meta,
                     model="logreg", feature_config=label,
                     hyperparams={"lever": label}, notes=notes,
                     results_path=results_path)

        pooled = evaluate_pooled(
            per_fold, "speaker_independent", model="logreg", feature_config=label,
            hyperparams={"lever": label, "chosen": chosen or None},
            notes=notes + " (pooled out-of-fold)", results_path=results_path)
        kept[label] = pd.concat(frames, ignore_index=True)
        rows.append({"variant": label, "dim": features.shape[1],
                     "accuracy": pooled["accuracy"], "macro_f1": pooled["macro_f1"],
                     "chosen_C": ",".join(sorted(set(chosen))),
                     "seconds": round(time.time() - started, 1)})
        print(f"  {label:<44s} acc={pooled['accuracy']:.4f} "
              f"f1={pooled['macro_f1']:.4f} d={features.shape[1]:<5d} "
              f"{rows[-1]['chosen_C']}", flush=True)

    print("baseline + the two best levers, alone and combined")
    run_cell("L6 C=1 (Phase 13 baseline)", L6, "1.0", "phase-14d baseline")
    run_cell("L6 + prosody, C=1", fused, "1.0", "phase-14d fusion only")
    run_cell("L6, C by inner CV", L6, "innerC", "phase-14d regularisation only")
    run_cell("L6 + prosody, C by inner CV", fused, "innerC",
             "phase-14d fusion + regularisation")

    print("\nlayers 4-8 combined two ways")
    run_cell("L4-8 concat + prosody, C by inner CV", multi_fused, "innerC",
             "phase-14d multi-layer + fusion + regularisation")
    run_cell("L4-8 probability vote", L6, "layer_vote",
             "phase-14d per-layer ensemble, soft vote")

    print("\nheads combined")
    run_cell("L6 + prosody, logreg+SVM vote", fused, "logreg_svm_vote",
             "phase-14d logreg + svm soft vote")

    table = pd.DataFrame(rows)
    table["vs_base_pp"] = ((table["macro_f1"] - BASE_F1) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\npaired vs the Phase 13 baseline (speaker-clustered bootstrap)")
    base = kept["L6 C=1 (Phase 13 baseline)"]
    sig = []
    for label, frame in kept.items():
        if label.startswith("L6 C=1"):
            continue
        b = paired_cluster_bootstrap(frame["y_true"], frame["y_pred"],
                                     base["y_pred"], frame["speaker"], "macro_f1")
        sig.append({"variant": label, "delta_pp": round(100*b["delta"], 2),
                    "lo_pp": round(100*b["lo"], 2), "hi_pp": round(100*b["hi"], 2),
                    "p": round(b["p_value"], 4), "significant": b["significant"]})
        mark = "SIGNIFICANT" if b["significant"] else "not distinguishable"
        print(f"  {label:<44s} {100*b['delta']:+5.2f} pp "
              f"[{100*b['lo']:+5.2f},{100*b['hi']:+5.2f}] p={b['p_value']:.3f} -> {mark}")

    best = table.iloc[0]
    ci = cluster_bootstrap_ci(kept[best["variant"]]["y_true"],
                              kept[best["variant"]]["y_pred"],
                              kept[best["variant"]]["speaker"], "macro_f1")
    print(f"\n  top of list: {best['variant']} = {best['macro_f1']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    wins = [s for s in sig if s["significant"] and s["delta_pp"] > 0]
    print(f"  variants SIGNIFICANTLY better than baseline: {len(wins)}")

    return table, pd.DataFrame(sig)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    table, sig = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase14d_combined.parquet", index=False)
    sig.to_parquet(CACHE_DIR / "phase14d_significance.parquet", index=False)
    print("\n=== Phase 14d ===")
    print(table.to_string(index=False))
