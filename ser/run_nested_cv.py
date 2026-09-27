"""Phase 17 — how much of 0.7944 is selection bias?

Phase 13 chose the winning configuration — layer, pooling, normalisation, head —
by comparing **pooled test-fold scores** across roughly eighteen candidates and
taking the best. That is selection on the test set. The reported number is
therefore the maximum of eighteen noisy estimates, and the maximum of a noisy
set is optimistically biased even when every individual estimate is unbiased.

Phase 14a was written with this hazard in mind and said so explicitly about its
own sweep. Phase 13 has the same flaw and it went unremarked, which is worse
because 0.7944 is the number the whole project reports.

**Nested cross-validation is the fix.** The configuration is chosen by an inner,
speaker-grouped cross-validation *inside each training fold*, and then applied
once to the outer held-out fold. The outer fold never participates in the
choice, so the resulting score is an unbiased estimate of "run this selection
procedure on new data".

Two numbers come out, and the difference between them is the bias:

``selected_on_test``  the Phase 13 procedure — pick the best by outer score
``nested``            the honest procedure — pick inside, score outside

If they are close, the headline stands. If nested is materially lower, the
headline needs restating.
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ser import SEED, set_seeds
from ser.embeddings import EmbeddingConfig, select
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR, per_speaker_normalise
from ser.run_wav2vec2 import load_cached, ravdess_view
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

#: The candidate space. Deliberately the axes Phase 13 actually swept, so the
#: comparison is against that procedure rather than an invented one. Heads are
#: restricted to logistic regression because Phase 13 axis 4 found the linear
#: and RBF heads statistically indistinguishable (p = 0.442), and 20 candidates
#: keep the nested loop affordable.
LAYERS: tuple[int, ...] = (3, 5, 6, 7, 9)
POOLINGS: tuple[str, ...] = ("mean", "mean_std")
NORMS: tuple[str, ...] = ("global", "per_speaker")
INNER_SPLITS = 3


def candidates() -> list[EmbeddingConfig]:
    return [
        EmbeddingConfig(layer=l, pooling=p, normalisation=n)
        for l, p, n in product(LAYERS, POOLINGS, NORMS)
    ]


def matrices(tensor, index) -> dict[tuple, pd.DataFrame]:
    """Precompute every candidate's feature matrix once.

    Twenty slices of a cached tensor, built once and reused across 5 outer x 3
    inner folds. Rebuilding them inside the loop would dominate the runtime.
    """
    out = {}
    speakers = index["speaker_id"].to_numpy()
    for cfg in candidates():
        block = select(tensor, cfg)
        if cfg.normalisation == "per_speaker":
            block = per_speaker_normalise(block, speakers)
        out[(cfg.layer, cfg.pooling, cfg.normalisation)] = pd.DataFrame(
            block, index=index["filename"]
        )
    return out


def fit_predict(train, test, feats, labels, standardise: bool):
    """Logistic head, with global standardisation fitted on the training rows."""
    x_tr, x_te = feats.loc[train].to_numpy(), feats.loc[test].to_numpy()
    if standardise:
        scaler = StandardScaler().fit(x_tr)
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
    model = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED)
    model.fit(x_tr, labels.loc[train].to_numpy())
    return model.predict(x_te)


def run(results_path: Path | None = None):
    set_seeds()
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    tensor, index = ravdess_view(tensor, index)
    indexed = index.set_index("filename")
    labels = pd.Series(index["emotion_label"].to_numpy(), index=index["filename"])
    speakers = indexed["speaker_id"]

    print(f"building {len(candidates())} candidate matrices ...", flush=True)
    mats = matrices(tensor, index)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique())

    # ---- A. the Phase 13 procedure: score every candidate on the test folds
    print("\nA. selected on test (the Phase 13 procedure)")
    outer = []
    for key, feats in mats.items():
        standardise = key[2] == "global"
        truths, preds = [], []
        for fold in fold_numbers:
            tr, te = get_split("speaker_independent", fold, folds=folds)
            preds.extend(fit_predict(tr, te, feats, labels, standardise))
            truths.extend(labels.loc[te].to_numpy())
        outer.append({
            "layer": key[0], "pooling": key[1], "normalisation": key[2],
            "macro_f1": f1_score(truths, preds, average="macro", zero_division=0),
        })
    outer_table = pd.DataFrame(outer).sort_values("macro_f1", ascending=False)
    top = outer_table.iloc[0]
    print(outer_table.head(5).to_string(index=False))
    print(f"  -> best by outer score: L{int(top.layer)} {top.pooling} "
          f"{top.normalisation} = {top.macro_f1:.4f}")

    # ---- B. nested: choose inside the training fold, score outside ---------
    print("\nB. nested CV (choice made inside each training fold)")
    per_fold, frames, chosen = [], [], []
    started = time.time()
    for fold in fold_numbers:
        tr, te = get_split("speaker_independent", fold, folds=folds)
        names = np.asarray(tr)
        y_tr = labels.loc[tr].to_numpy()
        inner = list(GroupKFold(n_splits=INNER_SPLITS).split(
            names, y_tr, speakers.loc[tr].to_numpy()))

        best_key, best_score = None, -np.inf
        for key, feats in mats.items():
            standardise = key[2] == "global"
            scores = []
            for a, b in inner:
                itr, ite = list(names[a]), list(names[b])
                pred = fit_predict(itr, ite, feats, labels, standardise)
                scores.append(f1_score(labels.loc[ite].to_numpy(), pred,
                                       average="macro", zero_division=0))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_key, best_score = key, mean

        chosen.append(f"L{best_key[0]}-{best_key[1]}-{best_key[2]}")
        pred = fit_predict(tr, te, mats[best_key], labels, best_key[2] == "global")
        truth = labels.loc[te].to_numpy()
        meta = indexed.loc[te].reset_index()
        per_fold.append((truth, pred, meta))
        frames.append(pd.DataFrame({
            "y_true": truth, "y_pred": pred,
            "speaker": meta["speaker_id"].to_numpy()}))
        evaluate(truth, pred, "speaker_independent", fold, meta,
                 model="logistic-regression", feature_config="nested-cv",
                 hyperparams={"chosen": chosen[-1]},
                 notes="phase-17 nested CV", results_path=results_path)
        print(f"  fold {fold}: inner CV chose {chosen[-1]}", flush=True)

    nested = evaluate_pooled(
        per_fold, "speaker_independent", model="logistic-regression",
        feature_config="nested-cv", hyperparams={"chosen": chosen},
        notes="phase-17 nested CV (pooled out-of-fold)", results_path=results_path)

    allp = pd.concat(frames, ignore_index=True)
    ci = cluster_bootstrap_ci(allp.y_true, allp.y_pred, allp.speaker, "macro_f1")

    bias = 100 * (float(top.macro_f1) - nested["macro_f1"])
    print(f"\n  selected-on-test : {top.macro_f1:.4f}")
    print(f"  nested (honest)  : {nested['macro_f1']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"  optimism         : {bias:+.2f} pp")
    print(f"  chosen per fold  : {chosen}")
    print(f"  ({time.time() - started:.0f}s)")

    summary = pd.DataFrame([
        {"procedure": "selected on test (Phase 13)", "macro_f1": float(top.macro_f1),
         "lo": np.nan, "hi": np.nan},
        {"procedure": "nested CV (unbiased)", "macro_f1": nested["macro_f1"],
         "lo": ci["lo"], "hi": ci["hi"]},
    ])
    summary["optimism_pp"] = [round(bias, 2), 0.0]
    return summary, outer_table, allp


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    summary, outer_table, _ = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    summary.to_parquet(CACHE_DIR / "phase17_nested_cv.parquet", index=False)
    outer_table.to_parquet(CACHE_DIR / "phase17_candidates.parquet", index=False)
    print("\n=== Phase 17 ===")
    print(summary.to_string(index=False))
