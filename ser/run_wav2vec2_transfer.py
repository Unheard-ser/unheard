"""Phase 13b — does the merged corpus help once the features are learned?

`docs/merged_baseline_findings.md` closed with a bounded claim: naive pooling,
corpus subsetting and four covariate-shift remedies all fail *on handcrafted
features with shallow classifiers*, and the damage tracks the volume of
out-of-domain data at r = -0.98. It explicitly left one hypothesis untested --
that the external corpora help **representation learning** rather than
classification.

wav2vec2 is the test. Its encoder is already pretrained on ~960 hours of
out-of-domain speech, orders of magnitude more than this 12,162-clip merge, so
the features it produces should be far less sensitive to the domain of the
training clips laid on top. If the volume effect that dominated Phase 12 is a
property of brittle handcrafted features, it should shrink or vanish here. If
it survives, it is a property of the corpora themselves.

Same questions, same frozen RAVDESS test folds, same Architecture A -- only the
feature source changes. That is what makes the comparison against Phase 12
meaningful.
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ser import SEED, set_seeds
from ser import corpora
from ser.embeddings import EmbeddingConfig
from ser.evaluate import evaluate_pooled
from ser.features import CACHE_DIR
from ser.run_wav2vec2 import build_matrix, load_cached
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, mcnemar, paired_cluster_bootstrap

EXTERNAL: tuple[str, ...] = ("CREMA-D", "TESS", "SAVEE")

#: The Phase 13 winner. Fixed across every row so the comparison isolates the
#: training data rather than the feature configuration.
WINNER = EmbeddingConfig(layer=6, pooling="mean", normalisation="per_speaker")

#: External-clip weights, matching Phase 12b so the two are comparable.
WEIGHT_GRID: tuple[float, ...] = (1.0, 0.1, 0.01)


def linear_head():
    """A linear head that scales to 12,000 training clips.

    Phase 13's winner used `LogisticRegression(max_iter=2000)`, which is fine on
    1,140 RAVDESS clips but not here: lbfgs on 8,882 x 768 with 8 classes takes
    roughly 35 minutes for a single fit, and this phase needs 50 of them.
    `LinearSVC` fits the same clip set in 41 seconds -- measured, not assumed --
    and is the same kind of model: a linear decision boundary on frozen
    embeddings.

    The swap costs nothing in validity because **every cell in this phase uses
    it, including the RAVDESS-only control**. The comparison is internal: each
    row differs from the control only by which training clips were added. It
    does mean this phase's control will not be numerically identical to Phase
    13's 0.7944, and that number is reported here rather than assumed.

    Phase 13 also established that the linear and RBF heads are statistically
    indistinguishable on these features (p = 0.442), so nothing rests on which
    linear solver is used.
    """
    from sklearn.svm import LinearSVC

    return LinearSVC(max_iter=2000, random_state=SEED)


def subsets() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [()]
    for size in range(1, len(EXTERNAL) + 1):
        out.extend(combinations(EXTERNAL, size))
    return out


def run(results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    base = EmbeddingConfig(batch_size=8)
    print(f"loading cached embeddings: {base.cache_path.name}", flush=True)
    tensor, index = load_cached(base)

    # Per-speaker normalisation over all 121 speakers, so a RAVDESS clip is
    # normalised identically whether or not external clips are in training.
    features, labels = build_matrix(tensor, index, WINNER)
    indexed = index.set_index("filename")
    is_external = indexed["source"].astype(str) != "RAVDESS"
    external_files = indexed.index[is_external].tolist()
    print(f"  {features.shape[0]:,} x {features.shape[1]} | {len(external_files):,} external\n", flush=True)

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )

    kept: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    def cell(label: str, extra: list[str], weight: float, notes: str) -> None:
        started = time.time()
        per_fold, collected = [], []
        for fold in fold_numbers:
            train_files, test_files = get_split("speaker_independent", fold, folds=folds)
            train = train_files + extra
            sample_weight = (
                np.where(is_external.loc[train].to_numpy(), weight, 1.0)
                if extra else None
            )
            estimator = linear_head()
            estimator.fit(
                features.loc[train].to_numpy(),
                labels.loc[train].to_numpy(),
                **({"sample_weight": sample_weight} if sample_weight is not None else {}),
            )
            predicted = estimator.predict(features.loc[test_files].to_numpy())
            meta = indexed.loc[test_files].reset_index()
            per_fold.append((labels.loc[test_files].to_numpy(), predicted, meta))
            collected.append(
                pd.DataFrame({
                    "y_true": labels.loc[test_files].to_numpy(),
                    "y_pred": predicted,
                    "speaker": meta["speaker_id"].to_numpy(),
                })
            )

        pooled = evaluate_pooled(
            per_fold, "speaker_independent",
            model="linear-svc", feature_config=WINNER.name,
            hyperparams={"max_iter": 2000, "head": "LinearSVC"},
            notes=notes + " (pooled out-of-fold)", results_path=results_path,
        )
        kept[label] = pd.concat(collected, ignore_index=True)
        rows.append({
            "variant": label,
            "n_train_fold0": len(fold_numbers and get_split("speaker_independent", 0, folds=folds)[0]) + len(extra),
            "accuracy": pooled["accuracy"],
            "macro_f1": pooled["macro_f1"],
            "seconds": round(time.time() - started, 1),
        })
        print(
            f"  {label:<34s} acc={pooled['accuracy']:.4f} f1={pooled['macro_f1']:.4f} "
            f"({rows[-1]['seconds']}s)",
            flush=True,
        )

    # --- corpus-subset sweep, mirroring Phase 12 ---------------------------
    print("corpus-subset sweep (all external clips at full weight)")
    for sources in subsets():
        label = "+".join(sources) if sources else "(ravdess only)"
        extra = (
            indexed.index[indexed["source"].astype(str).isin(sources)].tolist()
            if sources else []
        )
        cell(label, extra, 1.0, f"phase-13b wav2vec2 subset {label}")

    # --- down-weighting, mirroring Phase 12b -------------------------------
    print("\nuniform down-weighting of all external clips")
    for weight in WEIGHT_GRID:
        cell(
            f"all external w={weight:g}", external_files, weight,
            f"phase-13b wav2vec2 all external, weight {weight:g}",
        )

    table = pd.DataFrame(rows)
    control = float(table.loc[table["variant"] == "(ravdess only)", "macro_f1"].iloc[0])
    table["vs_control_pp"] = ((table["macro_f1"] - control) * 100).round(2)
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # --- significance against the control ----------------------------------
    print("\npaired differences vs the RAVDESS-only control")
    baseline = kept["(ravdess only)"]
    interval = cluster_bootstrap_ci(
        baseline["y_true"], baseline["y_pred"], baseline["speaker"], "macro_f1"
    )
    print(
        f"  control macro-F1 {interval['point']:.4f} "
        f"[{interval['lo']:.4f}, {interval['hi']:.4f}]",
        flush=True,
    )

    comparisons = []
    for label, frame in kept.items():
        if label == "(ravdess only)":
            continue
        boot = paired_cluster_bootstrap(
            frame["y_true"], frame["y_pred"], baseline["y_pred"], frame["speaker"],
            "macro_f1",
        )
        exact = mcnemar(frame["y_true"], frame["y_pred"], baseline["y_pred"])
        comparisons.append({
            "variant": label,
            "delta_pp": round(100 * boot["delta"], 2),
            "lo_pp": round(100 * boot["lo"], 2),
            "hi_pp": round(100 * boot["hi"], 2),
            "boot_p": round(boot["p_value"], 4),
            "significant": boot["significant"],
            "mcnemar_p": float(f"{exact['p_value']:.3g}"),
        })
        verdict = "SIGNIFICANT" if boot["significant"] else "not distinguishable"
        print(
            f"  {label:<34s} {100*boot['delta']:+6.2f} pp "
            f"[{100*boot['lo']:+6.2f}, {100*boot['hi']:+6.2f}] "
            f"p={boot['p_value']:.4f} -> {verdict}",
            flush=True,
        )

    return table, pd.DataFrame(comparisons)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table, significance = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase13b_transfer.parquet", index=False)
    significance.to_parquet(CACHE_DIR / "phase13b_significance.parquet", index=False)
    print("\n=== wav2vec2 + external data, frozen RAVDESS test folds ===")
    print(table.to_string(index=False))
    print("\n=== paired vs control ===")
    print(significance.to_string(index=False))
