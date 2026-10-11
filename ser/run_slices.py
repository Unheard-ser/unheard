"""Phase 14c — what the winning model actually does, not just how well.

CLAUDE.md requires a confusion matrix and slices by gender, intensity and
emotion for every result. `ser.evaluate` computes all of them and returns them,
but only accuracy and macro-F1 reach `results/results.csv` -- so for the Phase
13 winner they have been computed and discarded on every one of ~100 runs. This
driver surfaces them.

It matters beyond compliance. Phase 7 found a **12.8 pp macro-F1 gender gap,
worse on male**, on handcrafted features, and that figure is quoted in the
deck's demographic-risk slide. It has never been re-measured on the model that
is now the headline. It could have closed, or widened, and the slide is wrong
either way until someone checks.

Three questions:

1. **Fairness.** Is the gender gap still there at 0.7944, and which direction?
2. **Intensity.** RAVDESS records most emotions at normal and strong intensity.
   A model that only works on strong, theatrical delivery is not useful for
   contact-centre audio, where nobody is performing.
3. **Error structure.** Phase 7 asked whether errors cluster by arousal. The
   two-stage arousal model failed there (-1.62 pp) because the gate was only
   85.1% accurate. Whether learned features change that structure is worth
   knowing, and the confusion matrix is deck material regardless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser.embeddings import EmbeddingConfig
from ser.evaluate import LABELS, evaluate_pooled
from ser.models.classical import build, default_hyperparams
from ser.run_wav2vec2 import build_matrix, load_cached, ravdess_view
from ser.run_wav2vec2_probes import FIGURES
from ser.splits import get_split, load_folds
from ser.stats import cluster_bootstrap_ci, paired_cluster_bootstrap

#: The Phase 13 winner.
WINNER = EmbeddingConfig(layer=6, pooling="mean", normalisation="per_speaker")
MODEL = "logistic-regression"

#: Arousal grouping, matching Phase 7 so the two analyses are comparable.
HIGH_AROUSAL: frozenset[str] = frozenset(
    {"angry", "happy", "fearful", "surprised", "disgust"}
)


def pooled_predictions(results_path: Path | None = None):
    """Refit the winner across the frozen folds and collect held-out predictions."""
    tensor, index = load_cached(EmbeddingConfig(batch_size=8))
    tensor, index = ravdess_view(tensor, index)
    features, labels = build_matrix(tensor, index, WINNER)
    indexed = index.set_index("filename")

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )

    per_fold, frames = [], []
    for fold in fold_numbers:
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        estimator = build(MODEL).named_steps["estimator"]
        estimator.fit(
            features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy()
        )
        predicted = estimator.predict(features.loc[test_files].to_numpy())
        meta = indexed.loc[test_files].reset_index()
        per_fold.append((labels.loc[test_files].to_numpy(), predicted, meta))
        frames.append(meta.assign(
            y_true=labels.loc[test_files].to_numpy(), y_pred=predicted
        ))

    result = evaluate_pooled(
        per_fold, "speaker_independent",
        model=MODEL, feature_config=WINNER.name,
        hyperparams=default_hyperparams(MODEL),
        notes="phase-14c slice analysis (pooled out-of-fold)",
        results_path=results_path,
    )
    return result, pd.concat(frames, ignore_index=True)


def slice_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Accuracy and macro-F1 within each level of ``column``, with intervals."""
    from sklearn.metrics import f1_score

    rows = []
    for level in sorted(frame[column].dropna().unique()):
        sub = frame[frame[column] == level]
        interval = cluster_bootstrap_ci(
            sub["y_true"], sub["y_pred"], sub["speaker_id"], "macro_f1"
        )
        rows.append({
            column: level,
            "n": len(sub),
            "accuracy": float((sub["y_true"] == sub["y_pred"]).mean()),
            "macro_f1": float(
                f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
            ),
            "f1_lo": interval["lo"],
            "f1_hi": interval["hi"],
        })
    return pd.DataFrame(rows)


def arousal_confusion(frame: pd.DataFrame) -> dict[str, float]:
    """Do errors respect the arousal boundary?

    If most confusions stay within an arousal group, the emotion space is
    organised by arousal and a two-stage model has something to exploit. Phase 7
    tried exactly that on handcrafted features and it failed.
    """
    wrong = frame[frame["y_true"] != frame["y_pred"]]
    same_arousal = (
        wrong["y_true"].isin(HIGH_AROUSAL) == wrong["y_pred"].isin(HIGH_AROUSAL)
    )
    return {
        "n_errors": len(wrong),
        "within_arousal": float(same_arousal.mean()),
        "n_high": int(frame["y_true"].isin(HIGH_AROUSAL).sum()),
        "n_low": int((~frame["y_true"].isin(HIGH_AROUSAL)).sum()),
    }


def plot_confusion(matrix: pd.DataFrame):
    """Row-normalised confusion matrix. Returns a Figure."""
    import matplotlib.pyplot as plt

    normalised = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(normalised.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("wav2vec2 L6 + per-speaker + linear head\nrow-normalised, pooled out-of-fold")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = normalised.iloc[i, j]
            if value >= 0.01:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.5 else "black",
                )
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    return fig


def plot_gender_gap(by_gender: pd.DataFrame):
    """Per-gender macro-F1 with intervals, against Phase 7's handcrafted gap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(by_gender))
    errors = np.vstack([
        by_gender["macro_f1"] - by_gender["f1_lo"],
        by_gender["f1_hi"] - by_gender["macro_f1"],
    ])
    ax.bar(positions, by_gender["macro_f1"], yerr=errors, capsize=6, width=0.5)
    ax.set_xticks(positions, by_gender["gender"])
    ax.set_ylabel("macro-F1 (95% cluster bootstrap)")
    ax.set_title("Gender fairness of the winning model")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def run(results_path: Path | None = None):
    set_seeds()
    result, frame = pooled_predictions(results_path)
    print(f"winner: acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f}\n")

    print("=== per class ===")
    per_class = result["per_class"]
    print(per_class.to_string())

    print("\n=== by gender ===")
    by_gender = slice_table(frame, "gender")
    print(by_gender.to_string(index=False))
    if len(by_gender) == 2:
        male = frame[frame["gender"] == "male"]
        female = frame[frame["gender"] == "female"]
        # Not paired -- different clips -- so intervals are compared, not a delta.
        gap = abs(float(by_gender["macro_f1"].iloc[0] - by_gender["macro_f1"].iloc[1]))
        print(f"\n  gender gap: {100*gap:.2f} pp   (Phase 7 handcrafted: 12.8 pp, worse on male)")
        print(f"  male n={len(male)}, female n={len(female)}")

    print("\n=== by intensity ===")
    by_intensity = slice_table(frame, "intensity_label")
    print(by_intensity.to_string(index=False))

    # Macro-F1 across the two intensity slices is NOT comparable as printed.
    # RAVDESS has no strong-intensity neutral, so the strong slice holds 7
    # classes; macro-F1 over the project's fixed 8 labels scores the absent
    # class 0 and divides by 8, understating strong by a factor of 7/8. The
    # honest comparison restricts both slices to the 7 shared classes.
    shared = sorted(set(frame.loc[frame["intensity_label"] == "strong", "y_true"]))
    print(f"\n  restricted to the {len(shared)} classes present at both intensities:")
    fair = []
    for level in ("normal", "strong"):
        sub = frame[
            (frame["intensity_label"] == level) & (frame["y_true"].isin(shared))
        ]
        interval = cluster_bootstrap_ci(
            sub["y_true"], sub["y_pred"], sub["speaker_id"], "macro_f1"
        )
        fair.append({
            "intensity": level, "n": len(sub),
            "accuracy": float((sub["y_true"] == sub["y_pred"]).mean()),
            "macro_f1": interval["point"],
            "f1_lo": interval["lo"], "f1_hi": interval["hi"],
        })
    fair_table = pd.DataFrame(fair)
    print(fair_table.to_string(index=False))
    by_intensity = pd.concat(
        [by_intensity.assign(scope="all 8 labels"),
         fair_table.rename(columns={"intensity": "intensity_label"}).assign(
             scope="7 shared labels")],
        ignore_index=True,
    )

    print("\n=== error structure ===")
    arousal = arousal_confusion(frame)
    print(
        f"  {arousal['n_errors']} errors; {arousal['within_arousal']:.1%} stay within "
        f"the same arousal group"
    )
    print(f"  (Phase 7's two-stage arousal model FAILED at -1.62 pp on handcrafted)")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_confusion(result["confusion_matrix"]).savefig(
        FIGURES / "21_confusion_wav2vec2.png", dpi=150
    )
    plot_gender_gap(by_gender).savefig(
        FIGURES / "22_gender_fairness_wav2vec2.png", dpi=150
    )
    print(f"\nwrote figures 21 and 22 to {FIGURES}")

    return result, frame, by_gender, by_intensity, per_class, arousal


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    result, frame, by_gender, by_intensity, per_class, arousal = run(
        results_path=args.results
    )
    out = Path("features")
    out.mkdir(exist_ok=True)
    by_gender.to_parquet(out / "phase14c_gender.parquet", index=False)
    by_intensity.to_parquet(out / "phase14c_intensity.parquet", index=False)
    per_class.to_parquet(out / "phase14c_per_class.parquet")
    result["confusion_matrix"].to_parquet(out / "phase14c_confusion.parquet")

    print("\n=== confusion matrix (counts) ===")
    print(result["confusion_matrix"].to_string())
