"""Phase 12b — applying the external corpora in a way that actually helps.

`docs/merged_baseline_findings.md` established that pooling the external
corpora into training costs 11.11 pp macro-F1, that the damage scales with the
*volume* of external data (r = -0.98), and that `neutral` (+14.6 pp) and
`fearful` (+8.3 pp) nevertheless improved. That is not "the data is useless" --
it is a covariate-shift problem being handled by ignoring it.

Four strategies, each attacking a different part of that diagnosis:

``uniform``
    Down-weight every external clip by a constant. Directly counters the volume
    effect: 10,722 external clips stop outvoting 1,140 in-domain ones.

``importance``
    The principled version. Fit a discriminator to separate RAVDESS-train from
    external clips, then weight each external clip by its density ratio
    ``p(RAVDESS|x) / (1 - p(RAVDESS|x))`` -- textbook covariate-shift
    correction. An external clip that looks like RAVDESS counts nearly fully;
    one that does not counts for almost nothing. Motivated directly by the
    corpus probe: raw features identify the corpus 91.8% of the time, so that
    discriminator is easy to fit and its output is informative.

``class_selective``
    Add external clips only for the classes they help. **The class set is
    chosen by inner cross-validation inside the training fold**, never from the
    per-class deltas in the findings doc -- those were measured on the test
    folds, and selecting on them would be fitting to test data (CLAUDE.md
    rule 2).

``scaler_only``
    Use the external clips to fit the feature scaler, but train the classifier
    on RAVDESS alone. Tests whether the extra data helps the *representation*
    without polluting the decision boundary.

Every strategy is evaluated on the frozen RAVDESS test folds (Architecture A),
so all of it is comparable to the 0.6311 control and to each other.

Hyperparameter honesty
----------------------
For ``uniform``, the full weight sweep is reported for insight, **and**
separately a weight chosen by inner CV inside each training fold is applied to
the outer test fold. Only the second number is a claim; the first is a curve.
Reporting only the best point of a sweep evaluated on test would be choosing a
hyperparameter on the test set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ser import SEED, set_seeds
from ser import corpora
from ser.evaluate import evaluate, evaluate_pooled
from ser.features import CACHE_DIR
from ser.run_early_read import EXTERNAL, MODEL, prepare_merged
from ser.run_phase5 import WINNER
from ser.splits import get_split, load_folds
from ser.splits_merged import augmented_train

#: Weights swept for the ``uniform`` strategy. 1.0 reproduces the naive pooling
#: that lost 11.11 pp; 0.0 would be the RAVDESS-only control.
WEIGHT_GRID: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)

#: Inner folds used to choose a weight or a class set without touching test.
INNER_SPLITS = 3


def _pipeline(class_weight: str | None = None) -> Pipeline:
    """Scaler + SVM-RBF, matching `ser.models.classical.build` plus class_weight.

    Built here rather than reusing ``build`` only because these experiments
    need ``class_weight``; every other parameter is the library default, as in
    Phase 5.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", SVC(kernel="rbf", random_state=SEED, class_weight=class_weight)),
        ]
    )


def _fit_predict(
    train_files: list[str],
    test_files: list[str],
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: np.ndarray | None = None,
    class_weight: str | None = None,
) -> np.ndarray:
    """Fit on the training files and predict the test files."""
    pipeline = _pipeline(class_weight)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["estimator__sample_weight"] = sample_weight
    pipeline.fit(
        features.loc[train_files].to_numpy(),
        labels.loc[train_files].to_numpy(),
        **fit_kwargs,
    )
    return pipeline.predict(features.loc[test_files].to_numpy())


def _macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(true, pred, average="macro", zero_division=0))


# --- strategy: uniform down-weighting --------------------------------------


def _uniform_weights(
    train_files: list[str], is_external: pd.Series, weight: float
) -> np.ndarray:
    """1.0 for RAVDESS clips, ``weight`` for external ones."""
    return np.where(is_external.loc[train_files].to_numpy(), weight, 1.0)


def select_weight_by_inner_cv(
    ravdess_train: list[str],
    external_files: list[str],
    features: pd.DataFrame,
    labels: pd.Series,
    speakers: pd.Series,
    is_external: pd.Series,
) -> float:
    """Choose an external-clip weight using only the training fold.

    The inner split is grouped by speaker, like the outer one, so the chosen
    weight is not itself propped up by speaker leakage.
    """
    groups = speakers.loc[ravdess_train].to_numpy()
    inner = list(
        GroupKFold(n_splits=INNER_SPLITS).split(
            np.zeros(len(ravdess_train)), labels.loc[ravdess_train].to_numpy(), groups
        )
    )
    names = np.asarray(ravdess_train)

    best_weight, best_score = 0.0, -np.inf
    for weight in (0.0, *WEIGHT_GRID):
        scores = []
        for inner_train, inner_test in inner:
            train = list(names[inner_train])
            if weight > 0:
                train = train + external_files
            sample_weight = (
                _uniform_weights(train, is_external, weight) if weight > 0 else None
            )
            predicted = _fit_predict(
                train, list(names[inner_test]), features, labels, sample_weight
            )
            scores.append(
                _macro_f1(labels.loc[list(names[inner_test])].to_numpy(), predicted)
            )
        mean = float(np.mean(scores))
        if mean > best_score:
            best_weight, best_score = weight, mean
    return best_weight


# --- strategy: importance weighting ----------------------------------------


def importance_weights(
    ravdess_train: list[str],
    external_files: list[str],
    features: pd.DataFrame,
    clip: float = 10.0,
) -> np.ndarray:
    """Density-ratio weights for external clips, fit on training data only.

    A logistic discriminator separates RAVDESS-train from external clips; each
    external clip's weight is ``p / (1 - p)`` where ``p`` is its predicted
    probability of being RAVDESS. Clips that look in-domain keep their vote;
    clips that do not are nearly silenced.

    Args:
        ravdess_train: RAVDESS training filenames.
        external_files: External training filenames.
        features: Feature matrix keyed by filename.
        clip: Upper bound on a single clip's weight, so one outlier cannot
            dominate the fold.

    Returns:
        Weights aligned to ``ravdess_train + external_files``: 1.0 for the
        RAVDESS block, the density ratio for the external block.
    """
    from sklearn.linear_model import LogisticRegression

    train = ravdess_train + external_files
    target = np.concatenate(
        [np.ones(len(ravdess_train)), np.zeros(len(external_files))]
    )
    discriminator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    discriminator.fit(features.loc[train].to_numpy(), target)

    probability = discriminator.predict_proba(
        features.loc[external_files].to_numpy()
    )[:, 1]
    ratio = probability / np.clip(1.0 - probability, 1e-6, None)
    return np.concatenate([np.ones(len(ravdess_train)), np.clip(ratio, 0.0, clip)])


# --- strategy: class-selective ---------------------------------------------


def select_classes_by_inner_cv(
    ravdess_train: list[str],
    external_files: list[str],
    features: pd.DataFrame,
    labels: pd.Series,
    speakers: pd.Series,
) -> list[str]:
    """Pick the classes whose external clips help, judged inside the fold.

    Two inner runs -- with and without the external clips -- give a per-class
    recall delta measured entirely on training-fold data. Classes with a
    positive delta keep their external clips; the rest are dropped.

    This is the leak-free counterpart of the per-class table in
    `docs/merged_baseline_findings.md`, which was measured on the test folds
    and therefore cannot be used to make this decision.
    """
    from sklearn.metrics import classification_report

    groups = speakers.loc[ravdess_train].to_numpy()
    names = np.asarray(ravdess_train)
    inner = list(
        GroupKFold(n_splits=INNER_SPLITS).split(
            np.zeros(len(ravdess_train)), labels.loc[ravdess_train].to_numpy(), groups
        )
    )

    recalls: dict[str, dict[str, list[float]]] = {"with": {}, "without": {}}
    for tag, add_external in (("without", False), ("with", True)):
        trues, preds = [], []
        for inner_train, inner_test in inner:
            train = list(names[inner_train])
            if add_external:
                train = train + external_files
            test = list(names[inner_test])
            preds.extend(_fit_predict(train, test, features, labels))
            trues.extend(labels.loc[test].to_numpy())
        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        for label in corpora.CANONICAL:
            if label in report:
                recalls[tag].setdefault(label, []).append(report[label]["recall"])

    helped = [
        label
        for label in corpora.CANONICAL
        if recalls["with"].get(label, [0.0])[0] > recalls["without"].get(label, [0.0])[0]
    ]
    return helped


# --- strategy: scaler only --------------------------------------------------


def scaler_only_predict(
    ravdess_train: list[str],
    external_files: list[str],
    test_files: list[str],
    features: pd.DataFrame,
    labels: pd.Series,
) -> np.ndarray:
    """Fit the scaler on all training data, the classifier on RAVDESS alone.

    Both blocks are training data, so nothing is fit on test.
    """
    scaler = StandardScaler().fit(
        features.loc[ravdess_train + external_files].to_numpy()
    )
    estimator = SVC(kernel="rbf", random_state=SEED)
    estimator.fit(
        scaler.transform(features.loc[ravdess_train].to_numpy()),
        labels.loc[ravdess_train].to_numpy(),
    )
    return estimator.predict(scaler.transform(features.loc[test_files].to_numpy()))


# --- driver -----------------------------------------------------------------


def run(results_path: Path | None = None) -> pd.DataFrame:
    set_seeds()
    index = corpora.load_index()
    indexed, features, labels = prepare_merged(index)
    speakers = indexed["speaker_id"]
    is_external = indexed["source"].astype(str) != "RAVDESS"

    folds = load_folds()
    fold_numbers = sorted(
        folds.loc[folds["protocol"] == "speaker_independent", "fold"].unique()
    )
    external_files = indexed.index[is_external].tolist()

    rows: list[dict[str, object]] = []

    def score(tag: str, predict_fn, notes: str) -> dict[str, float]:
        started = time.time()
        per_fold = []
        for fold in fold_numbers:
            ravdess_train, test_files = get_split("speaker_independent", fold, folds=folds)
            predicted = predict_fn(ravdess_train, test_files, fold)
            test_meta = indexed.loc[test_files].reset_index()
            evaluate(
                labels.loc[test_files].to_numpy(), predicted,
                "speaker_independent", fold, test_meta,
                model=MODEL, feature_config=WINNER.name,
                hyperparams={"strategy": tag},
                notes=notes, results_path=results_path,
            )
            per_fold.append((labels.loc[test_files].to_numpy(), predicted, test_meta))
        pooled = evaluate_pooled(
            per_fold, "speaker_independent",
            model=MODEL, feature_config=WINNER.name,
            hyperparams={"strategy": tag},
            notes=notes + " (pooled out-of-fold)", results_path=results_path,
        )
        out = {
            "accuracy": pooled["accuracy"],
            "macro_f1": pooled["macro_f1"],
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"  {tag:<34s} acc={out['accuracy']:.4f} f1={out['macro_f1']:.4f} "
            f"({out['seconds']}s)",
            flush=True,
        )
        return out

    # --- control -----------------------------------------------------------
    print("control")
    rows.append({"strategy": "ravdess only (control)", **score(
        "control",
        lambda tr, te, f: _fit_predict(tr, te, features, labels),
        "phase-12b control: ravdess only",
    )})

    # --- uniform weight sweep (a curve, not a claim) ------------------------
    print("\nuniform down-weighting -- sweep, reported for insight")
    for weight in WEIGHT_GRID:
        def predict(tr, te, f, w=weight):
            train = tr + external_files
            return _fit_predict(
                train, te, features, labels, _uniform_weights(train, is_external, w)
            )
        rows.append({"strategy": f"uniform w={weight:g}", **score(
            f"uniform w={weight:g}", predict,
            f"phase-12b uniform external weight {weight:g}",
        )})

    # --- uniform, weight chosen by inner CV (the honest claim) --------------
    print("\nuniform down-weighting -- weight chosen by inner CV per fold")
    chosen: dict[int, float] = {}

    def predict_inner(tr, te, fold):
        weight = select_weight_by_inner_cv(
            tr, external_files, features, labels, speakers, is_external
        )
        chosen[fold] = weight
        if weight == 0.0:
            return _fit_predict(tr, te, features, labels)
        train = tr + external_files
        return _fit_predict(
            train, te, features, labels, _uniform_weights(train, is_external, weight)
        )

    rows.append({"strategy": "uniform (inner-CV weight)", **score(
        "uniform (inner-CV weight)", predict_inner,
        "phase-12b uniform, weight selected by inner CV",
    )})
    print(f"    weights chosen per fold: {chosen}")

    # --- importance weighting ----------------------------------------------
    print("\nimportance weighting (covariate shift)")

    def predict_importance(tr, te, f):
        train = tr + external_files
        weights = importance_weights(tr, external_files, features)
        return _fit_predict(train, te, features, labels, weights)

    rows.append({"strategy": "importance weighting", **score(
        "importance weighting", predict_importance,
        "phase-12b density-ratio importance weighting",
    )})

    # --- class-selective ----------------------------------------------------
    print("\nclass-selective (classes chosen by inner CV)")
    picked: dict[int, list[str]] = {}

    def predict_selective(tr, te, fold):
        helped = select_classes_by_inner_cv(
            tr, external_files, features, labels, speakers
        )
        picked[fold] = helped
        keep = [
            name for name in external_files
            if labels.loc[name] in set(helped)
        ]
        train = tr + keep
        return _fit_predict(train, te, features, labels)

    rows.append({"strategy": "class-selective (inner CV)", **score(
        "class-selective (inner CV)", predict_selective,
        "phase-12b class-selective external data, classes from inner CV",
    )})
    for fold, labels_kept in picked.items():
        print(f"    fold {fold}: kept external for {labels_kept}")

    # --- scaler only --------------------------------------------------------
    print("\nexternal data for the scaler only")
    rows.append({"strategy": "scaler-only", **score(
        "scaler-only",
        lambda tr, te, f: scaler_only_predict(
            tr, external_files, te, features, labels
        ),
        "phase-12b external clips fit the scaler, classifier on ravdess only",
    )})

    table = pd.DataFrame(rows)
    control = float(table.loc[table["strategy"].str.contains("control"), "macro_f1"].iloc[0])
    table["vs_control_pp"] = ((table["macro_f1"] - control) * 100).round(2)
    return table.sort_values("macro_f1", ascending=False)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    table = run(results_path=args.results)
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_parquet(CACHE_DIR / "phase12b_transfer.parquet", index=False)
    print("\n=== applying the external corpora (frozen RAVDESS test folds) ===")
    print(table.to_string(index=False))
