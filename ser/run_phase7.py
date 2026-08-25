"""Phase 7 — novelty experiments.

Five experiments, each with a **stated hypothesis, a result, and an
interpretation — including the ones that fail**. A negative result that is
correctly measured is worth more than a high number that is not reproducible
(CLAUDE.md, "what good output looks like").

Everything reuses the frozen folds and the single scoring function. Nothing
here invents a new evaluation path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ser import set_seeds
from ser.evaluate import LABELS, evaluate_pooled
from ser.features import (
    FeatureConfig,
    extract,
    feature_columns,
    per_speaker_normalise,
)
from ser.metadata import load as load_metadata
from ser.models.classical import build
from ser.splits import get_split, load_folds

#: High-arousal emotions, for the two-stage experiment. Grouped on the
#: prosody measured in Phase 2 (F0 and RMS), not on intuition.
HIGH_AROUSAL: frozenset[str] = frozenset({"angry", "happy", "fearful", "surprised"})

WINNER = FeatureConfig(
    normalisation="per_speaker",
    n_mfcc=20,
    extras=("chroma", "spectral_contrast", "spectral_centroid", "zcr"),
)

MODEL = "svm-rbf"


def _load(normalise: bool):
    """Features with per-speaker normalisation on or off."""
    meta = load_metadata()
    frame = extract(WINNER, meta)
    columns = feature_columns(frame, meta)
    indexed = frame.set_index("filename")
    values = indexed[columns].to_numpy()
    if normalise:
        values = per_speaker_normalise(values, indexed["actor"].to_numpy())
    return meta, indexed, pd.DataFrame(values, index=indexed.index, columns=columns)


# --- 1. Leakage diagnosis ---------------------------------------------------


def experiment_leakage_diagnosis(folds: pd.DataFrame) -> pd.DataFrame:
    """Can a model predict WHO is speaking from these features?

    Hypothesis: if the features encode speaker identity, a classifier should
    recover the actor far above the 1/24 = 4.2% chance rate. If per-speaker
    normalisation removes identity, that accuracy should collapse.

    Actor identity cannot be tested speaker-independently -- a held-out actor
    has no training examples of themselves -- so this uses a stratified split
    over actors, which is the appropriate design for this question.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold

    rows = []
    for normalise in (False, True):
        _, indexed, features = _load(normalise)
        actors = indexed["actor"].astype(str).to_numpy()
        scores = cross_val_score(
            build(MODEL), features.to_numpy(), actors,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1,
        )
        rows.append({
            "features": "per-speaker normalised" if normalise else "raw",
            "actor_id_accuracy": float(scores.mean()),
            "sd": float(scores.std(ddof=1)),
            "chance": 1 / 24,
        })
    table = pd.DataFrame(rows)
    table["x_chance"] = (table["actor_id_accuracy"] / table["chance"]).round(1)
    return table


# --- 2. Lexical invariance --------------------------------------------------


def experiment_lexical_invariance(folds: pd.DataFrame) -> pd.DataFrame:
    """Does emotion recognition survive a change of sentence?

    Hypothesis: if the model reads prosody rather than lexical content, it
    should transfer from statement 01 to statement 02. Comparing
    ``statement_holdout`` (same speakers both sides) against
    ``statement_holdout_si`` (speakers held out too) separates prosodic
    generalisation from speaker familiarity.
    """
    _, indexed, features = _load(normalise=True)
    labels = indexed["emotion_label"]
    rows = []

    for protocol in ("statement_holdout", "statement_holdout_si", "speaker_independent"):
        fold_numbers = sorted(folds.loc[folds["protocol"] == protocol, "fold"].unique())
        per_fold = []
        for fold in fold_numbers:
            train_files, test_files = get_split(protocol, fold, folds=folds)
            pipeline = build(MODEL)
            pipeline.fit(features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy())
            predicted = pipeline.predict(features.loc[test_files].to_numpy())
            per_fold.append(
                (labels.loc[test_files].to_numpy(), predicted, indexed.loc[test_files].reset_index())
            )
        pooled = evaluate_pooled(
            per_fold, protocol, model=MODEL, feature_config=WINNER.name,
            notes="phase-7 lexical invariance",
        )
        rows.append({
            "protocol": protocol,
            "accuracy": pooled["accuracy"],
            "macro_f1": pooled["macro_f1"],
            "n": pooled["n"],
        })
    return pd.DataFrame(rows)


# --- 3, 4. Intensity and gender, from one pooled run -----------------------


def experiment_slices(folds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Intensity conditioning and gender fairness on the tuned pipeline.

    Hypotheses: strong-intensity clips are easier than normal (acted emotion
    is exaggerated); and per-gender performance differs enough to matter for
    the demographic-bias line in the business case.
    """
    _, indexed, features = _load(normalise=True)
    labels = indexed["emotion_label"]
    per_fold = []
    for fold in range(5):
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        pipeline = build(MODEL)
        pipeline.fit(features.loc[train_files].to_numpy(), labels.loc[train_files].to_numpy())
        predicted = pipeline.predict(features.loc[test_files].to_numpy())
        per_fold.append(
            (labels.loc[test_files].to_numpy(), predicted, indexed.loc[test_files].reset_index())
        )
    pooled = evaluate_pooled(
        per_fold, "speaker_independent", model=MODEL, feature_config=WINNER.name,
        notes="phase-7 slice analysis",
    )
    return {
        "gender": pooled["slices"]["gender"],
        "intensity": pooled["slices"]["intensity_label"],
        "per_class": pooled["per_class"],
        "confusion": pooled["confusion_matrix"],
        "overall": pd.DataFrame([{
            "accuracy": pooled["accuracy"], "macro_f1": pooled["macro_f1"],
        }]),
    }


# --- 5. Arousal-first two-stage --------------------------------------------


def experiment_two_stage(folds: pd.DataFrame) -> pd.DataFrame:
    """Predict arousal first, then emotion within that branch.

    Hypothesis: Phase 2 showed prosody orders by arousal, and the largest
    confusions (sad->calm, happy->fearful) sit *within* arousal bands. If the
    hard part is discriminating within a band, splitting the problem should
    let each specialist focus on that and beat the flat 8-way model.
    """
    _, indexed, features = _load(normalise=True)
    labels = indexed["emotion_label"]
    arousal = labels.map(lambda e: "high" if e in HIGH_AROUSAL else "low")

    per_fold, stage1_accuracy = [], []
    for fold in range(5):
        train_files, test_files = get_split("speaker_independent", fold, folds=folds)
        x_train = features.loc[train_files].to_numpy()
        x_test = features.loc[test_files].to_numpy()

        gate = build(MODEL)
        gate.fit(x_train, arousal.loc[train_files].to_numpy())
        predicted_arousal = gate.predict(x_test)
        stage1_accuracy.append(
            float((predicted_arousal == arousal.loc[test_files].to_numpy()).mean())
        )

        # A specialist per branch, each trained only on that branch's clips.
        specialists = {}
        for band in ("high", "low"):
            mask = arousal.loc[train_files].to_numpy() == band
            specialist = build(MODEL)
            specialist.fit(x_train[mask], labels.loc[train_files].to_numpy()[mask])
            specialists[band] = specialist

        predicted = np.empty(len(x_test), dtype=object)
        for band in ("high", "low"):
            mask = predicted_arousal == band
            if mask.any():
                predicted[mask] = specialists[band].predict(x_test[mask])

        per_fold.append(
            (labels.loc[test_files].to_numpy(), predicted, indexed.loc[test_files].reset_index())
        )

    pooled = evaluate_pooled(
        per_fold, "speaker_independent",
        model=f"{MODEL}-two-stage-arousal", feature_config=WINNER.name,
        hyperparams={"stage1": "arousal high/low", "stage2": "emotion within branch"},
        notes="phase-7 arousal-first two-stage",
    )
    return pd.DataFrame([{
        "model": "two-stage (arousal first)",
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "stage1_arousal_accuracy": float(np.mean(stage1_accuracy)),
    }])


# --- confusion structure ----------------------------------------------------


def confusion_structure(confusion: pd.DataFrame) -> pd.DataFrame:
    """Do errors cluster by arousal, or are they spread evenly?

    Splits every off-diagonal error into within-band and across-band, and
    compares against what random errors would give.
    """
    within = across = 0
    for true in LABELS:
        for predicted in LABELS:
            if true == predicted:
                continue
            count = int(confusion.loc[true, predicted])
            same_band = (true in HIGH_AROUSAL) == (predicted in HIGH_AROUSAL)
            if same_band:
                within += count
            else:
                across += count

    total = within + across
    # Chance baseline: of the 7 wrong labels for any true class, how many sit
    # in the same arousal band? 3 of 7 for high-arousal, 3 of 7 for low.
    expected_within = 3 / 7
    return pd.DataFrame([{
        "within_arousal_errors": within,
        "across_arousal_errors": across,
        "within_share": within / total if total else 0.0,
        "chance_within_share": expected_within,
        "lift": (within / total) / expected_within if total else 0.0,
    }])


def run() -> dict[str, pd.DataFrame]:
    """Run all five experiments and return their tables."""
    set_seeds()
    folds = load_folds()

    print("1. leakage diagnosis -- can we predict the speaker?")
    leakage = experiment_leakage_diagnosis(folds)
    print(leakage.round(4).to_string(index=False), "\n")

    print("2. lexical invariance")
    lexical = experiment_lexical_invariance(folds)
    print(lexical.round(4).to_string(index=False), "\n")

    print("3+4. intensity and gender slices")
    slices = experiment_slices(folds)
    print(slices["gender"].round(4).to_string(index=False))
    print(slices["intensity"].round(4).to_string(index=False), "\n")

    print("5. arousal-first two-stage")
    two_stage = experiment_two_stage(folds)
    print(two_stage.round(4).to_string(index=False), "\n")

    print("confusion structure")
    structure = confusion_structure(slices["confusion"])
    print(structure.round(4).to_string(index=False))

    return {
        "leakage": leakage, "lexical": lexical, "two_stage": two_stage,
        "structure": structure, **slices,
    }


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    tables = run()
    out = Path("features")
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"phase7_{name}.parquet")
    print(f"\nwrote {len(tables)} tables to {out}/")
