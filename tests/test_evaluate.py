"""Scoring invariants, verified with synthetic predictions only.

No model is trained anywhere in this file. Every test constructs labels
directly, so the scoring logic is checked against arithmetic we can do by
hand rather than against whatever a model happens to output.

Every test writes to a tmp_path. Nothing here may touch the real
results/results.csv -- polluting the shared log with test rows would
undermine CLAUDE.md rule 5.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from ser.evaluate import (
    LABELS,
    RESULTS_COLUMNS,
    _resolve_owner,
    evaluate,
    evaluate_pooled,
)
from ser.metadata import load


@pytest.fixture(scope="module")
def meta():
    """A 300-clip slice standing in for one speaker-independent test fold."""
    return load().iloc[:300].reset_index(drop=True)


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results.csv"


def _score(meta, y_true, y_pred, results, **kw):
    kw.setdefault("model", "synthetic")
    kw.setdefault("feature_config", "none")
    return evaluate(
        y_true, y_pred, "speaker_independent", 0, meta,
        results_path=results, **kw,
    )


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- 1 & 2. metrics against hand-computable cases --------------------------


def test_perfect_predictions(meta, results):
    truth = meta["emotion_label"].tolist()
    out = _score(meta, truth, truth, results)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    matrix = out["confusion_matrix"].to_numpy()
    assert np.array_equal(matrix, np.diag(np.diag(matrix)))


def test_single_class_prediction_scores_the_class_prior(meta, results):
    truth = meta["emotion_label"].tolist()
    pred = ["angry"] * len(truth)
    expected = truth.count("angry") / len(truth)
    out = _score(meta, truth, pred, results)
    assert out["accuracy"] == pytest.approx(expected)


# --- 3-6. confusion matrix and per-class shape -----------------------------


def test_confusion_matrix_totals(meta, results):
    truth = meta["emotion_label"].tolist()
    pred = np.roll(truth, 1).tolist()
    out = _score(meta, truth, pred, results)
    matrix = out["confusion_matrix"]
    assert matrix.to_numpy().sum() == len(truth)
    counts = pd.Series(truth).value_counts()
    for label in LABELS:
        assert matrix.loc[label].sum() == counts.get(label, 0)


def test_confusion_matrix_is_always_8x8_in_fixed_order(meta, results):
    """A class the model never predicts still gets a row and a column."""
    truth = meta["emotion_label"].tolist()
    pred = ["angry"] * len(truth)
    out = _score(meta, truth, pred, results)
    matrix = out["confusion_matrix"]
    assert matrix.shape == (8, 8)
    assert tuple(matrix.index) == LABELS
    assert tuple(matrix.columns) == LABELS


def test_absent_class_scores_zero_not_nan(meta, results):
    truth = meta["emotion_label"].tolist()
    pred = ["angry"] * len(truth)
    out = _score(meta, truth, pred, results)
    per_class = out["per_class"].set_index("emotion")
    assert per_class.loc["neutral", "precision"] == 0.0
    assert not out["per_class"][["precision", "recall", "f1"]].isna().any().any()


def test_per_class_support_sums_to_n(meta, results):
    truth = meta["emotion_label"].tolist()
    out = _score(meta, truth, truth, results)
    assert out["per_class"]["support"].sum() == len(truth)
    assert out["n"] == len(truth)


# --- 7 & 8. slices ----------------------------------------------------------


def test_gender_slices_recombine_to_overall_accuracy(meta, results):
    rng = np.random.default_rng(0)
    truth = meta["emotion_label"].to_numpy()
    pred = np.where(rng.random(len(truth)) < 0.6, truth, "calm")
    out = _score(meta, truth, pred, results)

    gender = out["slices"]["gender"]
    weighted = (gender["accuracy"] * gender["n"]).sum() / gender["n"].sum()
    assert weighted == pytest.approx(out["accuracy"])
    assert gender["n"].sum() == len(truth)


def test_slices_cover_gender_intensity_and_emotion(meta, results):
    truth = meta["emotion_label"].tolist()
    out = _score(meta, truth, truth, results)
    assert set(out["slices"]) == {"gender", "intensity_label", "emotion_label"}
    for column, frame in out["slices"].items():
        assert set(frame["level"]) == set(meta[column].unique())
        assert frame["n"].sum() == len(truth)


# --- 9-11. the results log --------------------------------------------------


def test_one_row_appended_with_exactly_the_schema(meta, results):
    truth = meta["emotion_label"].tolist()
    _score(meta, truth, truth, results, model="svm-rbf", feature_config="mfcc40")
    rows = _read(results)
    assert len(rows) == 1
    assert tuple(rows[0]) == RESULTS_COLUMNS
    assert rows[0]["model"] == "svm-rbf"
    assert rows[0]["feature_config"] == "mfcc40"
    assert rows[0]["protocol"] == "speaker_independent"
    assert float(rows[0]["accuracy"]) == 1.0


def test_two_calls_append_two_rows_with_distinct_run_ids(meta, results):
    truth = meta["emotion_label"].tolist()
    _score(meta, truth, truth, results)
    _score(meta, truth, truth, results)
    rows = _read(results)
    assert len(rows) == 2
    assert rows[0]["run_id"] != rows[1]["run_id"]


def test_header_written_once(meta, results):
    truth = meta["emotion_label"].tolist()
    for _ in range(3):
        _score(meta, truth, truth, results)
    text = results.read_text(encoding="utf-8")
    assert text.count("run_id") == 1
    assert len(_read(results)) == 3


# --- 12-14. logging details -------------------------------------------------


def test_hyperparams_round_trip_as_json(meta, results):
    truth = meta["emotion_label"].tolist()
    params = {"C": 10, "gamma": "scale", "kernel": "rbf"}
    _score(meta, truth, truth, results, hyperparams=params)
    assert json.loads(_read(results)[0]["hyperparams"]) == params


def test_owner_defaults_and_overrides(meta, results):
    truth = meta["emotion_label"].tolist()
    _score(meta, truth, truth, results)
    assert _read(results)[0]["owner"]
    _score(meta, truth, truth, results, owner="teammate-b")
    assert _read(results)[1]["owner"] == "teammate-b"
    assert _resolve_owner("explicit") == "explicit"
    assert _resolve_owner() != ""


def test_append_false_writes_nothing_but_returns_metrics(meta, results):
    truth = meta["emotion_label"].tolist()
    out = _score(meta, truth, truth, results, append=False)
    assert out["accuracy"] == 1.0
    assert not results.exists()


# --- 15. pooled scoring -----------------------------------------------------


def test_pooled_matches_direct_evaluation_of_the_concatenation(results, tmp_path):
    full = load()
    rng = np.random.default_rng(7)
    per_fold, all_true, all_pred, all_meta = [], [], [], []

    for start in range(0, 1000, 200):
        chunk = full.iloc[start : start + 200].reset_index(drop=True)
        truth = chunk["emotion_label"].to_numpy()
        pred = np.where(rng.random(len(truth)) < 0.5, truth, "sad")
        per_fold.append((truth, pred, chunk))
        all_true.extend(truth.tolist())
        all_pred.extend(pred.tolist())
        all_meta.append(chunk)

    pooled = evaluate_pooled(
        per_fold, "speaker_independent",
        model="synthetic", feature_config="none", results_path=results,
    )
    direct = evaluate(
        all_true, all_pred, "speaker_independent", "pooled",
        pd.concat(all_meta, ignore_index=True),
        model="synthetic", feature_config="none",
        results_path=tmp_path / "other.csv",
    )

    assert pooled["accuracy"] == direct["accuracy"]
    assert pooled["macro_f1"] == direct["macro_f1"]
    assert pooled["n"] == direct["n"] == 1000
    assert _read(results)[0]["fold"] == "pooled"


def test_pooled_rejects_an_explicit_fold(meta):
    with pytest.raises(ValueError, match="do not pass fold"):
        evaluate_pooled(
            [(["angry"], ["angry"], meta.iloc[:1])],
            "speaker_independent",
            fold=0, model="m", feature_config="f",
        )


def test_pooled_rejects_empty_input():
    with pytest.raises(ValueError, match="nothing to pool"):
        evaluate_pooled([], "speaker_independent", model="m", feature_config="f")


# --- 16. input validation ---------------------------------------------------


def test_mismatched_lengths_raise(meta, results):
    truth = meta["emotion_label"].tolist()
    with pytest.raises(ValueError, match="y_pred"):
        _score(meta, truth, truth[:-1], results)


def test_metadata_length_mismatch_raises(meta, results):
    truth = meta["emotion_label"].tolist()
    with pytest.raises(ValueError, match="metadata has"):
        evaluate(
            truth, truth, "speaker_independent", 0, meta.iloc[:10],
            model="m", feature_config="f", results_path=results,
        )
