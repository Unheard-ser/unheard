"""Invariants for the model factory and tuning.

The assertion that matters most here is that tuning cannot see the held-out
fold. Everything else in the project guards leakage at the *split* level;
`GridSearchCV` is the one place where a second, inner split is created, and
that is exactly where leakage is easy to reintroduce by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser import SEED
from ser.models.classical import (
    MODELS,
    TUNING_GRIDS,
    build,
    default_hyperparams,
    tune,
)


@pytest.fixture(scope="module")
def toy():
    """Small separable problem with speaker groups, for fast tuning tests."""
    rng = np.random.default_rng(SEED)
    n_per_class = 40
    classes = ["angry", "calm", "sad"]
    features = np.vstack(
        [rng.normal(offset, 1.0, (n_per_class, 6)) for offset in (0, 4, 8)]
    )
    labels = np.repeat(classes, n_per_class)
    groups = np.tile(np.arange(8), n_per_class * len(classes) // 8)
    return features, labels, groups


# --- the factory ------------------------------------------------------------


def test_all_seven_models_are_registered():
    assert set(MODELS) == {
        "svm-rbf", "random-forest", "xgboost", "logistic-regression",
        "knn", "adaboost", "gaussian-nb",
    }


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_fits_and_predicts_string_labels(name, toy):
    features, labels, _ = toy
    pipeline = build(name)
    pipeline.fit(features, labels)
    predicted = pipeline.predict(features)

    assert len(predicted) == len(labels)
    assert set(predicted) <= set(labels)
    # On a deliberately separable problem every model should beat chance.
    assert (predicted == labels).mean() > 0.5, f"{name} failed a separable problem"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_is_a_scaler_plus_estimator_pipeline(name):
    """The Pipeline is the structural guarantee that scaling is train-only."""
    pipeline = build(name)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "estimator"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_build_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build("transformer")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_hyperparams_are_json_serialisable(name):
    import json

    json.dumps(default_hyperparams(name))


def test_seeded_models_are_reproducible(toy):
    features, labels, _ = toy
    first = build("random-forest").fit(features, labels).predict(features)
    second = build("random-forest").fit(features, labels).predict(features)
    assert np.array_equal(first, second)


# --- tuning must not see the held-out fold ---------------------------------


def test_tuning_never_touches_data_it_was_not_given(toy):
    """The inner search sees only what is passed as training data.

    Held-out rows are simply absent from the call, so there is no path by
    which the search could score against them.
    """
    features, labels, groups = toy
    train_mask = np.arange(len(labels)) % 4 != 0

    search = tune(
        "knn", features[train_mask], labels[train_mask], cv=2,
        groups=groups[train_mask],
    )
    assert search.best_estimator_ is not None
    # n_samples_ is recorded by the scaler, and reflects only training rows.
    scaler = search.best_estimator_.named_steps["scaler"]
    assert scaler.n_samples_seen_ == train_mask.sum()


def test_grouped_inner_cv_keeps_speakers_apart(toy):
    """With groups supplied the inner CV must be speaker-grouped too.

    Otherwise the hyperparameters are chosen against a leaky inner estimate,
    which quietly inflates them even though the outer fold stays clean.
    """
    from sklearn.model_selection import GroupKFold

    features, labels, groups = toy
    search = tune("knn", features, labels, cv=3, groups=groups)
    assert isinstance(search.cv, GroupKFold)


def test_tuning_optimises_macro_f1_not_accuracy(toy):
    """Accuracy would let the search abandon the under-represented class."""
    features, labels, groups = toy
    search = tune("knn", features, labels, cv=2, groups=groups)
    assert search.scoring == "f1_macro"


def test_tune_rejects_a_model_with_no_grid():
    with pytest.raises(ValueError, match="No tuning grid"):
        tune("gaussian-nb", np.zeros((10, 3)), np.array(["a"] * 5 + ["b"] * 5))


def test_tuning_grids_only_reference_real_pipeline_parameters():
    """A typo in a grid key fails at fit time, deep inside a long run."""
    for name, grid in TUNING_GRIDS.items():
        valid = set(build(name).get_params())
        assert set(grid) <= valid, f"{name}: unknown keys {set(grid) - valid}"
