"""Classical sklearn baselines.

**Defaults only, except where explicitly tuned.** These numbers are the floor
every later model must beat to justify its complexity, not a leaderboard
entry. Only the winner gets tuned, and only via CV *inside* the training
folds.

Scaling is fit on the training fold and applied to the held-out fold
(CLAUDE.md rule 2). Every estimator goes through the same `Pipeline`, so the
only difference between two runs is the estimator itself -- tree models do not
need scaling, but keeping the pipeline identical means a difference in score
cannot be an artefact of different preprocessing.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ser import SEED  # single source; do not redeclare

#: Model name -> factory. Every estimator uses library defaults apart from the
#: fixed seed, n_jobs, and max_iter where the default does not converge.
MODELS: dict[str, Callable[[], Any]] = {
    "svm-rbf": lambda: SVC(kernel="rbf", random_state=SEED),
    "random-forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "xgboost": lambda: _xgboost(),
    # lbfgs does not converge on 162 features at the default 100 iterations;
    # raising the cap is a convergence fix, not tuning.
    # n_jobs is deliberately absent: it has had no effect since sklearn 1.8
    "logistic-regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "knn": lambda: KNeighborsClassifier(n_jobs=-1),
    "adaboost": lambda: AdaBoostClassifier(random_state=SEED),
    "gaussian-nb": lambda: GaussianNB(),
}


def _xgboost():
    """XGBoost needs integer labels, so it is wrapped at fit time.

    Imported lazily so the module still imports without xgboost installed.
    """
    from xgboost import XGBClassifier

    return _LabelEncodedXGB(
        XGBClassifier(
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    )


class _LabelEncodedXGB(BaseEstimator, ClassifierMixin):
    """Adapt XGBoost's integer-label requirement to our string labels.

    Every other estimator here accepts ``"angry"`` directly. Rather than
    special-casing the caller, the encoding is hidden behind the usual
    fit/predict interface so all seven models are driven identically.

    Inherits `BaseEstimator` so it satisfies the sklearn estimator protocol
    -- `Pipeline` and `GridSearchCV` both introspect estimators via
    ``__sklearn_tags__`` and ``get_params``, which come from that base.
    """

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    def fit(self, x, y):
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder()
        self.classes_ = np.unique(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, self.encoder_.fit_transform(y))
        return self

    def predict(self, x):
        return self.encoder_.inverse_transform(self.estimator_.predict(x))


#: Grid for the winner only, searched by CV *inside* the training fold.
#: Deliberately small: a wide grid on 1,140 training clips overfits the inner
#: CV and buys a number that will not survive the outer fold.
TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "svm-rbf": {
        "estimator__C": [1, 5, 10, 50],
        "estimator__gamma": ["scale", 0.01, 0.001],
    },
    "random-forest": {
        "estimator__n_estimators": [200, 500],
        "estimator__max_depth": [None, 20],
        "estimator__min_samples_leaf": [1, 3],
    },
    "logistic-regression": {"estimator__C": [0.1, 1, 10]},
    "knn": {
        "estimator__n_neighbors": [3, 5, 11, 21],
        "estimator__weights": ["uniform", "distance"],
    },
}


def build(name: str) -> Pipeline:
    """Build a scaler + estimator pipeline.

    Wrapping the scaler in a `Pipeline` is what guarantees it is fit on
    training data only: `Pipeline.fit` fits the scaler on the training fold,
    and `Pipeline.predict` merely transforms the test fold with those
    already-fitted statistics. There is no code path that fits on test.

    Args:
        name: A key of `MODELS`.

    Returns:
        An unfitted pipeline.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}")
    return Pipeline([("scaler", StandardScaler()), ("estimator", MODELS[name]())])


def default_hyperparams(name: str) -> dict[str, object]:
    """The estimator's actual parameters, for the results log."""
    estimator = build(name).named_steps["estimator"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in sorted(params.items())
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def tune(
    name: str,
    x_train,
    y_train,
    cv: int = 3,
    scoring: str = "f1_macro",
    groups=None,
):
    """Tune one model by CV **inside** the training fold.

    The held-out fold is never touched here -- that is the whole point. When
    ``groups`` is supplied the inner CV is also speaker-grouped, so the inner
    estimate does not itself leak across speakers and inflate the choice of
    hyperparameters.

    Args:
        name: Model name; must have an entry in `TUNING_GRIDS`.
        x_train: Training-fold features.
        y_train: Training-fold labels.
        cv: Inner CV folds.
        scoring: Optimised metric. Macro-F1, not accuracy, so the search does
            not simply abandon neutral.
        groups: Optional speaker ids for grouped inner CV.

    Returns:
        The fitted `GridSearchCV`.

    Raises:
        ValueError: If no grid is defined for ``name``.
    """
    from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

    if name not in TUNING_GRIDS:
        raise ValueError(
            f"No tuning grid for {name!r}; defined for {sorted(TUNING_GRIDS)}"
        )

    splitter = (
        GroupKFold(n_splits=cv)
        if groups is not None
        else StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    )
    search = GridSearchCV(
        build(name),
        TUNING_GRIDS[name],
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups)
    return search
