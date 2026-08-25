"""Classical sklearn baselines.

**Defaults only.** No hyperparameter tuning happens here — these numbers are
a floor that later models must beat to justify themselves, not a result. The
winner gets tuned in Phase 5, via CV *inside* the training folds.

Scaling is fit on the training fold and applied to the held-out fold
(CLAUDE.md rule 2). SVM-RBF is scale-sensitive and would be meaningless
without it; RandomForest is scale-invariant but goes through the same
pipeline so the only difference between the two runs is the estimator.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ser import SEED  # single source; do not redeclare

#: Model name -> factory. Every estimator uses library defaults apart from
#: the fixed seed and n_jobs.
MODELS: dict[str, Callable[[], object]] = {
    "svm-rbf": lambda: SVC(kernel="rbf", random_state=SEED),
    "random-forest": lambda: RandomForestClassifier(
        random_state=SEED, n_jobs=-1
    ),
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
    return Pipeline(
        [("scaler", StandardScaler()), ("estimator", MODELS[name]())]
    )


def default_hyperparams(name: str) -> dict[str, object]:
    """The estimator's actual parameters, for the results log."""
    estimator = build(name).named_steps["estimator"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in sorted(params.items())
        if isinstance(value, (int, float, str, bool, type(None)))
    }
