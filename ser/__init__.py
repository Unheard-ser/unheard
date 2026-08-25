"""Speech emotion recognition on RAVDESS (ISB AMPBA foundation project).

`SEED` is defined here and imported everywhere else. It used to be redeclared
in four separate modules, which meant nothing stopped them drifting apart.
"""

from __future__ import annotations

import os
import random

#: The one seed. Import from here; do not redeclare.
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source we can reach (CLAUDE.md rule 6).

    Passing ``random_state=`` to an estimator only covers that estimator.
    This covers the global generators that otherwise make a rerun
    irreproducible -- numpy's legacy global RNG, the stdlib ``random``
    module, and torch once Phase 6 introduces it.

    Args:
        seed: The value to seed with. Defaults to `SEED`.

    Note:
        ``PYTHONHASHSEED`` is set here for completeness, but CPython reads it
        at interpreter startup. To make string hashing deterministic it must
        be exported *before* Python launches::

            PYTHONHASHSEED=42 python -m pytest

        Nothing we currently do depends on string hash order, so this is
        belt-and-braces rather than load-bearing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:  # pragma: no cover - torch arrives in Phase 6
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
