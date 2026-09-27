"""The four training arms, as reusable data splits for *any* model.

`docs/two_stage_findings.md` compared four ways of using the same four corpora
and found that the way the extra data is used matters more than whether it is
used at all: naive pooling costs 4.68 pp, while a frozen-backbone two-stage
approach costs 1.17 pp, which is inside the noise.

That comparison was run on frozen wav2vec2 embeddings with an MLP. The *protocol*
is model-agnostic, and anyone training a CNN, an attention model or anything else
should be able to run the same four arms without reimplementing them — otherwise
the results are not comparable and the comparison was pointless.

This module hands back file lists. It trains nothing and knows nothing about any
particular model.

The four arms
-------------
``ravdess_only``
    The control. Train on the RAVDESS training actors only.

``pooled``
    One training set containing the RAVDESS training actors **and** every
    external clip, fitted once. This is the obvious approach and the worst
    performing.

``two_stage``
    Fit on everything, then **continue** fitting on the RAVDESS training actors
    alone. The model starts from what the large mixed set taught it and then
    moves toward the distribution it will be tested on.

``two_stage_frozen``
    Fit on everything, then freeze the learned representation and retrain only
    the final classification layer on the RAVDESS training actors. The external
    data decides *what features exist*; RAVDESS alone decides *where the
    decision boundaries go*. Best of the four.

In every arm the **test set is identical** — the frozen RAVDESS
`speaker_independent` fold. No external clip is ever tested, which is what keeps
the numbers comparable to every other result in the project and stops a model
scoring on `calm` merely by recognising which corpus a clip came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ser import corpora
from ser.splits import get_split

ARMS: tuple[str, ...] = ("ravdess_only", "pooled", "two_stage", "two_stage_frozen")

#: Arms whose second stage must not update the learned representation.
FREEZE_BACKBONE: frozenset[str] = frozenset({"two_stage_frozen"})


@dataclass(frozen=True)
class ArmSplit:
    """The data a single arm needs, for one fold.

    Attributes:
        arm: Which arm this is.
        fold: Fold number, 0-based.
        stage1: Filenames to fit on first, or ``None`` when the arm is
            single-stage. For the two-stage arms this is every clip from all
            four corpora that is not in the test fold.
        stage2: Filenames for the second fit. For single-stage arms this is the
            only fit. Always RAVDESS-only except under ``pooled``.
        test: The held-out filenames. **Always RAVDESS-only**, and identical
            across all four arms.
        freeze_backbone: Whether stage 2 should update only the final
            classification layer and leave the representation fixed.
    """

    arm: str
    fold: int
    stage1: list[str] | None
    stage2: list[str]
    test: list[str]
    freeze_backbone: bool

    def describe(self) -> str:
        """One-line summary, for logging what an experiment actually ran on."""
        first = f"{len(self.stage1):,}" if self.stage1 else "-"
        note = " (frozen backbone)" if self.freeze_backbone else ""
        return (
            f"{self.arm:<17s} fold {self.fold}  "
            f"stage1={first:>7s}  stage2={len(self.stage2):>6,}{note}  "
            f"test={len(self.test):>4,}"
        )


def arm_split(
    arm: str,
    fold: int,
    index: pd.DataFrame | None = None,
    protocol: str = "speaker_independent",
) -> ArmSplit:
    """Build one arm's data for one fold.

    Args:
        arm: One of `ARMS`.
        fold: Fold number, 0-based.
        index: Merged index from `ser.corpora.build_index`. Defaults to the
            built index on disk. Only needed for the arms that use external
            data.
        protocol: Which frozen RAVDESS protocol supplies the fold. Leave as the
            default unless you know why you are changing it.

    Returns:
        An `ArmSplit`.

    Raises:
        ValueError: On an unknown arm, or if an external clip would reach the
            test side -- which would silently invalidate the comparison.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    ravdess_train, test = get_split(protocol, fold)

    if arm == "ravdess_only":
        return ArmSplit(arm, fold, None, list(ravdess_train), list(test), False)

    if index is None:
        index = corpora.load_index()
    source = index.set_index("filename")["source"].astype(str)
    external = source.index[source != "RAVDESS"].tolist()

    leaked = {str(source[n]) for n in test if n in source.index and source[n] != "RAVDESS"}
    if leaked:
        raise ValueError(
            f"test side contains clips from {sorted(leaked)}; every arm must be "
            f"scored on RAVDESS alone or the comparison is meaningless"
        )

    everything = list(ravdess_train) + external

    if arm == "pooled":
        return ArmSplit(arm, fold, None, everything, list(test), False)

    # both two-stage arms share the same data; they differ only in whether the
    # representation is allowed to move during stage 2
    return ArmSplit(
        arm, fold, everything, list(ravdess_train), list(test),
        arm in FREEZE_BACKBONE,
    )


def all_arms(
    fold: int, index: pd.DataFrame | None = None
) -> dict[str, ArmSplit]:
    """Every arm for one fold, so a caller can loop over the comparison."""
    if index is None:
        index = corpora.load_index()
    return {arm: arm_split(arm, fold, index=index) for arm in ARMS}


def summarise(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """The four arms across all five folds, as a table.

    Reproduces the design table in `docs/two_stage_findings.md` from the actual
    fold definitions rather than from prose, so the two cannot drift apart.
    """
    if index is None:
        index = corpora.load_index()
    rows = []
    for fold in range(5):
        for arm, split in all_arms(fold, index=index).items():
            rows.append({
                "arm": arm,
                "fold": fold,
                "stage1": len(split.stage1) if split.stage1 else 0,
                "stage2": len(split.stage2),
                "test": len(split.test),
                "freeze_backbone": split.freeze_backbone,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    for arm, split in all_arms(0).items():
        print(split.describe())
