"""Materialise a filename-only stand-in for the audio tree, for CI.

The clips are gitignored (CLAUDE.md rule 7), so a CI runner has no
``data/data/PersonNN/``. But most of the suite never opens an audio file:
`ser.metadata` parses filenames, `ser.splits` works off those, and
`ser.evaluate` scores synthetic labels. Those tests only need the *tree* to
exist.

Every one of the 1,440 filenames is already committed, in
``splits/folds.csv`` -- the speaker-independent protocol assigns each clip to
exactly one fold. So the tree is reconstructable from the repo alone, and the
fold-contract and metric tests run for real in CI rather than being skipped.

The files written are empty. Anything that decodes audio will fail on them,
which is deliberate: ``tests/test_preprocess.py`` and ``tests/test_audit.py``
need the real clips and are excluded from the CI run instead of being faked.

Refuses to touch an existing tree, so running this on a machine that holds
the real dataset cannot destroy it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDS = REPO_ROOT / "splits" / "folds.csv"
DEFAULT_ROOT = REPO_ROOT / "data" / "data"

EXPECTED_CLIPS = 1440
EXPECTED_ACTORS = 24


def filenames(folds_path: Path) -> list[str]:
    """Every distinct clip name in the frozen fold table."""
    with folds_path.open(newline="", encoding="utf-8") as handle:
        names = {row["filename"] for row in csv.DictReader(handle)}
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--folds", type=Path, default=FOLDS)
    args = parser.parse_args()

    if args.root.exists() and any(args.root.iterdir()):
        print(f"refusing to write into non-empty {args.root}", file=sys.stderr)
        return 1

    names = filenames(args.folds)
    if len(names) != EXPECTED_CLIPS:
        print(f"expected {EXPECTED_CLIPS} clips, found {len(names)}", file=sys.stderr)
        return 1

    actors = set()
    for name in names:
        actor = name.split("-")[4].removesuffix(".wav")
        actors.add(actor)
        folder = args.root / f"Person{actor}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).touch()

    if len(actors) != EXPECTED_ACTORS:
        print(f"expected {EXPECTED_ACTORS} actors, found {len(actors)}", file=sys.stderr)
        return 1

    print(f"wrote {len(names)} empty stubs across {len(actors)} actors -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
