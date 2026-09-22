"""Run a command with the machine held awake for exactly its duration.

Usage:  python keepawake.py <command> [args...]

Uses SetThreadExecutionState, which registers a power request for this
process only. Nothing in the system's power plan is modified, and the
request is released the moment this process exits -- including on crash,
Ctrl-C, or being killed -- so the laptop goes back to its normal sleep
behaviour with no cleanup step to forget.

ES_DISPLAY_REQUIRED is deliberately NOT set: the screen is allowed to
switch off, which is what you want for an overnight extraction run. Only
system sleep is blocked.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def hold_awake() -> bool:
    """Ask Windows to keep the system running. True if the request was taken."""
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    if ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0:
        return True
    # Away-mode is refused on some machines; system-required alone is enough.
    return ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ) != 0


def release() -> None:
    """Hand sleep control back to Windows."""
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    held = hold_awake()
    print(f"[keepawake] sleep {'BLOCKED' if held else 'NOT blocked (request refused)'}"
          f" for: {' '.join(sys.argv[1:])}", flush=True)
    try:
        return subprocess.call(sys.argv[1:])
    finally:
        release()
        print("[keepawake] released -- normal sleep behaviour restored", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
