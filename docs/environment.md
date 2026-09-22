# Environment notes

## Do not keep the project inside OneDrive (Windows)

**Symptom.** `import sklearn` fails with:

```
ImportError: DLL load failed while importing _cythonized_array_utils:
An Application Control policy has blocked this file.
```

The blocked file changes as you retry — `_cythonized_array_utils`, then
`_minpack`, then another — because each compiled extension is checked the
first time it is loaded.

**Cause.** Windows **Smart App Control** refuses to load unsigned compiled
binaries out of OneDrive-synced directories. scipy's `.pyd` extensions are
unsigned, and OneDrive stores files as cloud placeholders (the file carries the
`ReparsePoint` attribute). Nothing about the package is wrong: the identical
wheel imports fine from a local path on the same machine, in the same minute.

Diagnosed 2026-09-22. The first block event on this repo's scipy is from that
date, months after the venv was built and 446 experiments were logged against
it — so this appears when the policy's verdict changes, not when your code
does.

**Fix.** Keep the repository on local disk, e.g. `C:\dev\Unheard`, and build
the virtual environment there. Confirm with:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState).VerifiedAndReputablePolicyState
# 0 = off, 1 = enforcing, 2 = evaluation
Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 50 |
  Where-Object Id -in 3033,3077 | Select-Object TimeCreated, Message
```

**What not to do.** Turning Smart App Control off also fixes it, but it
**cannot be turned back on without reinstalling Windows**, and it weakens
protection for every application on the machine. Moving a folder is reversible;
that is not.

Losing OneDrive backup costs nothing here: `data/`, `data_external/` and
`features/` are all gitignored and regenerable, and the code is on GitHub.
Moving out also stops OneDrive continuously syncing ~3 GB of audio and parquet
caches, which is a real speed-up during extraction runs.

## Installing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

For the Phase 13-14 embedding work, additionally:

```bash
.venv/Scripts/python -m pip install -r requirements-deep.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The deep dependencies are kept separate because they are ~2 GB, CI needs none
of them, and the `+cpu` local version specifier does not resolve on CI's Linux
runner. `ser.embeddings` imports torch inside its functions, so its tests run
without it.

## Long jobs and machine sleep

Feature extraction over 12,162 clips and the wav2vec2 passes run for a long
time, and a laptop sleeping mid-run costs the whole pass. Long-running drivers
hold a Windows power request for exactly as long as they run and release it on
exit — including on crash or Ctrl-C — so no power-plan setting is changed and
there is nothing to remember to undo. The display is still allowed to sleep;
only system sleep is blocked.

## Downloading the external corpora

```bash
.venv/Scripts/python -c "import ser.corpora as c; c.fetch_cremad(); c.fetch_tess(); c.fetch_savee()"
```

CREMA-D and TESS need no authentication. CREMA-D is the slow one — 7,442
individual Git-LFS requests, 20-40 minutes, and GitHub throttles partway
through. All three fetchers are resumable: re-running downloads only what is
missing.

SAVEE is not freely redistributable. `kagglehub` will use credentials you have
already authorised under `~/.kaggle/`; otherwise register at
[Surrey](http://kahlan.eps.surrey.ac.uk/savee/Download.html) and drop
`AudioData.zip` into `data_external/`. The pipeline runs without it — SAVEE is
3.9% of the merge.
