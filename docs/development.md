# Development Guide

## Local setup

```bash
python -m pip install -e ".[dev]"
```

## Validation commands

Run the full regression suite:

```bash
python -m pytest -q
```

Run bytecode compilation checks:

```bash
python -m compileall src tests scripts
```

Run a replay verification against an existing output:

```bash
python scripts/replay_run.py --run-dir outputs/stage5_reproduce_smoke/runs/s1_local_minima__adaptive_apf --seed 0 --verify-summary
```

## CI expectations

The CI workflow runs:

- editable install with `.[dev]`
- `python -m compileall src tests scripts`
- `python -m pytest -q`

## Packaging notes

- `README.md` describes the supported workflows
- `CITATION.cff` provides citation metadata for the artifact
- `.github/workflows/ci.yml` provides the baseline validation job

## Research notes

The remaining open work is mostly method tuning and larger experiment sweeps. Infrastructure items such as replay, matrix execution, export, and regression testing are already in place.
