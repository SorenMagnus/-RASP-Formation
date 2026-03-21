# APF-LF Research Artifact

This repository contains a deterministic platoon simulation stack for APF-LF research:

- scenario generation and road geometry
- nominal controllers: `adaptive_apf`, `apf_lf`, `st_apf`, `apf`
- strong baselines: `dwa`, `orca`
- mode decision: static and FSM-based topology/behavior switching
- safety layer: discrete-consistent `CBF-QP`
- analysis: metrics, statistics, figure/table export
- replay: saved rollout verification from persisted artifacts

The current codebase is no longer a Phase A scaffold. It is a Phase E research prototype with a working paper-style experiment pipeline, replay support, and regression tests.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Run a single experiment:

```bash
python scripts/run_experiment.py --config configs/default.yaml --seeds 0 1 --exp-id smoke_run
```

Replay a saved run and verify its summary:

```bash
python scripts/replay_run.py --run-dir outputs/smoke_run --seed 0 --verify-summary
```

Run a small paper-style experiment matrix:

```bash
python scripts/reproduce_paper.py --exp-id paper_smoke --seeds 0 --scenarios s1_local_minima --methods adaptive_apf apf dwa orca
```

Re-export figures and tables from an existing matrix:

```bash
python scripts/export_figures.py --input-dir outputs/paper_smoke
```

## Config Layout

- `configs/scenarios/`: benchmark scenario families
- `configs/baselines/`: controller overrides for baseline methods
- `configs/ablations/`: contribution-removal ablations

Generated experiment directories contain:

- `config_resolved.yaml`
- `summary.csv`
- `traj/seed_XXXX.npz`
- `tables/`
- `figures/`

## Core Package Layout

- `src/apflf/env/`: geometry, road, dynamics, obstacles, scenarios
- `src/apflf/controllers/`: APF-family controllers plus DWA and ORCA baselines
- `src/apflf/decision/`: static and FSM mode decision
- `src/apflf/safety/`: CBF-QP safety filter and solver wrapper
- `src/apflf/sim/`: world loop, batch runner, replay
- `src/apflf/analysis/`: metrics, stats, export

## Reproducibility

See [docs/reproducibility.md](/C:/Users/27363/Desktop/论文级代码/docs/reproducibility.md) for the end-to-end workflow and [docs/development.md](/C:/Users/27363/Desktop/论文级代码/docs/development.md) for local validation and CI expectations.

## Current Status

The artifact is runnable, replayable, and regression-tested. The main remaining research work is behavioral tuning and larger paper-scale experiment sweeps rather than missing infrastructure.
