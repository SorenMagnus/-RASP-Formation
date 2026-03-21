# Reproducibility Guide

This repository is designed around deterministic rollouts and persisted artifacts.

## 1. Run a single experiment

```bash
python scripts/run_experiment.py --config configs/default.yaml --seeds 0 1 --exp-id local_smoke
```

This creates:

- `outputs/local_smoke/config_resolved.yaml`
- `outputs/local_smoke/summary.csv`
- `outputs/local_smoke/traj/seed_0000.npz`
- `outputs/local_smoke/traj/seed_0001.npz`

## 2. Replay a saved run

Replay recomputes metrics from the saved trajectory artifact and can verify them against `summary.csv`.

```bash
python scripts/replay_run.py --run-dir outputs/local_smoke --seed 0 --verify-summary
```

If verification succeeds, the command prints the recomputed summary and finishes with `summary verified`.

## 3. Run a paper-style matrix

```bash
python scripts/reproduce_paper.py ^
  --exp-id paper_smoke ^
  --seeds 0 1 ^
  --scenarios s1_local_minima s2_dynamic_crossing s3_narrow_passage ^
  --methods adaptive_apf apf apf_lf st_apf dwa orca ^
  --ablations no_cbf no_fsm no_risk_adaptation no_st_terms no_escape
```

On shells that do not support `^` line continuation, put the command on one line.

The output directory contains:

- `all_runs.csv`
- `tables/main_results.csv`
- `tables/pairwise_vs_reference.csv`
- `figures/*.pdf`
- `runs/<scenario>__<method>/...`

## 4. Re-export paper artifacts

If `all_runs.csv` already exists, figures and tables can be regenerated without rerunning simulations:

```bash
python scripts/export_figures.py --input-dir outputs/paper_smoke
```

## 5. Determinism notes

- trajectory state, actions, modes, and replayed metrics are expected to be deterministic
- wall-clock runtime and QP solve time measurements are intentionally treated as nondeterministic diagnostics
- `config_resolved.yaml` is stored in a replay-compatible format and can be loaded directly by the config loader

## 6. Current repository note

If the workspace is not initialized as a git repository, experiment summaries record `git_commit=nogit`. The rest of the artifact pipeline still works.
