# AI_MEMORY - Current Handoff Memo

> Read this file first, then read `PROMPT_SYSTEM.md` and `RESEARCH_GOAL.md`, and only then touch code.
> This file is intentionally ASCII-only to avoid Windows editor / shell encoding regressions.
> It reflects the real repository and disk state on `2026-04-02`.

---

## 0. Current Cursor

- Date:
  - `2026-04-02`
- Git cursor:
  - `HEAD = 020053c798731622eadd48bd727ed925bf3b75f6`
- Worktree snapshot at rewrite time:
  - repo-tracked modified files:
    - `scripts/reproduce_paper.py`
    - `tests/test_reproduce_paper.py`
    - `AI_MEMORY.md`

### Live canonical status

- Main paper method remains:
  - `FSM + adaptive_apf + CBF-QP`
- RL remains appendix-only, not the paper mainline.
- `outputs/paper_canonical` already exists and must be preserved in place.
- Current bundle artifacts present:
  - `outputs/paper_canonical/manifest.json`
  - `outputs/paper_canonical/run_progress.json`
  - `outputs/paper_canonical/cell_progress.csv`
  - `outputs/paper_canonical/matrix_index.csv`
  - `outputs/paper_canonical/paper_acceptance.json`
  - `outputs/paper_canonical/run_runtime_state.json`
  - `outputs/paper_canonical/cell_runtime_state.csv`
- Current canonical completion truth:
  - `bundle_progress = 0.01818181818181818`
  - `num_expected_cells = 55`
  - `num_complete_cells = 1`
  - the only complete cell is:
    - `s1_local_minima__no_rl`
  - current active cell is:
    - `s1_local_minima__apf`
- Current live runtime truth from `status-only`:
  - `runner_pid = 25392`
  - `process_alive = true`
  - `orphaned = false`
  - `stalled = false`
  - `heartbeat_age_seconds ~= 17-19` at the last check
- Current acceptance truth:
  - `bundle_complete = false`
  - `primary_safety_valid = false`
  - this is still expected at this stage because 54 cells remain missing

### Most important conclusion

- The main blocker is no longer missing framework code.
- The canonical bundle is running again and is currently healthy.
- The remaining project-critical task is to let `paper_canonical` finish and then evaluate final acceptance.
- The new code added in this session fixed a real Windows-specific liveness bug in `status-only` / runtime audit.

---

## 1. Completed Work

### 1.1 Earlier mainline work that still stands

- Gatefix completed:
  - Beta-variance `confidence_raw`
  - hysteresis RL gate
- Reward v2 completed:
  - safety-aware reward shaping
  - reward config externalization
  - PPO reward diagnostics
- Warm-start completed:
  - `tau_enter_start = 0.25`
  - `gate_warmup_timesteps = 20000`
- Effective runtime threshold persistence completed.

### 1.2 Canonical artifact infrastructure already completed

- `scripts/reproduce_paper.py` already supports:
  - `--status-only`
  - `--validate-only`
- Manifest-first canonical audit is already in place.
- Runtime heartbeat / running-cell journal is already in place.
- Process-aware liveness / orphan reconciliation is already in place.

### 1.3 This session: Windows liveness fix and canonical recovery

This session resolved the last practical execution blocker on Windows.

- Diagnosed the real failure mode:
  - `paper_canonical` had an orphaned running cell on disk.
  - `status-only` itself could crash on Windows when `_check_process_alive(...)` used `os.kill(pid, 0)` against a detached child process.
  - The symptom was:
    - `WinError 87`
    - false liveness failure in pure-disk audit
    - inability to reliably monitor a live detached canonical process
- Fixed `scripts/reproduce_paper.py`:
  - In `_check_process_alive(...)`, on `win32`, replaced `os.kill(pid, 0)` with:
    - `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, ...)`
    - `GetExitCodeProcess(...)`
    - `STILL_ACTIVE == 259`
  - Non-Windows behavior remains unchanged.
- Extended `tests/test_reproduce_paper.py`:
  - added detached child liveness regression test
  - verified that a child launched with:
    - `CREATE_NEW_PROCESS_GROUP`
    - `CREATE_NO_WINDOW`
    is still recognized as alive
- Recovered canonical execution in detached mode:
  - previous orphaned runtime state:
    - `runner_pid = 15380`
    - `process_alive = false`
    - `orphaned = true`
  - current stable detached run:
    - `runner_pid = 25392`
    - `process_alive = true`
    - `orphaned = false`

### 1.4 Verification completed in this session

- `python -m compileall src tests scripts`
  - passed
- `python -m pytest -q tests/test_reproduce_paper.py`
  - passed
  - `19 passed`
- `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
  - passed
  - `27 passed`
- `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
  - now works correctly against the detached live canonical process
- `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
  - returns incomplete / not accepted exactly as expected at the current bundle stage

---

## 2. Current Research Judgment

- The project is no longer blocked by missing algorithm modules or missing paper infrastructure.
- The white-box mainline is still the correct paper narrative:
  - `FSM + adaptive_apf + CBF-QP`
- RL remains an appendix-only branch and must not be promoted back to the mainline.
- At the current moment, `primary_safety_valid = false` should not yet be interpreted as a proven white-box safety failure.
  - The bundle is still incomplete.
  - Only 1 of 55 cells is complete.
- Therefore the immediate priority is:
  - keep the canonical run healthy
  - finish the 55-cell bundle
  - then inspect final acceptance

---

## 3. Next Instruction

### 3.1 General rule

The next engineer / AI must:

- not change RL reward
- not change RL gate
- not change warm-start math
- not rewrite manifest-first audit
- not rewrite runtime heartbeat / orphan reconciliation
- not delete `outputs/paper_canonical`
- not treat runtime journal as acceptance
- not spawn a second canonical run if the live one with PID `25392` is still alive

### 3.2 Immediate operational instruction

Before writing new code, the next engineer must first check whether the current canonical process is still alive:

- `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`

Interpretation:

- If:
  - `process_alive = true`
  - `orphaned = false`
  - `stalled = false`
  then do **not** start another canonical run. Just monitor.

- If:
  - `running_cell_count = 0` while `bundle_progress < 1.0`
  - or `orphaned = true`
  - or `stalled = true`
  then recover the same bundle in detached mode. Do not delete outputs.

### 3.3 The next code to write

The next code task is **not** another liveness fix. The next code task should only be used if the canonical bundle reaches completion and final acceptance still fails.

If `bundle_complete = true` but `primary_safety_valid = false`, the next engineer should immediately implement:

- `primary acceptance failure diagnostics export`

Target files:

- `scripts/reproduce_paper.py`
- `src/apflf/analysis/stats.py`
- `tests/test_reproduce_paper.py`

Required new outputs:

- `outputs/paper_canonical/primary_safety_failure_report.json`
- `outputs/paper_canonical/primary_safety_failure_rows.csv`

### 3.4 Required math for that next code task

Let:

- `P` be the set of raw rows where `method == PRIMARY_METHOD == "no_rl"`
- `S` be the set of scenarios in the canonical manifest
- `K = {0, 1, ..., 29}`

Define per-row indicators:

- `v_collision(r) = 1[collision_count(r) > 0]`
- `v_boundary(r) = 1[boundary_violation_count(r) > 0]`
- `v_safety(r) = 1[v_collision(r) = 1 or v_boundary(r) = 1]`

Define per-scenario aggregate failure counts:

- `collision_total(s) = sum_{r in P, scenario(r)=s} collision_count(r)`
- `boundary_total(s) = sum_{r in P, scenario(r)=s} boundary_violation_count(r)`
- `violation_seed_count(s) = |{ k in K : exists r in P with scenario(r)=s, seed(r)=k, v_safety(r)=1 }|`

Acceptance logic must remain unchanged:

- `primary_safety_valid = true`
  iff
  for every scenario `s in S`:
  - `collision_total(s) = 0`
  - `boundary_total(s) = 0`

The new diagnostics layer must:

- not change acceptance logic
- only explain failure after the fact
- include offending scenario, seed, run_id, collision_count, boundary_violation_count, and config_hash

### 3.5 Next run sequence

The next engineer should continue in this order:

1. Check live status:
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
2. If live run is healthy:
   - do not restart it
   - keep monitoring every 10 minutes
3. If orphaned or stalled:
   - recover the same bundle in detached mode
4. Once bundle completes:
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
5. If:
   - `bundle_complete = true`
   - `primary_safety_valid = false`
   then write the diagnostics export code described above

### 3.6 Final acceptance target

The project only reaches final artifact completion when:

- `outputs/paper_canonical` exists
- final `run_progress.json` has `bundle_progress = 1.0`
- final `paper_acceptance.json` has `bundle_complete = true`
- final `paper_acceptance.json` has `primary_safety_valid = true`
- all expected canonical cells cover `30` seeds

---

## 4. Technical Red Lines

The following constraints must remain untouched.

### 4.1 Main architecture red lines

- The paper mainline remains:
  - `FSM + adaptive_apf + CBF-QP`
- `paper_canonical` is only for the white-box paper matrix.
- RL must not be promoted back to the mainline.

### 4.2 RL red lines

- RL remains strictly:
  - `param-only supervisor`
- Do not let RL output continuous control directly.
- Do not introduce `SB3`.

### 4.3 Public interface red lines

Do not change:

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

### 4.4 Safety red lines

- Do not modify safety red-line files.
- Do not loosen safety acceptance to chase efficiency.
- The final paper acceptance must still require:
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  for the primary method at the canonical acceptance level.

### 4.5 Statistics / paper red lines

- Keep deterministic bootstrap CI as the default main-table statistics rule.
- Keep `paper_acceptance.json` as the single paper artifact acceptance entrypoint.
- Paired deltas must continue to use same-seed pairing only.
- Failure diagnostics, if added, must be post-processing only and must not alter acceptance rules.

---

## 5. One-Line Handoff

The real blocker was no longer missing paper infrastructure but a Windows-specific liveness bug in `status-only`; that is now fixed, detached-child liveness is regression-tested, `paper_canonical` has been safely resumed under PID `25392`, the bundle currently sits at `1 / 55` complete cells with `process_alive=true` and `orphaned=false`, and the next engineer should primarily keep the canonical run healthy until completion, only adding primary safety failure diagnostics if final acceptance still fails after the full bundle is done.
