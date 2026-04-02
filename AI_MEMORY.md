# AI_MEMORY - Current Handoff Memo

> Read this file first, then read `PROMPT_SYSTEM.md` and `RESEARCH_GOAL.md`, and only then touch code.
> This file is intentionally ASCII-only to avoid Windows editor / shell encoding regressions.
> It reflects the real repository and disk state on `2026-04-02`.

---

## 0. Current Cursor

- Date:
  - `2026-04-02`
- Git cursor:
  - `HEAD = 8831705a25e872d74f5b0967d2a616e37f7ac793`
- Worktree snapshot at rewrite time:
  - repo-tracked modified files:
    - `AI_MEMORY.md`
  - there are no other repo-tracked modified source files at this moment

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
- Current bundle truth from disk:
  - `bundle_progress = 0.0`
  - `bundle_completed_progress = 0.0`
  - `num_expected_cells = 55`
  - `num_complete_cells = 0`
  - `num_running_cells = 1`
  - `num_pending_cells = 54`
  - there is still no `summary.csv` anywhere under `outputs/paper_canonical`
- Current active runtime state from `run_runtime_state.json`:
  - active cell:
    - `scenario = s1_local_minima`
    - `variant_type = method`
    - `variant_name = no_rl`
    - `method = no_rl`
    - `run_id = s1_local_minima__no_rl`
  - active cell state:
    - `runtime_status = running`
    - `completed_seed_count = 0`
    - `completed_progress = 0.0`
    - `stalled = false`
    - `started_at = 2026-04-02T05:03:13Z`
    - `last_heartbeat` is being refreshed
- Current processes seen at rewrite time:
  - likely active canonical process:
    - `PID 5196`, start time `2026-04-02 13:03:12`, CPU has continued increasing
  - another older python process also exists:
    - `PID 9412`
- Current logs:
  - `outputs/paper_canonical_run_stdout.log` is still empty
  - `outputs/paper_canonical_run_stderr.log` still shows repeated:
    - `solver_status=maximum iterations reached`
    - `preview_violation_after_qp`
  - no fatal traceback has been observed yet

### Most important conclusion

- RL is no longer the main engineering target.
- The actual remaining gap is not algorithm design anymore.
- The real remaining gap is:
  - finish the white-box `paper_canonical` bundle
  - improve process-aware observability of the long canonical run
- Manifest-first canonical audit already exists.
- Runtime heartbeat / running-cell journal already exists.
- What is still missing is final canonical completion, final acceptance, and process-aware orphan detection.

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
- Training warm-start completed:
  - `tau_enter_start = 0.25`
  - `tau_exit_start = 0.15`
  - `gate_warmup_timesteps = 20000`
- Effective runtime threshold persistence completed:
  - runtime diagnostics
  - replay
  - RL attribution
- RL engineering conclusion is already strong enough:
  - RL can enter the gate and affect nominal behavior
  - but multi-seed efficiency still loses to `no_rl`
  - therefore RL remains appendix-only

### 1.2 Manifest-first canonical audit completed

- `scripts/reproduce_paper.py` already supports:
  - `--status-only`
  - `--validate-only`
- Both are manifest-first:
  1. if `outputs/<exp_id>/manifest.json` exists, restore canonical spec from manifest
  2. if manifest is missing but `--canonical-matrix` is explicitly provided, fallback to canonical constants
  3. if manifest is missing and `--canonical-matrix` is not provided, exit nonzero with semantic code `2`
- Pure-disk audit artifacts already exist:
  - `manifest.json`
  - `run_progress.json`
  - `cell_progress.csv`
  - `matrix_index.csv`
  - `paper_acceptance.json`
- Audit no longer depends on parser default seeds / scenarios / methods / ablations.

### 1.3 Runtime heartbeat / running-cell journal completed

- `scripts/reproduce_paper.py` now also writes:
  - `run_runtime_state.json`
  - `cell_runtime_state.csv`
- Runtime fields already implemented:
  - `runtime_status in {pending, running, complete, failed}`
  - `started_at`
  - `last_heartbeat`
  - `finished_at`
  - `heartbeat_age_seconds`
  - `stalled`
  - `completed_seed_count`
  - `completed_progress`
- Heartbeat thread already implemented:
  - fixed heartbeat period `30` seconds
  - updates runtime journal only
  - does not relax acceptance logic
- Serial execution invariant already enforced:
  - `running_cell_count in {0, 1}`
  - if multiple rows appear as `running`, only the newest heartbeat stays `running`
- `--status-only` prints runtime summary:
  - `bundle_completed_progress`
  - `running_cell_count`
  - `num_pending_cells`
  - `num_running_cells`
  - `num_complete_cells`
  - `num_failed_cells`
  - active cell details if a running cell exists
- `--validate-only` acceptance invariance is preserved:
  - only sealed summary data can make a cell complete
  - runtime `running` does not count as complete

### 1.4 Current known verification baseline

- Freshly passed in the previous implementation round:
  - `python -m compileall src tests scripts`
  - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
- Fresh targeted result from that round:
  - `20 passed`
- Not freshly rerun in this rewrite-only turn:
  - full `python -m pytest -q`
- Therefore do not claim "fresh full-suite green" for this rewrite-only turn.

### 1.5 Real command-line behavior already verified

- Verified on the real partial bundle:
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
- Verified behavior:
  - `status-only` exposes live runtime summary on the real partial canonical bundle
  - `validate-only` still returns incomplete / failing status while no `summary.csv` has landed
- This proves:
  - runtime heartbeat is connected to the real canonical long run
  - acceptance logic has not been polluted by runtime state

### 1.6 Encoding incident already diagnosed

- `AI_MEMORY.md` had previously become mojibake.
- That corruption was checked explicitly.
- The corruption was isolated to `AI_MEMORY.md`.
- The modified code files from the heartbeat round were checked and did not show the same mojibake patterns.
- This file is kept ASCII-only on purpose to prevent recurrence.

---

## 2. Current Research Judgment

- The project is no longer blocked by missing algorithm modules.
- The remaining blocker is:
  - the white-box canonical artifact is not finished yet
- More concretely:
  - canonical sealing exists
  - manifest-first audit exists
  - runtime heartbeat exists
  - but the bundle is still on the very first cell
  - and there is still no `summary.csv`
- So the next meaningful work should not be:
  - RL reward changes
  - RL gate changes
  - warm-start changes
  - rewriting audit again
- The next meaningful work is:
  - process-aware liveness
  - orphan reconciliation
  - then continue canonical until acceptance completes

In short:

- We are close to code closure.
- We are not yet at paper-artifact closure.
- The remaining engineering risk is operational observability of the long canonical run plus final acceptance.

---

## 3. Next Instruction

### 3.1 General rule

The next engineer / AI must:

- not change RL reward
- not change RL gate
- not change warm-start math
- not rewrite manifest-first audit
- not delete `outputs/paper_canonical`
- not treat runtime journal as acceptance

The immediate next coding task is:

- `process-aware liveness / orphan reconciliation`

on top of the already implemented runtime heartbeat.

### 3.2 Files to edit

Primary files:

- `scripts/reproduce_paper.py`
- `tests/test_reproduce_paper.py`

Only touch these additional files if strictly necessary:

- `src/apflf/analysis/stats.py`
- `tests/test_stats_export.py`

Reuse existing mechanisms:

- heartbeat thread
- `run_runtime_state.json`
- `cell_runtime_state.csv`
- `status-only`
- `validate-only`
- `manifest.json`
- `run_progress.json`
- `cell_progress.csv`

Do not create a separate watchdog script.

### 3.3 New fields that must be added

Add the following runtime fields:

- `runner_pid`
- `runner_started_at`
- `process_alive`
- `orphaned`

They should be persisted at least in:

- `run_runtime_state.json`
- `cell_runtime_state.csv`

Definitions:

- `runner_pid`:
  - PID of the Python process responsible for that cell run
- `runner_started_at`:
  - process start timestamp for that PID
- `process_alive`:
  - whether the recorded PID still refers to the same live process
- `orphaned`:
  - the cell is still marked `running`, but the recorded process is no longer alive

### 3.4 Math and state constraints that must hold

Let:

- `S = manifest.expected_seeds`
- `C = manifest.expected_cells`
- `|S| = 30`
- `|C| = 55`
- `H = 900` seconds
- process start matching tolerance:
  - `Delta = 5` seconds

For any time `t` and any cell `c in C`, define:

- `completed_seed_count_t(c)`:
  - number of completed seeds read from disk `summary.csv`
- `completed_progress_t(c) = completed_seed_count_t(c) / |S|`
- `bundle_completed_progress_t = (1 / (|C| * |S|)) * sum_c completed_seed_count_t(c)`
- `heartbeat_age_t(c) = max(0, now_t - last_heartbeat_t(c))`
- `process_alive_t(c) = 1[exists process p such that pid(p)=runner_pid(c) and |start_time_utc(p)-runner_started_at(c)| <= Delta]`
- `stalled_t(c) = 1[runtime_status_t(c)=running and heartbeat_age_t(c) > H]`
- `orphaned_t(c) = 1[runtime_status_t(c)=running and process_alive_t(c)=0]`

Required constraints:

- `0 <= completed_seed_count_t(c) <= |S|`
- `0 <= completed_progress_t(c) <= 1`
- `0 <= bundle_completed_progress_t <= 1`
- without deleting already finished results:
  - `completed_seed_count_t(c)` is monotone nondecreasing in `t`
  - `completed_progress_t(c)` is monotone nondecreasing in `t`
  - `bundle_completed_progress_t` is monotone nondecreasing in `t`
- `running_cell_count_t in {0, 1}`
- `process_alive_t(c) in {0, 1}`
- `stalled_t(c) in {0, 1}`
- `orphaned_t(c) in {0, 1}`

State semantics:

- when a cell starts:
  - `runtime_status = running`
  - `started_at = last_heartbeat`
  - `runner_pid` and `runner_started_at` must be written
- on normal completion:
  - `runtime_status: running -> complete`
  - `finished_at` must be written
- on explicit failure:
  - `runtime_status: running -> failed`
  - `finished_at` must be written
- on external kill / manual stop where failure cannot be written:
  - do not fake `failed`
  - expose the situation in `status-only` via:
    - `stalled = true`
    - `process_alive = false`
    - `orphaned = true`

Acceptance invariants that must remain unchanged:

- `validate-only` must still depend only on sealed summary data
- running / stalled / orphaned cells must not count as complete
- `primary_safety_valid` rule must remain unchanged

### 3.5 Tests that must be added next

The next round must add and pass:

- runtime process field generation test:
  - `run_runtime_state.json` and `cell_runtime_state.csv` must contain:
    - `runner_pid`
    - `runner_started_at`
    - `process_alive`
    - `orphaned`
- liveness test:
  - when PID exists and start time matches:
    - `process_alive = true`
  - otherwise:
    - `process_alive = false`
- orphan detection test:
  - if `runtime_status = running` and `process_alive = false`:
    - `orphaned = true`
- stall/orphan decoupling test:
  - `heartbeat_age_seconds > 900` implies `stalled = true`
  - `stalled` and `orphaned` must not be treated as the same signal
- pure-disk idempotence test for `status-only`:
  - repeated runs on the same disk state must keep these files data-identical:
    - `run_progress.json`
    - `cell_progress.csv`
    - `run_runtime_state.json`
    - `cell_runtime_state.csv`
    - `matrix_index.csv`
    - `paper_acceptance.json`
- acceptance invariance test:
  - even if runtime says `running`
  - even if `process_alive = true`
  - even if `orphaned = false`
  - if `summary.csv` is still incomplete, `validate-only` must still fail

### 3.6 Next run sequence

The next engineer should continue in this order:

1. Check whether `paper_canonical` is still running.
2. If it is running, do not delete `outputs/paper_canonical`.
3. Implement process-aware liveness / orphan reconciliation.
4. Verify:
  - `python -m compileall src tests scripts`
  - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
5. Then continue or resume canonical:
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`
6. Then run:
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`

Final acceptance target remains:

- `outputs/paper_canonical` exists
- final `run_progress.json` has `bundle_progress = 1.0`
- `paper_acceptance.json` has `bundle_complete = true`
- `paper_acceptance.json` has `primary_safety_valid = true`
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
- Do not revert to:
  - `mode-only RL`
  - `full supervisor continuous control`
- Do not let RL output continuous control directly.
- Do not introduce `SB3`.
- The old `mode-only RL` templates in `PROMPT_SYSTEM.md` and `RESEARCH_GOAL.md` are not the live contract.

### 4.3 Public interface red lines

Do not change:

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

### 4.4 Safety red lines

- Do not modify safety red-line files.
- Do not loosen CBF-QP safety acceptance to chase efficiency.

### 4.5 Statistics / paper red lines

- Keep deterministic bootstrap CI as the default main-table statistics rule.
- Keep `paper_acceptance.json` as the single paper artifact acceptance entrypoint.
- Paired deltas must continue to use same-seed pairing only.

---

## 5. One-Line Handoff

Manifest-first canonical audit is done, runtime heartbeat / running-cell journal is done, `paper_canonical` is currently running its first cell with no `summary.csv` yet, RL is no longer the main task, and the next engineer should immediately implement `process-aware liveness / orphan reconciliation` and then keep driving `paper_canonical` to `bundle_complete = true`.
