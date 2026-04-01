# AI_MEMORY - 当前周期交接文档

> 下一位 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。  
> 本文件按 `2026-04-02` 的真实仓库状态重写；旧版 AI_MEMORY 中关于 dirty worktree、旧 `HEAD`、以及“下一步去补 `status-only / validate-only`”的描述都已经过期。

---

## 0. 当前开发游标

- 日期：
  - `2026-04-02`
- Git 游标：
  - `HEAD = de0cf46ce6a2f11ee1a14132a88c82b2601e662a`
- 当前工作树：
  - 在本次重写 `AI_MEMORY.md` 之前，repo-tracked 改动只有 `2` 个文件：
    - `scripts/reproduce_paper.py`
    - `tests/test_reproduce_paper.py`
  - 本文件重写后，`AI_MEMORY.md` 也会进入 modified 状态
- 当前后台进程：
  - 当前有一个正在运行的正文 canonical 长跑：
    - `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`
  - 当前观察到的运行日志：
    - `outputs/paper_canonical_run_stdout.log`
    - `outputs/paper_canonical_run_stderr.log`
  - `stdout` 目前仍为空
  - `stderr` 目前只有 CBF-QP / fallback 警告，没有看到致命 traceback
- 当前正文 artifact 游标：
  - `outputs/paper_canonical` 已经创建
  - 当前已存在这些 bundle 产物：
    - `outputs/paper_canonical/manifest.json`
    - `outputs/paper_canonical/run_progress.json`
    - `outputs/paper_canonical/cell_progress.csv`
    - `outputs/paper_canonical/matrix_index.csv`
    - `outputs/paper_canonical/paper_acceptance.json`
  - 但截至本文件重写时，canonical 进度仍是：
    - `bundle_progress = 0.0`
    - `num_expected_cells = 55`
    - `num_complete_cells = 0`
    - `remaining_cell_count = 55`
  - 当前只看到第一个 cell 已进入运行前状态：
    - `outputs/paper_canonical/generated_configs/s1_local_minima__method__no_rl.yaml`
    - `outputs/paper_canonical/runs/s1_local_minima__no_rl/`
  - 该 cell 当前尚未落盘 `summary.csv`
- 当前正文主线：
  - 正文主方法仍然是白盒主链：`FSM + adaptive_apf + CBF-QP`
  - `paper_canonical` 只应包含 white-box 正文矩阵
- 当前 RL 游标：
  - RL 仍然只保留为附录增强候选，不是正文主方法
  - 当前官方 RL 结论仍来自：
    - `outputs/s5_rl_gate_warmstart_smoke__no_rl/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/analysis/rl_attribution/aggregate.json`
  - 当前 RL 关键结论继续保持：
    - `dominant_bottleneck = safety_engagement`
    - `rl_fallback_ratio_mean = 0.043939393939393945`
    - `gate_open_ratio_mean = 0.956060606060606`
    - `theta_change_ratio_mean = 0.956060606060606`
    - `leader_final_x_delta_mean = -0.061367621493906434`
    - `collision_count` 总和 = `0`
    - `boundary_violation_count` 总和 = `0`

### 当前最重要的结论

- RL 这条线的工程诊断已经足够清楚：
  - warm-start 与 effective-threshold 持久化已经解决了“完全进不了 gate”的结构问题
  - 但 multi-seed 效率仍劣于 `no_rl`
  - 因此 RL 当前不应继续绑架正文主线
- 当前真正还没收口的，不是算法，而是 **white-box canonical artifact 闭环**
- `manifest / matrix_index / paper_acceptance / run_progress / cell_progress / status-only / validate-only` 这些代码链路已经具备
- 但正文 canonical 真实长跑才刚启动，仍未产生任何完整 cell 的 `summary.csv`
- 也就是说，当前最紧迫的剩余问题已经从“功能缺失”变成了“长时间 canonical 运行的运行态可观测性不足”

---

## 1. 已完成工作

### 1.1 历史主线能力仍然成立

- gatefix 已完成：
  - `confidence_raw` 使用 Beta 方差校准
  - 两阈值滞回 gate 已在位
- reward_v2 已完成：
  - reward shaping
  - reward 配置化
  - PPO reward diagnostics
- training warm-start 已完成：
  - `tau_enter_start = 0.25`
  - `tau_exit_start = 0.15`
  - `gate_warmup_timesteps = 20000`
- effective runtime threshold 已完成并已贯通到：
  - runtime diagnostics
  - replay
  - RL attribution
- RL 路线当前已经完成“它能否真实介入 nominal layer”的工程诊断任务：
  - 可以进 gate
  - 但 multi-seed 结果仍不优于 white-box baseline

### 1.2 本轮源码实现：manifest-first canonical audit 已完成

- `scripts/reproduce_paper.py`
  - 已新增并落地 manifest-first audit helper：
    - `_load_manifest(...)`
    - `_manifest_list_of_str(...)`
    - `_manifest_list_of_int(...)`
    - `_manifest_expected_cells(...)`
    - `_resolve_audit_spec(...)`
  - `--status-only` / `--validate-only` 当前已改为：
    1. 若 `outputs/<exp_id>/manifest.json` 存在，则必须优先从 manifest 恢复 canonical spec
    2. 若 manifest 不存在但显式传了 `--canonical-matrix`，才允许退回 canonical 常量
    3. 若 manifest 不存在且也没传 `--canonical-matrix`，则明确失败
  - 缺 manifest 时，`main()` 当前会：
    - 向 `stderr` 打印：
      - `Manifest-driven audit requires an existing manifest.json or explicit --canonical-matrix.`
    - 返回语义码 `2`
  - `--status-only` / `--validate-only` 继续保持纯磁盘语义：
    - 不触发 simulation
    - 不触发 benchmark
    - 不触发 export 子运行
    - 只读取 `outputs/<exp_id>/...` 的现有 bundle 状态
- `tests/test_reproduce_paper.py`
  - 本轮已补齐 manifest-first 覆盖：
    - manifest 存在时，audit 必须按 manifest.expected_seeds / expected_cells 计算
    - manifest 缺失且未传 `--canonical-matrix` 时，audit 必须失败
    - `--status-only` 重复执行必须幂等
    - `--validate-only` 在 incomplete / invalid bundle 下非零，在 complete + primary-safe 下为零

### 1.3 本轮真实命令线验证已完成

- 已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py` 通过
    - 当前结果：`16 passed`
  - `python -m pytest -q tests/test_offline_reporting.py` 通过
    - 当前结果：`2 passed`
- 当前可信的定向验证基线：
  - 与本轮改动直接相关的测试链路合计 `18 passed`
- 还需诚实记录：
  - 本轮没有 fresh rerun 完整 `python -m pytest -q`
  - 因此不要宣称“本轮 full-suite fresh rerun 全绿”

### 1.4 本轮真实运行状态：`paper_canonical` 已正式启动

- 已正式启动正文 canonical 长跑：
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`
- 启动后，以下真实状态已被确认：
  - `outputs/paper_canonical` 已存在
  - `manifest.json` 已生成
  - `run_progress.json` 已生成
  - `cell_progress.csv` 已生成
  - `matrix_index.csv` 已生成
  - `paper_acceptance.json` 已生成
- 已确认新的 manifest-first audit 在真实磁盘 bundle 上可工作：
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
    - 在 **不传** `--canonical-matrix` 的情况下，仍能从磁盘 manifest 恢复 `55` 个 expected cells
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
    - 在当前 bundle 尚未完成的情况下，按预期给出“不通过”的逻辑结果
- 截至本文件重写时，canonical 的实际运行状态是：
  - 第一个 cell `s1_local_minima__no_rl` 已创建 output 目录
  - 但还没有任何一个 `summary.csv` 真正落盘
  - 因此 `bundle_progress` 仍为 `0.0`

---

## 2. 当前研究判断

- 当前项目不是“还差算法模块”，而是“还差正文 canonical 真正跑完并通过 acceptance”
- 本轮之前的主要代码缺口是：
  - `status-only / validate-only` 还没有做到 manifest-first
- 这个缺口现在已经补完
- 因此当前剩余问题不再是：
  - RL reward
  - RL gate
  - warm-start 公式
  - canonical sealing 本身
- 当前真正的剩余工程风险，已经变成：
  - **长时间 canonical 运行时，纯磁盘 ledger 在第一个 `summary.csv` 落盘之前无法区分“正在运行”与“已经卡住”**

更直白地说：

- 现在 `paper_canonical` 已经开跑
- 但 `run_progress.json` 目前只能告诉我们：
  - `bundle_progress = 0.0`
  - 所有 cell 仍是 missing / incomplete
- 它还不能显式告诉我们：
  - 当前是否有一个 cell 正在运行
  - 当前正在跑哪一个 cell
  - 当前这一个 cell 已经跑了多久
  - 当前是否已经“长时间无 heartbeat”

也就是说，当前下一段真正值得写的代码，不是再去碰算法，而是给 canonical 长跑补 **运行态 heartbeat / running-cell journal**。

---

## 3. 下一步指令

### 3.1 总原则

下一位工程师启动 AI 后：

- 不要再改 RL reward
- 不要再改 RL gate
- 不要再改 warm-start 公式
- 不要再把 RL 当成当前主任务
- 不要重复重写 manifest-first audit
- 不要删除当前正在运行的 `outputs/paper_canonical`

下一条立即执行的代码任务，应围绕 **white-box canonical 长跑的运行态 heartbeat / running-cell journal** 展开。

### 3.2 立即执行的代码任务

在以下文件中实现 **canonical runtime heartbeat / running-cell journal**：

- `scripts/reproduce_paper.py`
- `tests/test_reproduce_paper.py`

如确有必要，再最小增量触碰：

- `src/apflf/analysis/stats.py`
- `tests/test_stats_export.py`

优先复用当前已有能力：

- `_write_sealed_artifacts(...)`
- `_collect_disk_rows(...)`
- `validate_canonical_bundle(...)`
- `summarize_canonical_progress(...)`
- 当前已有的 `manifest.json`
- 当前已有的 `run_progress.json`
- 当前已有的 `cell_progress.csv`

不要另起一套平行 canonical validator。

### 3.3 这段代码必须产出什么

为 `outputs/<exp_id>/` 新增并维护下面两类运行态产物：

- `run_runtime_state.json`
- `cell_runtime_state.csv`

这两类产物的职责是：

- 在 **第一个 `summary.csv` 落盘之前**，也能让操作者知道 canonical 长跑是不是还活着
- 显式记录：
  - 当前是否有 cell 正在运行
  - 当前正在运行哪个 cell
  - 该 cell 的开始时间
  - 最近一次 heartbeat 时间
  - 当前 cell 的运行状态

### 3.4 必须满足的状态机与数学约束

记：

- `S = manifest.expected_seeds`
- `C = manifest.expected_cells`
- `|S| = 30`
- `|C| = 55`

对任意时刻 `t`、任意 cell `c ∈ C`，定义：

- `completed_seed_count_t(c)`：
  - 从磁盘上已经落盘的 `summary.csv` 中读出的已完成 seed 数
- `completed_progress_t(c) = completed_seed_count_t(c) / |S|`
- `bundle_completed_progress_t = (1 / (|C| * |S|)) * Σ_c completed_seed_count_t(c)`

必须满足：

- `0 <= completed_seed_count_t(c) <= |S|`
- `0 <= completed_progress_t(c) <= 1`
- `0 <= bundle_completed_progress_t <= 1`
- 在不删除已有结果的前提下：
  - `completed_seed_count_t(c)` 对 `t` 单调不减
  - `completed_progress_t(c)` 对 `t` 单调不减
  - `bundle_completed_progress_t` 对 `t` 单调不减

新增运行态状态机：

- `runtime_status_t(c) ∈ {pending, running, complete, failed}`

并约束：

- `running_cell_count_t = |{ c ∈ C : runtime_status_t(c) = running }|`
- 由于 `reproduce_paper.py` 当前按 cell 串行执行，必须始终满足：
  - `running_cell_count_t ∈ {0, 1}`

新增 heartbeat 相关量：

- `started_at_t(c)`：cell 开始运行时间
- `last_heartbeat_t(c)`：最近一次 heartbeat 时间
- `finished_at_t(c)`：cell 完成或失败时间
- `heartbeat_age_t(c) = max(0, now_t - last_heartbeat_t(c))`

默认 stall 阈值固定为：

- `H = 900` 秒

定义：

- `stalled_t(c) = 1[runtime_status_t(c) = running and heartbeat_age_t(c) > H]`

必须满足：

- 当 cell 刚进入运行时：
  - `runtime_status = running`
  - `started_at = last_heartbeat`
- 当 cell 正常完成时：
  - `runtime_status: running -> complete`
  - `finished_at` 必须写入
- 当 cell 运行异常退出时：
  - `runtime_status: running -> failed`
  - `finished_at` 必须写入
- `status-only` 必须基于纯磁盘同时输出：
  - `bundle_completed_progress`
  - `running_cell_count`
  - `running / complete / failed / pending` 的 cell 数量
  - 若存在运行中 cell，则输出其：
    - `scenario`
    - `variant_type`
    - `variant_name`
    - `heartbeat_age_seconds`
    - `stalled`

重要约束：

- `validate-only` 的 acceptance 口径 **不能** 因 heartbeat 而放松
- 也就是说：
  - `bundle_complete` 仍然只能由已完成并落盘的 `summary.csv` 决定
  - 运行中 cell 不能被算成 complete

### 3.5 下一轮测试要求

必须新增并通过以下测试：

- `runtime_state` 文件生成测试：
  - 启动一个 partial canonical run 时，`run_runtime_state.json` 与 `cell_runtime_state.csv` 必须生成
- 串行状态机测试：
  - 任意时刻 `running_cell_count ∈ {0, 1}`
- heartbeat / stall 判定测试：
  - `heartbeat_age_seconds > 900` 时，`stalled = true`
  - 否则 `stalled = false`
- 幂等性测试：
  - 对同一磁盘状态重复执行 `--status-only`
  - `run_progress.json`
  - `cell_progress.csv`
  - `run_runtime_state.json`
  - `cell_runtime_state.csv`
  - `matrix_index.csv`
  - `paper_acceptance.json`
  数据内容必须稳定一致
- acceptance 不变性测试：
  - 即使 runtime state 显示 `running`
  - 若 `summary.csv` 尚未完整，则 `validate-only` 仍必须失败

### 3.6 下一轮运行与验收顺序

下一位工程师应按这个顺序继续：

1. 先检查当前 `paper_canonical` 进程是否仍在运行
2. 若仍在运行，不要删除现有 `outputs/paper_canonical`
3. 先实现 runtime heartbeat / running-cell journal
4. 通过以下验证：
   - `python -m compileall src tests scripts`
   - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
5. 然后再继续/恢复正文 canonical：
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`
6. 再执行：
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`

最终验收标准仍固定为：

- `outputs/paper_canonical` 存在
- `run_progress.json` 最终 `bundle_progress = 1.0`
- `paper_acceptance.json` 中 `bundle_complete = true`
- `paper_acceptance.json` 中 `primary_safety_valid = true`
- 所有 expected canonical cells 覆盖 `30` seeds

---

## 4. 技术栈红线

以下要求必须继续严格保留，不得被后续工程师破坏：

### 4.1 主架构红线

- 正文主线始终是白盒主链：
  - `FSM + adaptive_apf + CBF-QP`
- `paper_canonical` 只服务于 white-box 正文矩阵
- RL 不得重新升级为当前主线

### 4.2 RL 红线

- RL 仍严格限制为：
  - `param-only supervisor`
- 不允许回滚成：
  - `mode-only RL`
  - `full supervisor continuous control`
- 不允许让 RL 直接输出连续控制量
- 不允许引入 `SB3`
- `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md` 里保留的旧 `mode-only RL` 理论模板不是当前活契约

### 4.3 公共接口红线

以下接口不得修改：

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

### 4.4 安全层红线

- 不允许修改 safety red-line 文件
- 不允许为了追求效率而放松 CBF-QP 的 safety acceptance 口径

### 4.5 统计与论文口径红线

- 默认主表统计继续使用 deterministic bootstrap CI
- `paper_acceptance.json` 必须继续作为正文 artifact 的唯一收口入口
- paired delta 继续只允许按相同 seed 配对

---

## 5. 一句话交接

当前最重要的事实是：

- manifest-first canonical audit 已经完成
- `paper_canonical` 正文长跑已经启动
- RL 线当前不再是主任务
- 下一位工程师不该再改算法，而应该立刻补 **canonical 长跑的运行态 heartbeat / running-cell journal**，然后继续把 `paper_canonical` 跑到 `bundle_complete = true`
