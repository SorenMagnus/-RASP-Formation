# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件按 `2026-04-01` 的真实仓库状态重写；旧版 AI_MEMORY 中关于“下一步去补 `status-only / validate-only`”的叙述已经过期，因为那部分代码本轮已经完成。

---

## 0. 当前开发游标

- 日期：
  - `2026-04-01`
- Git 游标：
  - `HEAD = 4360a95786611575a61d34ebe2dbb0311ea2b776`
- 当前工作树：
  - 当前 repo-tracked 改动有 `4` 个文件：
    - `AI_MEMORY.md`
    - `scripts/reproduce_paper.py`
    - `src/apflf/analysis/stats.py`
    - `tests/test_stats_export.py`
  - 当前 untracked 文件有 `1` 个：
    - `tests/test_reproduce_paper.py`
- 当前后台进程：
  - 当前没有活动中的 `train_rl_supervisor.py`
  - 当前没有活动中的 `benchmark_s5_rl.py`
  - 当前没有活动中的 `reproduce_paper.py`
  - 当前仅看到 VS Code / language server 相关 `python` 进程，不属于实验运行
- 当前论文主线：
  - 正文主方法仍然是白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 仍然只是“附录增强 / 可选增强候选”，不是正文主方法
  - `outputs/paper_canonical` 当前仍不存在，说明 canonical artifact 还没有真正落盘
- 当前 RL 结论游标：
  - 当前官方最新 RL 结论仍来自：
    - `outputs/s5_rl_gate_warmstart_smoke__no_rl/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/analysis/rl_attribution/aggregate.json`
  - 本轮没有新的 RL 算法结论，仍保持：
    - `dominant_bottleneck = safety_engagement`
    - `rl_fallback_ratio_mean = 0.043939393939393945`
    - `gate_open_ratio_mean = 0.956060606060606`
    - `theta_change_ratio_mean = 0.956060606060606`
    - `leader_final_x_delta_mean = -0.061367621493906434`
    - `collision_count` 总和 = `0`
    - `boundary_violation_count` 总和 = `0`

### 当前最重要的结论

- RL 这一条线的研究判断已经足够清楚：
  - warm-start 和 effective-threshold 持久化已经解决了“完全进不了 gate”的结构性问题
  - 但 multi-seed 效率仍劣于 `no_rl`
  - 因此 RL 当前不应继续绑架正文主线
- 当前真正还没收口的不是算法，而是 **white-box canonical artifact 闭环**
- 本轮已经把 `paper_canonical` 所需的：
  - `validator`
  - `manifest sealing`
  - `incremental progress ledger`
  - `status-only / validate-only`
  都写出来了
- 但 `outputs/paper_canonical` 仍然缺失，因为真正的 canonical matrix 还没有正式跑完
- 更关键的是：当前 `status-only / validate-only` 虽然已经存在，但**还没有做到 manifest 驱动的规范恢复**；如果调用者不传 `--canonical-matrix`，它会回退到 parser 默认 `seeds=[0,1,2]`，这对 `paper_canonical` 的纯磁盘验收是不安全的

---

## 1. 已完成工作

### 1.1 历史主线能力仍然成立

- gatefix 已完成：
  - `confidence_raw` 使用 Beta 方差校准
  - 两阈值滞回 gate 已经在位
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
- RL 路线当前已经完成“能否真正介入 nominal layer”的工程诊断任务：
  - 可以进 gate
  - 但 multi-seed 仍不优于白盒 baseline

### 1.2 本轮源码实现：canonical sealing / progress ledger / pure-disk audit 已落地

- `src/apflf/analysis/stats.py`
  - 已有 `validate_canonical_bundle(...)`
  - 已有 `summarize_canonical_progress(...)`
  - 当前 bundle 级字段已经包括：
    - `status`
    - `progress_ratio`
    - `actual_seed_count`
    - `complete`
    - `bundle_complete`
    - `primary_safety_valid`
    - `remaining_cell_count`
    - `bundle_progress`
  - validator / progress 链路当前能显式检查与汇总：
    - canonical seed completeness
    - `config_hash` consistency
    - paired seed alignment coverage
    - missing / invalid / unexpected cells
    - bundle 级进度与完成状态
  - 现有 deterministic bootstrap CI 默认口径没有被改动

- `scripts/reproduce_paper.py`
  - 已支持测试友好的入口：
    - `main(argv: list[str] | None = None)`
  - 已补齐磁盘重建相关辅助函数：
    - `_collect_disk_rows(...)`
    - `_cell_progress_rows(...)`
    - `_write_sealed_artifacts(...)`
    - `_print_bundle_audit_summary(...)`
  - 当前脚本会在启动时和每个 cell 完成后：
    - 从磁盘重建当前 bundle 状态
    - 写出 `run_progress.json`
    - 写出 `cell_progress.csv`
    - 刷新 `manifest.json`
    - 刷新 `matrix_index.csv`
    - 刷新 `paper_acceptance.json`
  - `--skip-existing` 当前已经基于磁盘真值恢复进度，而不是依赖内存缓存
  - 本轮新增了纯磁盘 CLI 模式：
    - `--status-only`
    - `--validate-only`
  - 这两个模式当前已经满足：
    - 不启动 simulation / benchmark / training
    - 只读磁盘 bundle
    - 刷新并输出 ledger / acceptance
    - `validate-only` 按 bundle 完整性和 `primary_safety_valid` 返回退出码
  - 当前 sealing / audit 产物链路已经包括：
    - `manifest.json`
    - `matrix_index.csv`
    - `paper_acceptance.json`
    - `run_progress.json`
    - `cell_progress.csv`

- `tests/test_stats_export.py`
  - 已新增 / 扩展 validator 测试
  - 已覆盖：
    - missing seeds
    - `config_hash` mismatch
    - `status`
    - `progress_ratio`
    - `bundle_progress`
    - invalid / incomplete bundle 报告

- `tests/test_reproduce_paper.py`
  - 本轮新建并继续扩展
  - 当前已覆盖：
    - `manifest.json` 生成
    - `matrix_index.csv` 生成
    - `paper_acceptance.json` 生成
    - `run_progress.json` 生成
    - `cell_progress.csv` 生成
    - partial bundle / invalid bundle 分支
    - `--skip-existing` 从磁盘恢复进度
    - `--status-only` 不触发新 run
    - `--validate-only` 在 incomplete / invalid bundle 下返回非零
    - `--validate-only` 在完整 bundle 下返回零

### 1.3 本轮验证状态

- 本轮已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py` 通过
    - 当前结果：`14 passed`
  - `python -m pytest -q tests/test_offline_reporting.py` 通过
    - 当前结果：`2 passed`
- 当前可信的结论是：
  - 与本轮改动直接相关的 canonical sealing / progress ledger / pure-disk audit 链路是绿的
  - 本轮直接相关的定向测试合计 `16 passed`
- 需要诚实记录：
  - 本轮 fresh `python -m pytest -q` 没有完整跑到结束
  - 之前的 full-suite rerun 曾在约 `10` 分钟处超时并被停止
  - 因此不要宣称“本轮 full-suite fresh rerun 全绿”

### 1.4 本轮运行验证与残留状态

- 当前还没有正式的 `paper_canonical` 运行产物
- 也就是说：
  - canonical sealing / ledger / audit 代码已经准备好
  - 但真正的 `outputs/paper_canonical` 仍不存在
- 当前没有后台残留 canonical 运行
- 当前没有半成品 smoke 输出需要清理

---

## 2. 当前研究判断

- 当前项目的真实状态不是“缺算法模块”，而是“白盒论文闭环只差最后一公里”
- 这一公里现在分成三个层次：
  1. 代码层已经补到位：
     - validator 有了
     - manifest 有了
     - acceptance sealing 有了
     - progress ledger 有了
     - `status-only / validate-only` 有了
  2. 纯磁盘审计仍差一个规范恢复缺口：
     - 当前 audit 模式还没有把 `manifest.json` 当成唯一的 canonical spec 来源
     - 如果用户直接运行：
       - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
       - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
       在没有 `--canonical-matrix` 的情况下，脚本会按 parser 默认 `seeds=[0,1,2]`、默认 methods / ablations 解释 bundle
     - 这会让纯磁盘验收结果依赖 CLI 默认值，而不是依赖磁盘真值
  3. artifact 层还没落盘：
     - `paper_canonical` 还没真正运行完成
     - 因此还没有唯一、可验收、可恢复的最终 bundle

- RL 分支当前的研究地位必须继续保持：
  - 它已经完成了“证明自己能不能真正介入 nominal layer”的任务
  - 但 multi-seed 结果已经证明它目前不优于 white-box baseline
  - 因此近期不要再把机器时间优先花在 RL 局部调参上

- 本轮之后，真正合理的路线是：
  1. 不再新增 RL 算法修改
  2. 先把 canonical 审计入口做成 **manifest 驱动**
  3. 然后正式跑 `paper_canonical`
  4. 最后以 `paper_acceptance.json` 为唯一验收入口收口正文主线

---

## 3. 下一步指令

### 3.1 总原则

下一个工程师启动 AI 后：

- 不要再改 RL reward
- 不要再改 RL gate
- 不要再改 warm-start 公式
- 不要再把 RL 当成当前主任务
- 不要重复写已经完成的 `progress ledger`
- 不要重复写已经完成的 `status-only / validate-only`
- 下一条立即执行的代码任务，必须围绕 **manifest 驱动的 canonical spec 恢复** 展开

### 3.2 立即执行的代码任务

在以下文件中实现 **manifest-driven audit bootstrap / canonical spec recovery**：

- `scripts/reproduce_paper.py`
- `tests/test_reproduce_paper.py`
- 如确有必要，再最小增量修改：
  - `src/apflf/analysis/stats.py`
  - `tests/test_stats_export.py`

优先复用现有：

- `_write_sealed_artifacts(...)`
- `_collect_disk_rows(...)`
- `validate_canonical_bundle(...)`
- `summarize_canonical_progress(...)`

不要再新增一套平行 audit 逻辑。

### 3.3 这段代码必须完成什么

当前已经有：

- `--status-only`
- `--validate-only`

但它们还不够“纯 canonical”，因为它们目前仍依赖 parser 默认：

- `seeds=[0,1,2]`
- 默认 scenarios
- 默认 methods
- 默认 ablations

这对 `paper_canonical` 是不安全的。  
因此下一条代码任务必须把 audit 模式改成下面的优先级：

#### A. manifest 存在时

若 `outputs/<exp_id>/manifest.json` 存在，且当前运行的是：

- `--status-only`
- 或 `--validate-only`

则必须：

- 直接从 `manifest.json` 恢复：
  - `canonical_matrix`
  - `expected_seeds`
  - `expected_scenarios`
  - `expected_methods`
  - `expected_ablations`
  - `expected_cells`
- 这些恢复值必须覆盖 parser 默认值
- 此时 CLI 中显式传入的默认矩阵参数不应影响 acceptance 结果

#### B. manifest 不存在时

若 audit 模式下 `manifest.json` 不存在：

- 若显式传了 `--canonical-matrix`
  - 允许用 canonical constants 构造 expected spec
- 否则必须：
  - 明确失败
  - 返回非零退出码
  - 并提示：
    - 需要现有 manifest
    - 或显式传入 `--canonical-matrix`

也就是说，不能再 silent 地回退到 parser 默认 `seeds=[0,1,2]` 来“假装完成验收”。

### 3.4 数学与状态约束

这些约束必须原样执行：

#### A. manifest 为 source of truth 时

- 若 manifest 存在，记：
  - `S_manifest = manifest.expected_seeds`
  - `C_manifest = manifest.expected_cells`
- 对任意 canonical cell `c ∈ C_manifest`：
  - `Seeds(c) = 从磁盘 summary.csv 读到的唯一 seed 集合`
  - `observed_seed_count(c) = |Seeds(c) ∩ S_manifest|`
  - `progress_ratio(c) = observed_seed_count(c) / |S_manifest|`
- 必须满足：
  - `0 <= progress_ratio(c) <= 1`
  - audit 结果不得依赖 parser 默认 seed 集合

#### B. bundle 进度定义固定

- 令 `C = C_manifest`
- 定义：
  - `bundle_progress = (1 / (|C| * |S_manifest|)) * Σ_c observed_seed_count(c)`
- 必须满足：
  - `0 <= bundle_progress <= 1`
  - 在不删除已有结果前提下，`bundle_progress` 单调不减
  - `bundle_progress = 1` 当且仅当所有 `c ∈ C` 达到 `complete`

#### C. 审计幂等性约束

- 对同一个磁盘状态，重复执行：
  - `python scripts/reproduce_paper.py --exp-id <exp_id> --status-only`
  - `python scripts/reproduce_paper.py --exp-id <exp_id> --validate-only`
- 若磁盘 bundle 没变，则输出的：
  - `run_progress.json`
  - `cell_progress.csv`
  - `matrix_index.csv`
  - `paper_acceptance.json`
  的**数据内容**必须不变
- 即 audit 入口必须是 deterministic / idempotent 的纯磁盘函数

#### D. 状态与 acceptance 约束继续不变

- `status(c)` 继续按现有规则：
  - `missing`：`observed_seed_count = 0`
  - `partial`：`0 < observed_seed_count < |S_manifest|` 且无 invalid 条件
  - `invalid`：duplicate seeds / unexpected seeds / `config_hash_consistent = false`
  - `complete`：`Seeds(c) = S_manifest` 且 `config_hash_consistent = true`
- `bundle_complete = true` 当且仅当所有 expected cells 都是 `complete`
- `primary_safety_valid = true` 当且仅当 primary method=`no_rl` 在每个 scenario 上：
  - `Σ collision_count = 0`
  - `Σ boundary_violation_count = 0`

#### E. 输出顺序与接口约束

- `status-only / validate-only` 都必须是纯磁盘函数：
  - 不允许启动 simulation / benchmark / export 子运行
  - 不允许依赖内存缓存恢复状态
- invalid / missing / unexpected cells 的输出顺序必须稳定：
  - 按 `(scenario, variant_type, variant_name, method)` 排序
- 默认统计口径继续保持：
  - reference method = `no_rl`
  - paired delta 只允许同 seed 配对
  - 默认主表 CI 继续使用 deterministic bootstrap
  - 不允许把默认主表 CI 改回 t-interval

### 3.5 下一轮执行顺序

#### A. 先写 manifest-driven audit bootstrap 并补测试

必须覆盖：

- manifest 存在时：
  - `--status-only` 直接按 manifest.expected_seeds / expected_cells 验收
  - 不依赖 parser 默认 `[0,1,2]`
- manifest 不存在时：
  - `--status-only` / `--validate-only` 在无 `--canonical-matrix` 情况下非零失败
- `--validate-only` 退出码继续正确：
  - incomplete / invalid 非零
  - complete + primary-safe 为零
- 重复运行 audit 时，磁盘输出内容稳定

#### B. 再正式运行 canonical matrix

运行：

- `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`

注意：

- 这一步只跑 white-box canonical matrix
- 不要把 RL method 混进 canonical bundle
- RL 分支此时不应继续阻塞正文 artifact 落盘

#### C. 再做 manifest-driven 纯磁盘验收

至少执行：

- `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
- `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`

并确保此时即便**不再传 `--canonical-matrix`**，脚本也会基于磁盘 manifest 做正确验收。

#### D. 下一轮验收标准

只有满足以下条件，才算 white-box 论文闭环真正进入最终冲刺：

- `outputs/paper_canonical` 存在
- `--status-only` 在不传 `--canonical-matrix` 时仍能正确工作
- `--validate-only` 在不传 `--canonical-matrix` 时仍能基于 manifest 正确返回退出码
- `run_progress.json` 中最终：
  - `bundle_progress = 1.0`
- `paper_acceptance.json` 中：
  - `bundle_complete = true`
  - `primary_safety_valid = true`
- canonical matrix 中每个 expected cell 都完整覆盖 `30` seeds
- 主表、图、显著性结果都有唯一、稳定、可重建的产物目录

---

## 4. 技术栈红线

### 4.1 架构红线

- 必须保持三层白盒闭环：
  - `Mode Decision -> Nominal Controller -> Safety Filter`
- 正文主线不能改写成端到端黑盒 RL
- 当前 RL 只能是 `param-only supervisor`

### 4.2 安全层红线

严禁修改以下文件：

- `src/apflf/safety/safety_filter.py`
- `src/apflf/safety/cbf.py`
- `src/apflf/safety/qp_solver.py`

### 4.3 接口红线

严禁破坏以下公共接口与数据结构：

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

### 4.4 RL 红线

- 不允许把 RL 扩成 `mode-only`
- 不允许把 RL 扩成 `full supervisor`
- 不允许 RL 直接输出 `mode`
- 不允许 RL 直接输出 `accel`
- 不允许 RL 直接输出 `steer`
- 不允许引入 `SB3`
- PPO 后端必须继续沿用仓库内自定义 Torch 实现
- 需要特别澄清：
  - `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md` 里旧的 “mode-only RL” 理论模板不是当前活契约
  - 当前活契约仍然是 `param-only supervisor + custom Torch PPO`

### 4.5 工程红线

- 所有实验必须保持 deterministic seeds，或在非 deterministic runtime 下明确记录可复现的 checkpoint 与 seed 路径
- 每次源码改动后至少执行：
  - `python -m compileall src tests scripts`
- 对论文主表链路的任何改动，必须同时补：
  - 统计层测试
  - 导出层测试
  - reproduce / manifest / acceptance 层测试

---

## 5. 一句话结论

- RL 现在已经“能进 gate”，但 multi-seed 结果表明它仍不优于 white-box baseline。
- canonical sealing、progress ledger、`status-only / validate-only` 这一轮已经写完；下一位工程师不要再去改 RL，而是立刻补 **manifest 驱动的 pure-disk audit bootstrap**，然后正式跑 `paper_canonical`，把白盒正文主线收口。
