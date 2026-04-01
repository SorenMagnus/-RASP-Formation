# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件已按 `2026-04-01` 的真实仓库状态重写。旧版 AI_MEMORY 中“去补 effective runtime threshold 持久化”的指令已经完成，不再是当前开发游标。

---

## 0. 当前开发游标

- 日期：
  - `2026-04-01`
- Git 游标：
  - `HEAD = d94378ca38f446fee15901741bee421ba60d0ca1`
- 当前工作树：
  - 当前 repo-tracked 改动有 7 个文件，全部属于 “effective runtime threshold 持久化 + attribution 可见性” 这一轮：
    - `src/apflf/analysis/rl_attribution.py`
    - `src/apflf/decision/rl_mode.py`
    - `src/apflf/sim/replay.py`
    - `src/apflf/sim/runner.py`
    - `src/apflf/utils/types.py`
    - `tests/test_rl_attribution.py`
    - `tests/test_rl_supervisor.py`
  - 本文件 `AI_MEMORY.md` 在本轮重写后也会成为新增改动的一部分
- 当前后台进程：
  - 当前没有活动中的 `train_rl_supervisor.py`
  - 当前也没有活动中的 `benchmark_s5_rl.py`
  - 当前唯一存活的 `python` 进程是 VS Code 的 `pylint` language server，不属于实验运行
- 当前论文主线：
  - 正文主方法仍然是白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 仍然只是“可选增强 / 附录候选”，不是正文主方法
  - `outputs/paper_canonical` 当前仍不存在，说明论文主表级 canonical artifact 还没有真正落盘
- 当前 checkpoint 游标：
  - 原始 `reward_v2` checkpoint：
    - `outputs/rl_train_s5_param_only_reward_v2/checkpoints/latest.pt`
    - `timesteps_done = 1024`
    - `rollout_seed_next = 2`
  - smoke checkpoint：
    - `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`
    - `timesteps_done = 1536`
    - `rollout_seed_next = 3`
    - 这是当前 warm-start 所有运行的统一起点
- 当前最新评估产物：
  - deterministic early-eval（旧结论，固定最终阈值）：
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__no_rl/summary.csv`
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/summary.csv`
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/analysis/rl_attribution/aggregate.json`
  - warm-start 单 seed 最小运行验证：
    - `outputs/s5_rl_gate_warmstart_seed0__no_rl/summary.csv`
    - `outputs/s5_rl_gate_warmstart_seed0__rl_param_only/summary.csv`
    - `outputs/s5_rl_gate_warmstart_seed0__rl_param_only/analysis/rl_attribution/aggregate.json`
  - warm-start 正式 multi-seed smoke（当前官方最新 RL 结果）：
    - `outputs/s5_rl_gate_warmstart_smoke__no_rl/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/summary.csv`
    - `outputs/s5_rl_gate_warmstart_smoke__rl_param_only/analysis/rl_attribution/aggregate.json`

### 当前最重要的结论

- 旧 deterministic early-eval 仍然说明“如果固定使用最终阈值，1536-step checkpoint 几乎进不了 gate”：
  - `dominant_bottleneck = supervisor_gating`
  - `rl_fallback_ratio_mean = 1.0`
  - `gate_open_ratio_mean = 0.0`
  - `theta_change_ratio_mean = 0.0`
  - `leader_final_x_delta_mean = 0.0`
- 但 current branch 的 warm-start + effective-threshold 持久化已经把这个结构性死锁彻底打破，并且这是 multi-seed 结果，不再只是 `seed 0`：
  - `dominant_bottleneck = safety_engagement`
  - `rl_fallback_ratio_mean = 0.043939393939393945`
  - `gate_open_ratio_mean = 0.956060606060606`
  - `theta_change_ratio_mean = 0.956060606060606`
  - `effective_tau_enter_mean_mean = 0.27304`
  - `effective_tau_exit_mean_mean = 0.17304000000000005`
  - `safety_intervention_ratio_mean = 0.3090909090909091`
  - `qp_engagement_ratio_mean = 0.3090909090909091`
  - `collision_count` 总和 = `0`
  - `boundary_violation_count` 总和 = `0`
- 但 RL 仍然不能升级为正文贡献，因为 multi-seed 效率已经明确劣于 white-box reference：
  - `no_rl leader_final_x_mean = 26.68925199612673`
  - `rl_param_only leader_final_x_mean = 26.627884374632824`
  - `leader_final_x_delta_mean = -0.061367621493906434`
- 因此当前研究结论已经足够明确：
  - RL 不再被 `supervisor_gating` 卡死
  - RL 也已经真正改变 nominal layer
  - 但它仍没有在 multi-seed 上优于白盒主线
  - RL 当前应继续停留在附录增强分支，不能继续阻塞 white-box 正文闭环

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

### 1.2 本轮源码实现：effective runtime threshold 已贯通到离线链路

- `src/apflf/utils/types.py`
  - `DecisionDiagnostics` 新增：
    - `effective_tau_enter`
    - `effective_tau_exit`
- `src/apflf/decision/rl_mode.py`
  - `RLSupervisor.select()` 现在会在每一步显式计算并持久化：
    - `effective_tau_enter`
    - `effective_tau_exit`
  - gate 判定实际使用的就是这两个值
  - `gate_reason` 与有效阈值保持一致
- `src/apflf/sim/runner.py`
  - 已将：
    - `decision_effective_tau_enters`
    - `decision_effective_tau_exits`
    持久化进 `.npz` artifact
- `src/apflf/sim/replay.py`
  - 已支持从 artifact 中读取：
    - `decision_effective_tau_enters`
    - `decision_effective_tau_exits`
  - 对旧 artifact 保持 backward-compatible fallback
- `src/apflf/analysis/rl_attribution.py`
  - attribution 现在不再只依赖配置中的最终阈值
  - 它会优先读取 replay 中真实持久化的：
    - `effective_tau_enter`
    - `effective_tau_exit`
  - 并导出：
    - `effective_tau_enter_mean/min/max`
    - `effective_tau_exit_mean/min/max`
  - 低置信度 fallback 的解释逻辑现在基于“真实有效阈值”，而不是只基于最终配置阈值

### 1.3 本轮测试补齐

- `tests/test_rl_supervisor.py`
  - 已补：
    - FSM fallback 路径的有效阈值默认值检查
    - deterministic eval 路径固定使用 `0.55 / 0.45`
    - 训练态 runtime 路径使用 annealed threshold
    - run artifact 中有效阈值可被 replay 回读
- `tests/test_rl_attribution.py`
  - 已补：
    - attribution 可以读出 replay 中真实生效的阈值
    - deterministic eval 会报告最终阈值
    - 训练态 runtime 会报告 annealed 阈值
    - `analyze_s5_rl_attribution.py` 输出中包含有效阈值统计

### 1.4 当前验证状态

- 本轮已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q tests/test_config.py tests/test_rl_supervisor.py tests/test_rl_attribution.py` 通过
  - 当前结果是 `22 passed`
- 需要诚实记录：
  - 本轮 fresh `python -m pytest -q` 没有完整跑到结束；它在约 15 分钟时超时
  - 因此不要宣称“本轮 fresh full-suite 全绿”
  - 当前可信的代码质量结论是：
    - 本轮直接相关的测试链路是绿的
    - 历史 warm-start 阶段曾有一次 `124 passed`，但那不是本轮 fresh rerun

### 1.5 本轮运行验证：multi-seed warm-start smoke 已正式闭环

- 本轮已完成：
  - `python scripts/benchmark_s5_rl.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --rl-checkpoint outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt --exp-id-prefix s5_rl_gate_warmstart_smoke`
  - `python scripts/analyze_s5_rl_attribution.py --rl-run-dir outputs/s5_rl_gate_warmstart_smoke__rl_param_only --reference-run-dir outputs/s5_rl_gate_warmstart_smoke__no_rl --as-json`
- 运行结论已经稳定：
  - warm-start 在 `seed 0/1/2` 上都能解锁 gate，不是单 seed 偶然
  - 主瓶颈已稳定从 `supervisor_gating` 转到 `safety_engagement`
  - RL 在安全约束下可运行，但效率端仍落后于 white-box reference

---

## 2. 当前研究判断

- 当前仓库最重要的变化不是“又多了一层 RL 技巧”，而是“我们已经有足够证据判断 RL 当前不该继续绑架正文主线”。
- 现在已经可以明确说：
  - `param-only RL` 不再被 gate 结构问题困住
  - 真实运行阈值已经可解释、可回放、可归因
  - 但 multi-seed 结果已经证明它目前仍不能稳定优于 `no_rl`
- 所以当前研究判断必须更新为：
  1. RL 分支现在保留为“附录增强候选”，不再作为近期主攻方向
  2. 不要继续在 gate / reward / warm-start 上追加局部算法修改
  3. 当前主线必须切回 white-box 论文闭环，即 `paper_canonical`
- 现阶段真正还没收口的，是：
  - `outputs/paper_canonical` 仍然缺失
  - `S1-S5 × baselines × ablations × 30 seeds` 还没有正式生成一份唯一的 canonical bundle
  - 论文主表、显著性报告、artifact sealing 还没有最终固化

---

## 3. 下一步指令

### 3.1 总原则

下一个工程师启动 AI 后：

- 不要再改 RL reward
- 不要再改 RL gate
- 不要再改 warm-start 公式
- 不要再花时间证明“RL 能不能进 gate”，这件事已经有 multi-seed 证据了
- 下一条立即执行的代码任务，必须切到 **white-box canonical paper bundle 封口**

### 3.2 立即执行的代码任务

在以下文件中实现 **canonical matrix manifest / validator / acceptance sealing**：

- `scripts/reproduce_paper.py`
- `src/apflf/analysis/stats.py`
- `tests/test_stats_export.py`
- `tests/test_reproduce_paper.py`  （若当前不存在，就新建）

如果实现中确实需要额外输出辅助文件，可以最小增量触碰：

- `src/apflf/analysis/export.py`

但不要改核心算法代码，不要碰 RL / safety 主链。

### 3.3 这段代码必须完成什么

为 `paper_canonical` 增加一套**唯一、可恢复、可验证**的 bundle 封口机制。最小产物必须包括：

- `outputs/paper_canonical/manifest.json`
- `outputs/paper_canonical/matrix_index.csv`
- `outputs/paper_canonical/paper_acceptance.json`

其中：

- `manifest.json`
  - 记录：
    - `git_commit`
    - `primary_method = no_rl`
    - `scenarios = [s1_local_minima, ..., s5_dense_multi_agent]`
    - `methods = [no_rl, apf, apf_lf, st_apf, dwa, orca]`
    - `ablations`
    - `seed_set = [0, 1, ..., 29]`
- `matrix_index.csv`
  - 每一行对应一个 `(scenario, method)` 或 `(scenario, ablation)` cell
  - 必须至少含：
    - `scenario`
    - `variant_type`
    - `variant_name`
    - `expected_seed_count`
    - `actual_seed_count`
    - `missing_seed_count`
    - `complete`
    - `config_hash_consistent`
    - `output_dir`
- `paper_acceptance.json`
  - 必须至少含：
    - `bundle_complete`
    - `primary_safety_valid`
    - `missing_cells`
    - `invalid_cells`
    - `primary_method`
    - `num_complete_cells`
    - `num_expected_cells`

### 3.4 数学与统计约束

这些约束必须原样执行，不给下一个工程师留自由发挥空间：

#### A. canonical seed set 固定

- 令 canonical seed 集合为：
  - `S = {0, 1, 2, ..., 29}`
- 对任意 cell `c = (scenario, variant)`：
  - `Seeds(c) = 该 cell 在 summary.csv 中出现的唯一 seed 集合`
  - `complete(c) = 1` 当且仅当 `Seeds(c) = S`
  - `actual_seed_count(c) = |Seeds(c)|`
  - `missing_seed_count(c) = 30 - |Seeds(c)|`

#### B. 配置一致性约束

- 对任意 cell `c`：
  - `ConfigHashes(c) = 该 cell 全部 seed 行中的 config_hash 集合`
  - `config_hash_consistent(c) = 1` 当且仅当 `|ConfigHashes(c)| = 1`
- 若 `complete(c) = 0` 或 `config_hash_consistent(c) = 0`，该 cell 必须被标记为 invalid

#### C. paired delta 约束

- primary reference method 固定为：
  - `no_rl`
- 对任意 scenario `s`、任意 method `m != no_rl`、任意 metric `x`：
  - 只能对齐同一 seed 的结果做 paired comparison
  - 若该 seed 在 reference 或 method 任一方缺失，则该 seed 不得参与 paired delta
  - 定义：
    - `Δ_i(s, m, x) = x_i(s, m) - x_i(s, no_rl)`
  - paired aggregate 只能基于这组 `Δ_i` 计算
- 不允许把不同 seed 的结果混在一起做伪 paired

#### D. CI 估计约束

- 不要重新发明统计方法
- 继续沿用 `src/apflf/analysis/stats.py` 现有 deterministic bootstrap CI 作为默认主表 CI
- 不允许把默认主表 CI 改回 t-interval

#### E. paper acceptance 约束

- `bundle_complete = 1` 当且仅当所有 expected canonical cells 都满足：
  - `complete(c) = 1`
  - `config_hash_consistent(c) = 1`
- `primary_safety_valid = 1` 当且仅当对 primary method = `no_rl` 的每个 scenario：
  - `Σ collision_count = 0`
  - `Σ boundary_violation_count = 0`
- 只要任一条件不满足，`paper_acceptance.json` 必须明确给出失败原因，不允许静默成功

### 3.5 下一轮执行顺序

#### A. 先写 canonical validator / manifest 代码并补测试

必须至少覆盖：

- canonical seed completeness 检测
- config hash consistency 检测
- paired seed alignment 检测
- `paper_acceptance.json` 失败原因输出
- `--skip-existing` 下的 resumable 行为

#### B. 再正式运行 white-box canonical matrix

运行：

- `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`

注意：

- 这一步只跑 white-box canonical matrix
- 不要把 RL method 混进 canonical bundle
- RL 分支此时不应继续阻塞正文 artifact 落盘

#### C. 再检查 canonical bundle 是否闭环

必须检查：

- `outputs/paper_canonical/manifest.json`
- `outputs/paper_canonical/matrix_index.csv`
- `outputs/paper_canonical/paper_acceptance.json`
- `outputs/paper_canonical/tables/`
- `outputs/paper_canonical/figures/`

#### D. 下一轮验收标准

只有满足以下条件，才算 white-box 论文闭环进入最终冲刺：

- `paper_acceptance.json` 中：
  - `bundle_complete = true`
  - `primary_safety_valid = true`
- `outputs/paper_canonical` 存在且不是空目录
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
  - reproduce / manifest 层测试

---

## 5. 一句话结论

- RL 现在已经“能进 gate”，但 multi-seed 结果表明它仍不优于 white-box baseline。
- 因此下一位工程师不要再继续追 RL 算法，而是立刻去写 `paper_canonical` 的 manifest / validator / acceptance sealing，把 white-box 论文主线正式收口。
