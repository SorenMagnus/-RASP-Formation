# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件已按 `2026-04-01` 的真实仓库状态重写。旧版 AI_MEMORY 中“下一步去做 warm-start 实现”的指令已经完成，不再是当前开发游标。

---

## 0. 当前开发游标

- 日期：
  - `2026-04-01`
- Git 游标：
  - `HEAD = 247073010054cffa34873e19713973f63fe59a8a`
- 当前工作树：
  - 当前 repo-tracked 改动集中在 8 个文件，全部是 runtime gate warm-start 相关：
    - `configs/default.yaml`
    - `src/apflf/decision/mode_base.py`
    - `src/apflf/decision/rl_mode.py`
    - `src/apflf/rl/policy.py`
    - `src/apflf/utils/config.py`
    - `src/apflf/utils/types.py`
    - `tests/test_config.py`
    - `tests/test_rl_supervisor.py`
  - 本文件 `AI_MEMORY.md` 现在也会成为当前周期新增改动的一部分
- 当前论文主线：
  - 正文主方法仍然是白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 仍然只是可选增强 / 附录候选，不是正文主方法
- 当前训练状态：
  - 当前没有活动中的 `train_rl_supervisor.py` 训练进程
  - 原始 `reward_v2` checkpoint 仍停在：
    - `outputs/rl_train_s5_param_only_reward_v2/checkpoints/latest.pt`
    - `timesteps_done = 1024`
    - `rollout_seed_next = 2`
  - 独立 smoke 续训 checkpoint 仍是：
    - `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`
    - `timesteps_done = 1536`
    - `rollout_seed_next = 3`
    - `num_logs = 3`
  - 这条 smoke 续训只多完成了 1 个 rollout：
    - `512` steps 用时 `412.8 s`
    - 吞吐仍然偏慢，不适合继续盲等 `20k / 50k / 200k`
- 当前最新 RL 评估产物：
  - 旧的 deterministic early-eval（3 seeds）：
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__no_rl/summary.csv`
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/summary.csv`
    - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/analysis/rl_attribution/aggregate.json`
  - 新的 warm-start 最小运行验证（1 seed）：
    - `outputs/s5_rl_gate_warmstart_seed0__no_rl/summary.csv`
    - `outputs/s5_rl_gate_warmstart_seed0__rl_param_only/summary.csv`
    - `outputs/s5_rl_gate_warmstart_seed0__rl_param_only/analysis/rl_attribution/aggregate.json`
- 当前最重要的结论：
  - warm-start 代码已经实现并通过测试
  - deterministic early-eval 仍然说明旧问题真实存在：
    - `dominant_bottleneck = supervisor_gating`
    - `rl_fallback_ratio_mean = 1.0`
    - `gate_open_ratio_mean = 0.0`
    - `theta_change_ratio_mean = 0.0`
    - `leader_final_x_delta_mean = 0.0`
  - 但 warm-start 后的最小运行验证已经证明结构性死锁被打破：
    - `dominant_bottleneck = safety_engagement`
    - `rl_fallback_ratio_mean = 0.045454545454545456`
    - `gate_open_ratio_mean = 0.9545454545454546`
    - `theta_change_ratio_mean = 0.9545454545454546`
    - `safety_intervention_ratio_mean = 0.31363636363636366`
    - `qp_engagement_ratio_mean = 0.31363636363636366`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
  - 当前仍未解决的问题是：
    - 这只是 `seed 0` 的最小运行验证
    - `leader_final_x_delta_mean = -0.021135497013787585`，说明效率端仍未优于 white-box reference
    - multi-seed warm-start 正式闭环还没有跑完

---

## 1. 已完成工作

### 1.1 旧主线能力：gatefix + reward_v2 基础设施仍然成立

- `src/apflf/rl/policy.py`
  - `TorchBetaPolicy.infer()` 仍使用 Beta 方差校准 confidence
  - 当前公式：
    - `var_i = alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1))`
    - `v_uniform = 1 / 12`
    - `confidence_raw = clip(1 - mean_i(var_i) / v_uniform, 0, 1)`
- `src/apflf/decision/rl_mode.py`
  - 旧 gatefix 仍保留两阈值滞回门控
  - 评估终态阈值仍为：
    - `tau_enter = 0.55`
    - `tau_exit = 0.45`
- `src/apflf/utils/types.py` / `src/apflf/utils/config.py`
  - reward 配置化与 gate 基础配置已在位
- `src/apflf/rl/env.py` / `src/apflf/rl/ppo.py`
  - reward decomposition 与 PPO rollout diagnostics 仍然工作
- `src/apflf/sim/runner.py` / `src/apflf/sim/replay.py` / `src/apflf/analysis/rl_attribution.py`
  - replay / attribution / gate 级诊断链路仍然可用

### 1.2 本轮源码实现：runtime gate warm-start / threshold annealing 已落地

- `configs/default.yaml`
  - 已新增：
    - `decision.rl.tau_enter_start = 0.25`
    - `decision.rl.tau_exit_start = 0.15`
    - `decision.rl.gate_warmup_timesteps = 20000`
- `src/apflf/utils/types.py`
  - `RLDecisionConfig` 已新增：
    - `tau_enter_start`
    - `tau_exit_start`
    - `gate_warmup_timesteps`
- `src/apflf/utils/config.py`
  - 已支持从 YAML 加载上述参数
  - 已增加合法性校验：
    - `0 <= tau_exit_start < tau_enter_start <= 1`
    - `0 <= tau_exit < tau_enter <= 1`
    - `tau_enter_start <= tau_enter`
    - `tau_exit_start <= tau_exit`
    - `gate_warmup_timesteps > 0`
- `src/apflf/rl/policy.py`
  - `PolicyBundle` 已最小增量带出：
    - `checkpoint_timesteps_done`
- `src/apflf/decision/mode_base.py`
  - 已将 checkpoint progress 透传给 `RLSupervisor`
- `src/apflf/decision/rl_mode.py`
  - 已新增训练态阈值退火逻辑
  - 当前规则是：
    - 只有 `deterministic_eval = False` 且 checkpoint 含 `timesteps_done` 时，才启用训练态 annealing
    - `deterministic_eval = True` 时仍严格使用最终评估阈值 `0.55 / 0.45`
  - 当前实现公式：
    - `tau_enter_train(k) = tau_enter_final - (tau_enter_final - tau_enter_start) * max(0, 1 - k / K)`
    - `tau_exit_train(k) = tau_exit_final - (tau_exit_final - tau_exit_start) * max(0, 1 - k / K)`
  - 并保持：
    - `OOD` gate 不变
    - `confidence_raw` 定义不变
    - `ModeDecision` 公共接口不变

### 1.3 本轮测试补齐：warm-start 相关单元测试已经落地

- `tests/test_config.py`
  - 已新增：
    - gate warm-start 参数可加载测试
    - 非法阈值关系拒绝测试
- `tests/test_rl_supervisor.py`
  - 已新增：
    - `tau_enter_train(k)` / `tau_exit_train(k)` 单调性测试
    - `k >= K` 恢复最终阈值测试
    - deterministic eval 下阈值不吃 warm-start 测试
    - 训练态 runtime 会使用 annealed threshold 的行为测试

### 1.4 当前验证状态：源码与运行验证都更新了

- 本轮已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q tests/test_config.py tests/test_rl_supervisor.py` 通过
  - `python -m pytest -q tests/test_rl_attribution.py` 通过
  - `python -m pytest -q` 通过
  - 当前结果是 `124 passed`
- 本轮额外完成了一个最小运行验证：
  - 用 `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`
  - 跑了非 deterministic `S5 seed 0`
  - 结果路径：
    - `outputs/s5_rl_gate_warmstart_seed0__no_rl/`
    - `outputs/s5_rl_gate_warmstart_seed0__rl_param_only/`
  - 该运行验证表明：
    - RL 已不再是“全程 fallback”
    - 结构性阻塞从 `supervisor_gating` 转到 `safety_engagement`
    - 但效率指标仍未证明优于 white-box baseline

---

## 2. 当前研究判断

- 当前仓库的主问题已经不是：
  - gate warm-start 还没写
  - reward 还没配
  - PPO 日志不够用
- 当前真正的新状态是：
  - warm-start 已经把“early checkpoint 完全进不了 gate”的结构性死锁解开了
  - 这一点已经在真实运行中得到验证
- 但当前仍然不能说 RL 分支已经重新可用，原因有三：
  1. warm-start 目前只验证了 `seed 0`
  2. `leader_final_x_delta_mean = -0.021135497013787585`，效率仍落后于 reference
  3. 当前 attribution 里的 `tau_enter / tau_exit` 仍显示最终配置值，尚未显式记录“实际运行时使用的有效阈值”
- 因此研究判断必须更新为：
  - RL 分支当前已从“完全进不了 gate”推进到“可以进 gate，但是否值得保留还未定”
  - 当前更深层的下一问题不是 reward 本身，而是：
    - multi-seed warm-start 是否稳定
    - gate 打开后，主瓶颈是否稳定转为 `safety_engagement`
    - 实际运行时阈值是否被准确记录并可被 attribution 解释
- 白盒主线 meanwhile 仍然是健康的，`paper_canonical` 尚未落盘，但这不应被 RL 分支继续阻塞

---

## 3. 下一步指令

### 3.1 第一优先级：不要再重写 warm-start 主逻辑，也不要回去重写 reward

下一个工程师启动 AI 后：

- 不要再重复写 reward shaping
- 不要再重复写 gate warm-start 核心公式
- 不要回退到“继续解释为什么 early checkpoint 进不了 gate”的旧结论

下一条立即执行的代码任务固定为：

- 在以下文件中补齐“实际运行时 gate 阈值”的诊断持久化与 attribution 可见性：
  - `src/apflf/utils/types.py`
  - `src/apflf/decision/rl_mode.py`
  - `src/apflf/sim/runner.py`
  - `src/apflf/sim/replay.py`
  - `src/apflf/analysis/rl_attribution.py`
  - `tests/test_rl_supervisor.py`
  - `tests/test_rl_attribution.py`

### 3.2 数学约束：下一段代码必须满足什么

对每个运行步，都要记录“实际用于 gate 判定的有效阈值”：

- `effective_tau_enter`
- `effective_tau_exit`

它们必须满足：

- 若 `deterministic_eval = True` 或没有 checkpoint progress：
  - `effective_tau_enter = tau_enter_final`
  - `effective_tau_exit = tau_exit_final`
- 否则，令 `k = checkpoint.timesteps_done`，`K = gate_warmup_timesteps`：
  - `effective_tau_enter = tau_enter_final - (tau_enter_final - tau_enter_start) * max(0, 1 - k / K)`
  - `effective_tau_exit = tau_exit_final - (tau_exit_final - tau_exit_start) * max(0, 1 - k / K)`

并且必须对所有运行满足：

- `0 <= effective_tau_exit < effective_tau_enter <= 1`
- `k` 固定时，一个 run 内所有 step 的 `effective_tau_enter / effective_tau_exit` 必须保持常数
- `k >= K` 时必须严格恢复到最终值：
  - `effective_tau_enter = 0.55`
  - `effective_tau_exit = 0.45`
- `OOD` gate 完全不变
- `confidence_raw` 的 Beta 方差定义完全不变
- 实际 gate 判定原因必须与记录阈值一致：
  - gate 关闭态且 `confidence_raw < effective_tau_enter` 时，只能是 `confidence_enter_threshold`
  - gate 打开态且 `confidence_raw < effective_tau_exit` 时，只能是 `confidence_exit_threshold`

### 3.3 下一轮实验顺序

#### A. 先写诊断可见性代码并补测试

必须补：

- 配置和诊断字段测试
- replay 对新字段的兼容读取测试
- attribution 能输出有效阈值统计测试

#### B. 再重新跑 multi-seed warm-start smoke benchmark

使用：

- checkpoint：
  - `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`
- 场景：
  - `configs/scenarios/s5_dense_multi_agent.yaml`
- seeds：
  - `0 1 2`
- 评估模式：
  - 非 deterministic runtime（即不加 `--deterministic-eval`）

#### C. 然后跑 attribution

必须对 warm-start multi-seed 结果执行：

- `benchmark_s5_rl.py`
- `analyze_s5_rl_attribution.py`

并核对 attribution 输出中：

- `effective_tau_enter`
- `effective_tau_exit`
- `gate_open_ratio`
- `rl_fallback_ratio`

#### D. 下一轮第一阶段验收目标

第一阶段目标不是直接超过 gatefix，而是先确认 warm-start 在 multi-seed 下不是偶然：

- `collision_count` 总和仍为 `0`
- `boundary_violation_count` 总和仍为 `0`
- `rl_fallback_ratio_mean < 1.0`
- `gate_open_ratio_mean > 0.0`
- `theta_change_ratio_mean > 0.0`

如果还想继续推进 RL 分支，第二阶段再看：

- `safety_intervention_ratio_mean` 是否维持不高于白盒参考级别
- `qp_engagement_ratio_mean` 是否维持不高于白盒参考级别
- `leader_final_x_delta_mean` 是否不再明显劣化

如果 multi-seed 仍不能稳定支撑这些目标，RL 继续停留在附录增强分支，白盒正文主线继续推进 `paper_canonical`

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
  - `PROMPT_SYSTEM.md` 中旧的 “mode-only RL” 理论模板不是当前活契约
  - 当前活契约仍然是 `param-only supervisor + custom Torch PPO`

### 4.5 工程红线

- 所有实验必须保持 deterministic seeds 或明确记录 stochastic runtime 的可复现种子路径
- 每次源码改动后至少执行：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`
- 不要回滚当前工作树里其他人的未提交改动

---

## 5. 不要做的事

- 不要再重复写 reward shaping
- 不要再重复写 warm-start 核心公式
- 不要把当前问题重新误判成 OOD
- 不要继续原样盲跑 `20k / 50k / 200k`
- 不要为了追求 RL 提升去动 safety red-line 文件
- 不要修改 `ModeDecision` 结构
- 不要修改 `compute_actions(observation, mode, theta=None)` 公共签名
- 不要让 RL 直接输出连续控制
- 不要在 attribution 里继续只显示最终配置阈值而忽略实际运行阈值
- 不要把当前 `seed 0` 的 warm-start 单次结果误判成已经足够支撑论文结论

---

## 6. 一句话交接

gatefix、reward shaping、reward 配置化和 PPO diagnostics 都已经完成；本轮新增的 runtime gate warm-start 也已经实现并通过 `124 passed` 验证，而且最小运行验证证明 RL 不再是“全程 fallback”。下一个工程师的第一件事，不是继续改 reward，也不是继续重写 warm-start，而是先把“实际运行时有效阈值”完整落盘到 diagnostics / replay / attribution，然后完成 multi-seed warm-start smoke benchmark 与 attribution，判断 RL 是否真的值得继续推进。
