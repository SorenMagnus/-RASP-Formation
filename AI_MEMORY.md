# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件已按 `2026-04-01` 的真实仓库状态重写。旧版 AI_MEMORY 中“下一步去做 smoke train / 继续盲等长训”的指令已经过期。

---

## 0. 当前开发游标

- 日期：
  - `2026-04-01`
- Git 游标：
  - `HEAD = 368d7efb22b857d41d8f591fbe595a5187ab77b1`
- 当前工作树：
  - 当前 repo-tracked 可见改动只有本文件 `AI_MEMORY.md`
- 当前论文主线：
  - 正文主方法仍然是白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 仍然只是可选增强 / 附录候选，不是正文主方法
- 当前最重要的最新状态：
  - gatefix 已完成并通过 S5 预设验收
  - reward shaping / reward config externalization / PPO reward diagnostics 也已经完成
  - 上一轮完整源码验证仍然有效：
    - `python -m compileall src tests scripts` 通过
    - `python -m pytest -q` 通过
    - 结果为 `120 passed`
  - 当前没有活动中的 `train_rl_supervisor.py` 训练进程
  - 原始 `reward_v2` checkpoint 仍停在：
    - `outputs/rl_train_s5_param_only_reward_v2/checkpoints/latest.pt`
    - `timesteps_done = 1024`
    - `rollout_seed_next = 2`
  - 独立 smoke 续训目录已经建立并实际推进到：
    - `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`
    - `timesteps_done = 1536`
    - `rollout_seed_next = 3`
  - smoke 续训只多完成了 1 个额外 rollout：
    - `512` steps 用时 `412.8 s`
    - 吞吐异常慢，不适合继续盲等 `20k / 50k / 200k`
- 当前最新 RL 早期评估产物：
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__no_rl/summary.csv`
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/summary.csv`
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/analysis/rl_attribution/aggregate.json`
- 当前最新 early-eval 的关键结论：
  - `dominant_bottleneck = supervisor_gating`
  - `rl_fallback_ratio_mean = 1.0`
  - `rl_active_ratio_mean = 0.0`
  - `theta_change_ratio_mean = 0.0`
  - `gate_open_ratio_mean = 0.0`
  - `leader_final_x_delta_mean = 0.0`
  - `safety_intervention_ratio_mean = 0.5272727272727273`
  - `qp_engagement_ratio_mean = 0.5287878787878788`
  - `confidence_raw_mean_mean ≈ 0.3165 < tau_enter = 0.55`

---

## 1. 已完成工作

### 1.1 代码实现层：gatefix 与 reward_v2 主链都已落地

- `src/apflf/rl/policy.py`
  - 已把 `TorchBetaPolicy.infer()` 的 confidence 从 entropy-based 改成 Beta 方差校准版本
  - 当前公式：
    - `var_i = alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1))`
    - `v_uniform = 1 / 12`
    - `confidence_raw = clip(1 - mean_i(var_i) / v_uniform, 0, 1)`
- `src/apflf/decision/rl_mode.py`
  - 已实现两阈值滞回门控
  - 当前默认值：
    - `tau_enter = 0.55`
    - `tau_exit = 0.45`
  - 仍保持 exact fallback：
    - `mode = fallback_decision.mode`
    - `theta = fallback_decision.theta`
    - `source = "rl_fallback"`
- `src/apflf/utils/types.py`
  - 已增加：
    - `RLDecisionConfig.tau_enter`
    - `RLDecisionConfig.tau_exit`
    - `DecisionDiagnostics.confidence_raw`
    - `DecisionDiagnostics.gate_open`
    - `DecisionDiagnostics.gate_reason`
    - `RLRewardConfig`
- `src/apflf/utils/config.py`
  - 已支持 `tau_enter / tau_exit`
  - 已支持 `decision.rl.reward.*`
  - 已做 reward 非负与 gate 合法性校验
- `src/apflf/rl/env.py`
  - `_reward_terms()` 已改成显式 safety-aware reward
  - `q_t / f_t` 已按车辆数 `N` 归一化
  - `env.step(...).info` 已输出 reward decomposition 与 safety occupancy
- `src/apflf/rl/ppo.py`
  - 已汇总 rollout 内：
    - `reward_progress_mean`
    - `reward_form_mean`
    - `reward_intervene_mean`
    - `reward_qp_mean`
    - `reward_fallback_mean`
    - `reward_slack_mean`
    - `reward_theta_rate_mean`
    - `reward_goal_mean`
    - `reward_collision_mean`
    - `reward_boundary_mean`
    - `reward_total_mean`
    - `qp_engagement_ratio_mean`
    - `fallback_ratio_mean`
    - `theta_delta_linf_mean`
- `src/apflf/sim/runner.py` / `src/apflf/sim/replay.py` / `src/apflf/analysis/rl_attribution.py`
  - replay / attribution / gate 级诊断链路已打通

### 1.2 测试层：上一轮完整代码验证已闭环

- 已覆盖：
  - `tests/test_config.py`
  - `tests/test_rl_supervisor.py`
  - `tests/test_rl_attribution.py`
- 上一次完整源码验证结果仍然有效：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q` 通过
  - 当前基线为 `120 passed`
- 今天没有新的源码改动，所以今天的新增进展来自运行产物，而不是新的实现代码

### 1.3 今天新增的真实进展：训练与评估闭环已经推进，但结论不乐观

- 今天没有新的 repo-tracked 源码提交
- 今天完成的是运行验证，而不是代码实现：
  - 从原始 `reward_v2` checkpoint 出发，单独拉起了 smoke 续训
  - 独立 smoke 训练目录：
    - `outputs/rl_train_s5_param_only_reward_v2_smoke20k/`
  - 该续训从 `1024 -> 1536` steps，只多完成了 1 个 rollout
  - `main_stdout.log` 明确显示：
    - `batch_steps = 512`
    - `elapsed_s = 412.8`
    - `reward_total_mean = -0.08782262409562257`
    - `qp_engagement_ratio_mean = 0.263021`
    - `fallback_ratio_mean = 0.205078`
  - 这条续训随后被人工停止，因为吞吐异常慢
- 今天已经拿 `1536-step` checkpoint 跑完 deterministic `S5 seed 0/1/2` benchmark：
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__no_rl/summary.csv`
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/summary.csv`
  - 结果是 `rl_param_only` 与 `no_rl` 完全一致
  - 关键统计：
    - `leader_final_x_mean = 26.68925199612673`
    - `collision_total = 0`
    - `boundary_total = 0`
    - `fallback_events_total = 338`
    - `safety_interventions_total = 413`
- 今天已经跑完 early attribution：
  - `outputs/s5_rl_reward_v2_ckpt1536_eval__rl_param_only/analysis/rl_attribution/aggregate.json`
  - 必须记住的结论：
    - `dominant_bottleneck = supervisor_gating`
    - `rl_fallback_ratio_mean = 1.0`
    - `rl_active_ratio_mean = 0.0`
    - `theta_change_ratio_mean = 0.0`
    - `gate_open_ratio_mean = 0.0`
    - `leader_final_x_delta_mean = 0.0`
    - `safety_intervention_ratio_mean = 0.5272727272727273`
    - `qp_engagement_ratio_mean = 0.5287878787878788`
    - `confidence_raw_mean_mean ≈ 0.3165`
    - `tau_enter = 0.55`
    - 也就是说 early checkpoint 几乎完全进不了 gate

---

## 2. 当前研究判断

- 当前仓库的主问题已经不是：
  - reward 没写完
  - reward 无法配置
  - PPO 日志缺 reward 分解
  - gatefix 没做
- 当前真正的新阻塞是：
  - `reward_v2` 的代码路径已经打通，但 `1536-step` early checkpoint 还没有真正改变 nominal 行为
  - 当前更深层的组合瓶颈是：
    - `early-training`
    - `fixed gate threshold`
    - `异常缓慢的训练吞吐`
- 具体来说：
  - reward_v2 训练环境本身不经过 `RLSupervisor` gate
  - gate 发生在 checkpoint-backed runtime / benchmark 包装层
  - 当前 early checkpoint 的 `confidence_raw_mean ≈ 0.3165`，远低于评估阈值 `tau_enter = 0.55`
  - 所以 benchmark 阶段表现为：
    - `rl_fallback_ratio_mean = 1.0`
    - `theta_change_ratio_mean = 0.0`
    - `leader_final_x_delta_mean = 0.0`
- 研究结论必须更新为：
  - 继续原样盲拉 `20k / 50k / 200k` 没有性价比
  - 当前不应再把问题误判成 OOD
  - 当前也不应再重写 reward
  - 下一步必须优先处理“早期 checkpoint 完全进不了 gate”这个结构性问题

---

## 3. 下一步指令

### 3.1 第一优先级：不要再重写 reward，也不要再盲跑 smoke train

下一个工程师启动 AI 后，不要再重复写 reward shaping。  
也不要继续原样盲等 `20k / 50k / 200k` 训练。

下一条立即执行的代码任务固定为：

- 在以下文件中实现“训练专用的 gate warm-start / threshold annealing”：
  - `src/apflf/utils/types.py`
  - `src/apflf/utils/config.py`
  - `src/apflf/decision/rl_mode.py`
- 如需把 checkpoint 中的 `timesteps_done` 传递到 supervisor，可最小增量补触：
  - `src/apflf/rl/policy.py`
  - `src/apflf/decision/mode_base.py`

注意：

- 当前 `SupervisorTrainingEnv` 本身不经过 `RLSupervisor` gate
- 所以这个任务的真正落点不是 reward env，而是 checkpoint-backed runtime wrapper
- warm-start 必须由 checkpoint 的 `timesteps_done` 驱动
- deterministic benchmark / replay / attribution 必须保持最终评估阈值不变

### 3.2 数学约束：必须原样执行，不留实现自由度

令：

- `k` 为 checkpoint 已完成训练步数，即 `timesteps_done`
- `K` 为 warmup horizon

训练态阈值定义为：

- `tau_enter_train(k) = tau_enter_final - (tau_enter_final - tau_enter_start) * max(0, 1 - k / K)`
- `tau_exit_train(k) = tau_exit_final - (tau_exit_final - tau_exit_start) * max(0, 1 - k / K)`

默认值固定为：

- `tau_enter_start = 0.25`
- `tau_exit_start = 0.15`
- `tau_enter_final = 0.55`
- `tau_exit_final = 0.45`
- `K = 20000`

必须满足对所有 `k >= 0`：

- `0 <= tau_exit_train(k) < tau_enter_train(k) <= 1`
- `tau_enter_train(k)` 与 `tau_exit_train(k)` 对 `k` 单调不减
- `k >= K` 时严格恢复到最终评估阈值 `0.55 / 0.45`

同时必须保持：

- `OOD` gate 完全不变：
  - `||z_t||_inf > ood_threshold` 时仍然强制 fallback
- `confidence_raw` 的 Beta 方差定义完全不变
- deterministic benchmark / replay / attribution 必须继续使用固定最终阈值：
  - `tau_enter = 0.55`
  - `tau_exit = 0.45`
- 不允许让训练态 warm-start 渗透进 deterministic eval

### 3.3 建议的实现落点

- `src/apflf/utils/types.py`
  - 为 `RLDecisionConfig` 新增训练态 gate warm-start 配置字段
  - 推荐固定新增：
    - `tau_enter_start`
    - `tau_exit_start`
    - `gate_warmup_timesteps`
- `src/apflf/utils/config.py`
  - 从 YAML 读取上述字段
  - 校验：
    - `0 <= tau_exit_start < tau_enter_start <= 1`
    - `0 <= tau_exit < tau_enter <= 1`
    - `gate_warmup_timesteps > 0`
- `src/apflf/rl/policy.py`
  - `PolicyBundle` 需要最小增量带出 checkpoint 的 `timesteps_done`
- `src/apflf/decision/mode_base.py`
  - 将 checkpoint progress 传给 `RLSupervisor`
- `src/apflf/decision/rl_mode.py`
  - 增加一个训练态阈值解析函数
  - 仅在“非 deterministic-eval 且存在 checkpoint progress”时启用 annealing
  - deterministic eval 下必须直接使用最终阈值 `0.55 / 0.45`

### 3.4 下一轮验收顺序

#### A. 先补单元测试

必须至少补：

- 配置解析测试：
  - training warm-start 参数可加载
  - 非法阈值关系会报错
- supervisor 测试：
  - `tau_enter_train(k)` / `tau_exit_train(k)` 单调
  - 边界正确
  - `k >= K` 时恢复最终值
- 评估不变性测试：
  - deterministic eval 下仍严格使用 `0.55 / 0.45`
  - `OOD` gate 逻辑不变

#### B. 再用同一 reward_v2 配方重启 smoke train

优先从最新可用 checkpoint 继续，而不是回退到 `1024-step`：

- `outputs/rl_train_s5_param_only_reward_v2_smoke20k/checkpoints/latest.pt`

不要改 reward 配方，不要改 safety 层。

#### C. 然后重新跑 deterministic S5 benchmark 与 attribution

必须再次执行：

- `benchmark_s5_rl.py`
- `analyze_s5_rl_attribution.py`

且 deterministic eval 仍必须用最终阈值。

#### D. 第一阶段验收目标

第一阶段目标不是直接超过 gatefix，而是先打破当前的“全程 fallback”：

- `collision_count` 总和仍为 `0`
- `boundary_violation_count` 总和仍为 `0`
- `rl_fallback_ratio_mean < 1.0`
- `gate_open_ratio_mean > 0.0`
- `theta_change_ratio_mean > 0.0`

只有先通过这一组门槛，才允许再谈：

- `50k / 200k` 训练
- 与 gatefix baseline 的进一步正面对比
- 是否值得推进 RL 附录增强分支

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

- 所有实验必须保持 deterministic seeds
- 每次源码改动后至少执行：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`
- 不要回滚当前工作树里其他人的未提交改动

---

## 5. 不要做的事

- 不要再重复写 gatefix
- 不要再重复写 reward shaping
- 不要把当前问题重新误判成 OOD
- 不要继续原样盲跑 `20k / 50k / 200k`
- 不要为了追求 RL 提升去动 safety red-line 文件
- 不要修改 `ModeDecision` 结构
- 不要修改 `compute_actions(observation, mode, theta=None)` 公共签名
- 不要让 RL 直接输出连续控制
- 不要把 training warm-start 渗透进 deterministic benchmark / replay / attribution
- 不要把当前 `1536-step` checkpoint 误判成“已经有效”

---

## 6. 一句话交接

gatefix 和 reward shaping 都已经完成，reward 配置与 PPO reward diagnostics 也已经打通；今天新增的真实进展证明：`reward_v2` 的 early checkpoint 目前仍然因为 `confidence_raw` 过低而几乎全程进不了 gate，同时训练吞吐又慢到不值得盲等。下一个工程师的第一件事，不是继续改 reward，也不是继续死跑长训，而是先把“基于 checkpoint timesteps 的训练态 gate warm-start / threshold annealing”做对，再重新跑 smoke benchmark 与 attribution。
