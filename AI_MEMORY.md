# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件已按 `2026-03-31` 的真实仓库状态重写，旧版 AI_MEMORY 中“下一步先写 gatefix”的指令已经完成，不再是当前开发游标。

---

## 0. 当前开发游标

- 日期：
  - `2026-03-31`
- Git 游标：
  - `HEAD = 8951c5db785ab2c6217abf387bfe154ef2961b02`
- 当前工作树不是干净状态，未提交改动如下：
  - `AI_MEMORY.md`
  - `src/apflf/analysis/__init__.py`
  - `src/apflf/analysis/rl_attribution.py`
  - `src/apflf/decision/mode_base.py`
  - `src/apflf/decision/rl_mode.py`
  - `src/apflf/rl/policy.py`
  - `src/apflf/sim/replay.py`
  - `src/apflf/sim/runner.py`
  - `src/apflf/utils/config.py`
  - `src/apflf/utils/types.py`
  - `tests/test_rl_attribution.py`
  - `tests/test_rl_supervisor.py`
- 当前论文主线：
  - 正文主方法仍然是白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 仍然只是可选增强 / 附录候选，不是正文主方法
- 当前训练状态：
  - `outputs/rl_train_s5_param_only/checkpoints/latest.pt` 已存在
  - `outputs/rl_train_s5_param_only/checkpoints/main.pt` 已存在
  - S5 gatefix 后 benchmark 已跑完：
    - `outputs/s5_rl_gatefix_eval__no_rl/summary.csv`
    - `outputs/s5_rl_gatefix_eval__rl_param_only/summary.csv`
  - gatefix 后 attribution 已生成：
    - `outputs/s5_rl_gatefix_eval__rl_param_only/analysis/rl_attribution/aggregate.json`
    - `outputs/s5_rl_gatefix_eval__rl_param_only/analysis/rl_attribution/seed_attribution.csv`
    - `outputs/s5_rl_gatefix_eval__rl_param_only/analysis/rl_attribution/attribution_overview.pdf`
- 当前最重要的实验结论：
  - gatefix 已经通过预设验收：
    - `collision_count` 总和保持 `0`
    - `boundary_violation_count` 总和保持 `0`
    - `rl_fallback_ratio_mean = 0.22272727272727275`
    - `rl_active_ratio_mean = 0.7772727272727273`
    - `leader_final_x_delta_mean = -0.014939257879620508 m`
  - 也就是说，RL 已经不再是“绝大多数时间被 gate 吞掉”
  - 但 RL 依然没有平均优于 `no_rl`
  - 当前主瓶颈已经从 `supervisor_gating` 转移为 `safety_engagement`
    - `dominant_bottleneck = safety_engagement`
    - `safety_intervention_ratio_mean = 0.3242424242424242`
    - `qp_engagement_ratio_mean = 0.3242424242424242`

---

## 1. 已完成工作

### 1.1 上一轮计划中的 gatefix 已全部落地

- `src/apflf/rl/policy.py`
  - 已把 `TorchBetaPolicy.infer()` 的 gate 置信度从 entropy-based 改成 Beta 方差校准版本
  - 当前公式已经是：
    - `var_i = alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1))`
    - `v_uniform = 1 / 12`
    - `confidence_raw = clip(1 - mean_i(var_i) / v_uniform, 0, 1)`
- `src/apflf/decision/rl_mode.py`
  - 已实现两阈值滞回门控
  - 当前逻辑已经是：
    - `||z_t||_inf > ood_threshold` 时强制 fallback
    - 未开门时使用 `tau_enter`
    - 已开门时使用 `tau_exit`
  - 当前默认阈值：
    - `tau_enter = 0.55`
    - `tau_exit = 0.45`
  - gate 拒绝时保持 exact fallback：
    - `mode = fallback_decision.mode`
    - `theta = fallback_decision.theta`
    - `source = "rl_fallback"`
- `src/apflf/utils/types.py`
  - 已给 `RLDecisionConfig` 增加：
    - `tau_enter`
    - `tau_exit`
  - 已给 `DecisionDiagnostics` 增加：
    - `confidence_raw`
    - `gate_open`
    - `gate_reason`
- `src/apflf/utils/config.py`
  - 已支持从配置读取 `tau_enter / tau_exit`
  - 已加入参数合法性校验：
    - `0 < tau_enter <= 1`
    - `0 <= tau_exit < tau_enter`
- `src/apflf/decision/mode_base.py`
  - 已把 `tau_enter / tau_exit` 正确传入 `RLSupervisor`

### 1.2 replay / attribution / artifact 诊断链路已经贯通

- `src/apflf/sim/runner.py`
  - `.npz` 回放产物已经额外落盘：
    - `decision_confidence_raw`
    - `decision_gate_opens`
    - `decision_gate_reasons`
- `src/apflf/sim/replay.py`
  - 已支持读取这些新字段
  - 对旧 replay 保持 backward compatible 默认值
- `src/apflf/analysis/rl_attribution.py`
  - 已把 gate 级别诊断纳入归因输出：
    - `accepted_enter_steps`
    - `accepted_hold_steps`
    - `fallback_enter_threshold_steps`
    - `fallback_exit_threshold_steps`
    - `ood_gate_steps`
    - `gate_open_steps`
    - `gate_open_ratio`
    - `confidence_raw_mean`
    - `confidence_raw_min`
    - `tau_enter`
    - `tau_exit`
- `src/apflf/analysis/__init__.py`
  - 已修掉 `export_paper_artifacts` 的导入环问题，改成 lazy import 包装，避免 `replay -> analysis -> export -> replay` 循环依赖

### 1.3 测试已经补齐并通过

- `tests/test_rl_supervisor.py`
  - 已新增 Beta 方差置信度测试
  - 已覆盖滞回 gate 的开启 / 保持 / 关闭
  - 已覆盖 OOD 强制 fallback
  - 已覆盖 replay 后新诊断字段持久化
- `tests/test_rl_attribution.py`
  - 已覆盖 attribution 对 `confidence_raw / gate_open / gate_reason` 的可见性
- 本周期已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q` 通过
  - 当前结果是 `116 passed`

### 1.4 真实 S5 gatefix benchmark 已完成，结论已经稳定

- 新的 RL 输出相对 gatefix 前已经明显改善：
  - `fallback_events_mean = 61.0`
  - `fallback_events_delta_mean = -51.666666666666664`
  - `theta_change_ratio_mean = 0.7772727272727273`
  - `gate_open_ratio_mean = 0.7772727272727273`
- 这说明：
  - RL 已经在多数时间真正接管参数输出
  - nominal 层已被真实改变，不再是“几乎全程白盒 exact fallback”
- 但新的主问题也已经明确：
  - `dominant_bottleneck = safety_engagement`
  - `safety_interventions_delta_mean = -49.0`
  - 虽然 safety engagement 比之前下降很多，但仍是决定性能上限的主要约束
  - `leader_final_x_delta_mean = -0.014939257879620508 m`
  - RL 仍略低于 `no_rl`，所以还不能升级为正文主贡献

### 1.5 工程过程中的重要事实

- 本轮真实 benchmark 中，`scripts/benchmark_s5_rl.py` 因为会同时重跑 `no_rl` 和 `rl_param_only`，在会话内超时
- 随后已用与计划完全一致的配置、checkpoint、seed 集合直接完成 RL 分支评估
- 当前 `outputs/s5_rl_gatefix_eval__*` 目录是有效结果，可以继续作为下游归因与比较的输入

---

## 2. 当前研究判断

- 现在仓库的主问题已经不是：
  - checkpoint 不存在
  - OOD 过多
  - gate 完全把 RL 吞掉
- 当前真正的研究瓶颈是：
  - RL 虽然能更稳定地输出 `theta`
  - 但这些 `theta` 还没有把 `safety_engagement` 压到足以优于 `no_rl`
- 因此当前最合理的下一步，不是继续修 gate，也不是先跑 canonical matrix，更不是盲目长训
- 当前最合理的下一步是：
  - 立刻改训练 reward，让 RL 在保持安全红线不变的前提下，显式学习“减少 QP 介入 / 减少 safety intervention / 减少 fallback 占空比”，而不是只追求短期 progress

---

## 3. 下一步指令

### 3.1 第一优先级

下一个工程师启动 AI 后，应该立刻去写 `safety-aware reward shaping + reward config externalization + reward diagnostics`，而不是再去改 gate。

### 3.2 必须修改的文件

- `src/apflf/rl/env.py`
- `src/apflf/utils/types.py`
- `src/apflf/utils/config.py`
- `scripts/train_rl_supervisor.py`
- `tests/` 下新增或补齐 RL env / reward 相关测试

如果需要把权重写入 YAML，再同步修改当前训练使用的配置文件，但不要动 safety red-line 文件。

### 3.3 立刻要写的代码与数学约束

#### A. 把当前硬编码 reward 改成“可配置 + safety-aware + 归一化”的 reward

当前 `src/apflf/rl/env.py` 的 `_reward_terms()` 已经有：

- `progress`
- `formation_recovery`
- `qp_correction_norm`
- `slack`
- `fallback`
- `theta_change_penalty`

但这些项仍然不够直接地压制 `safety_engagement` 占空比，且 fallback 惩罚是按原始计数累加，不利于跨场景比较。

请改成如下定义。记：

- `N = len(current_observation.states)`
- `p_t = x_leader(t+1) - x_leader(t)`
- `e_t = formation_error(current_observation)`
- `e_{t+1} = formation_error(next_observation)`
- `c_t = mean_i correction_norm_i`
- `q_t = (1 / N) * sum_i 1[qp_solve_time_i > 0]`
- `f_t = (1 / N) * sum_i 1[fallback_flag_i = True]`
- `s_t = max_i slack_i`
- `u_t = 1[max_i correction_norm_i > eps_corr]`
- `d_t = ||theta_t - theta_{t-1}||_inf`

新的逐步 reward 必须写成：

```text
r_t =
  w_progress * p_t
  + w_form * (e_t - e_{t+1})
  - w_intervene * u_t
  - w_qp * q_t
  - w_fallback * f_t
  - w_slack * s_t
  - w_theta_rate * d_t
  + w_goal * 1[reached_goal]
  - w_collision * 1[collision]
  - w_boundary * 1[boundary_violation]
```

其中必须满足：

- 所有权重 `w_* >= 0`
- `eps_corr` 默认取 `1e-6`
- `q_t` 与 `f_t` 必须按 `N` 归一化，不能继续使用原始计数
- 在 `p_t, e_t - e_{t+1}` 固定时，`r_t` 对 `u_t, q_t, f_t, s_t, d_t` 必须单调不增

#### B. 把 reward 权重外提到配置层

不要继续把 reward 权重硬编码在 `env.py`。

请在类型层和配置加载层新增一个 reward config，例如：

- `progress_weight`
- `formation_weight`
- `intervention_weight`
- `qp_weight`
- `fallback_weight`
- `slack_weight`
- `theta_rate_weight`
- `goal_reward`
- `collision_penalty`
- `boundary_penalty`
- `correction_epsilon`

要求：

- 所有惩罚项和奖励项都能从配置读取
- 加载时做非负校验
- 默认值要能精确复现当前数量级，不要突然把训练标度改坏

#### C. 把 reward 诊断打到训练日志里

训练端至少要能在 `info` / log 中稳定拿到：

- `reward_terms`
- `safety_interventions`
- `fallback_events`
- `qp_engagement_ratio_step`
- `fallback_ratio_step`
- `theta_delta_linf`

目的：

- 下一轮训练后，不需要再靠 replay 才知道 reward 是否真的在压 safety engagement
- 能直接从训练日志判断 reward shaping 是否有效

#### D. 必须保持的硬约束

这一步只准改 reward 与训练诊断，不准破坏以下数学与接口约束：

```text
theta_lower[j] <= theta_t[j] <= theta_upper[j]
|theta_t[j] - theta_{t-1}[j]| <= rate_limit[j]
```

并且必须继续保持：

- `mode_t` 只能来自 FSM
- RL 仍然只能输出 `theta`
- 不允许 RL 直接输出 `mode`
- 不允许 RL 直接输出 `accel`
- 不允许 RL 直接输出 `steer`
- `ModeDecision(mode, theta, source, confidence)` 不变
- `compute_actions(observation, mode, theta=None)` 不变

### 3.4 为什么下一步必须是 reward shaping

因为 gate 已经修完，而且当前真实结论已经是：

- `rl_active_ratio_mean = 0.7772727272727273`
- `theta_change_ratio_mean = 0.7772727272727273`
- `dominant_bottleneck = safety_engagement`
- `safety_intervention_ratio_mean = 0.3242424242424242`
- `qp_engagement_ratio_mean = 0.3242424242424242`
- `leader_final_x_delta_mean = -0.014939257879620508 m`

这说明：

- RL 现在已经“在工作”
- 但它还没有学会减少被 QP / safety 层修正的占空比
- 所以下一步最该改的是训练目标，而不是 gate 结构

### 3.5 写完 reward 代码后的立即验收

先做增量验证：

```bash
python -m compileall src tests scripts
python -m pytest -q
```

然后重新训练 `rl_param_only`：

```bash
python scripts/train_rl_supervisor.py --config <训练配置> --seed 0 --total-timesteps 200000 --device cuda --output outputs/rl_train_s5_param_only_reward_v2/checkpoints/main.pt
```

训练完成后立刻做 deterministic benchmark：

```bash
python scripts/benchmark_s5_rl.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --rl-checkpoint outputs/rl_train_s5_param_only_reward_v2/checkpoints/main.pt --exp-id-prefix s5_rl_reward_v2_eval --deterministic-eval
```

然后做 attribution：

```bash
python scripts/analyze_s5_rl_attribution.py --rl-run-dir outputs/s5_rl_reward_v2_eval__rl_param_only --reference-run-dir outputs/s5_rl_gatefix_eval__no_rl --as-json
```

下一轮验收标准：

- `collision_count` 总和必须保持 `0`
- `boundary_violation_count` 总和必须保持 `0`
- `rl_fallback_ratio_mean` 不得明显回退，目标保持 `<= 0.25`
- `safety_intervention_ratio_mean` 必须严格小于当前的 `0.3242424242424242`
- `qp_engagement_ratio_mean` 必须严格小于当前的 `0.3242424242424242`
- `leader_final_x_delta_mean` 不能比当前的 `-0.014939257879620508` 更差
- 如果能把 `leader_final_x_delta_mean` 推到 `>= 0`，才可以开始讨论 RL 是否值得进入正文

---

## 4. 第二优先级

只有在 reward shaping 版本完成并重新评估之后，才做下面两件事：

### 4.1 白盒 canonical paper matrix

```bash
python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix
```

### 4.2 RL 是否保留为附录还是升级

只有在新训练结果同时满足以下条件时，才允许把 RL 从“附录候选”升级为“正文候选”：

- 相对 `no_rl` 有稳定正向改进
- `collision_count` 不回归
- `boundary_violation_count` 不回归
- attribution 不再显示 `safety_engagement` 主导

---

## 5. 技术栈红线

### 5.1 架构红线

- 必须保持三层白盒闭环：
  - `Mode Decision -> Nominal Controller -> Safety Filter`
- 正文主线不能改写成端到端黑盒 RL
- 当前 RL 只能是 `param-only supervisor`

### 5.2 安全层红线

严禁修改以下文件：

- `src/apflf/safety/safety_filter.py`
- `src/apflf/safety/cbf.py`
- `src/apflf/safety/qp_solver.py`

### 5.3 接口红线

严禁破坏以下公共接口与数据结构：

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

必须保持：

- `theta=None` 时精确退化到白盒基线
- FSM 与 RL supervisor 共用同一 `ModeDecision` 输出形状

### 5.4 RL 红线

- 不允许把 RL 扩成 `mode-only`
- 不允许把 RL 扩成 `full supervisor`
- 不允许 RL 直接输出 `mode`
- 不允许 RL 直接输出 `accel`
- 不允许 RL 直接输出 `steer`

### 5.5 工程红线

- 不允许引入 `SB3`
- PPO 后端必须继续沿用仓库内自定义 Torch 实现
- 所有实验必须保持 deterministic seeds
- 每次改动后至少执行：
  - `python -m compileall src tests scripts`
  - 相关增量 `pytest`
- 不要回滚当前工作树里其他人的未提交改动

---

## 6. 不要做的事

- 不要再把下一步写成“继续修 gate”
- 不要把当前问题重新误判成 OOD
- 不要为了追求 RL 提升去动 safety red-line 文件
- 不要修改 `ModeDecision` 结构
- 不要修改 `compute_actions(observation, mode, theta=None)` 公共签名
- 不要在 reward shaping 还没验证前，先跑大规模 canonical RL sweep
- 不要把当前 RL 结果包装成“已经优于 no_rl”

---

## 7. 一句话交接

gatefix 已经完成并通过实测验收，RL 现在能够大部分时间真实输出 `theta`，但当前主瓶颈已经变成 `safety_engagement` 而不是 `supervisor_gating`。下一个工程师启动后，应该立刻去写 `safety-aware reward shaping + reward config externalization + reward diagnostics`，然后基于新的 reward 重新训练 `rl_param_only`，再用 deterministic S5 benchmark 和 attribution 验证它是否真的减少了 QP / safety 介入并开始逼近或超过 `no_rl`。
