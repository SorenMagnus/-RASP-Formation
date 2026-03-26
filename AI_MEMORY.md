# AI_MEMORY - 当前周期交接文档

> 致下一位 AI / 工程师：
> 请先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动手改代码。
> 当前项目已经进入 `S5-only rl_param_only` 第一阶段，主阻塞点不再是 nominal governor release 本身，而是长时 PPO 训练无法跨关机恢复。

---

## 0. 当前开发游标

- 当前扫描基线：
  - `HEAD = aa43784bd3b3e22961a934aa9bc8935e1133daf3`
  - 重写本文件前，`git status --short` 为空；repo-tracked worktree 干净
- 当前论文主线：
  - 白盒主链仍是 `FSM -> nominal controller -> CBF-QP safety`
  - RL 目前仅作为第一阶段 `param-only supervisor` 接入，不控制 `mode`，不控制 `accel/steer`
- 当前有效成果：
  - `outputs/rl_train_s5_param_only/checkpoints/smoke.pt`
  - `outputs/s5_rl_smoke_eval_cuda/summary.csv`
  - `outputs/s5_rl_stage1_baseline__no_rl/summary.csv`
- 当前未完成成果：
  - `outputs/rl_train_s5_param_only/checkpoints/main.pt` 仍不存在
  - 本轮曾启动过 `200000` 步正式训练，但训练器只在训练结束后保存 checkpoint；如果关机，这段运行不能算有效成果
- 当前真实阻塞：
  - 不是继续修 nominal governor release
  - 而是先给 PPO 训练链路补上 `periodic checkpoint + rollout-boundary resume`

---

## 1. 技术栈红线

### 1.1 架构红线

- 必须保持三层白盒闭环：
  - `Mode Decision -> Nominal Controller -> Safety Filter`
- 严禁修改以下安全层核心文件：
  - `src/apflf/safety/safety_filter.py`
  - `src/apflf/safety/cbf.py`
  - `src/apflf/safety/qp_solver.py`
- RL 当前只允许 `param-only supervisor`：
  - 不允许直接控制 `mode`
  - 不允许直接输出 `accel`
  - 不允许直接输出 `steer`
- 不允许把论文主线改写成端到端 RL / 黑盒策略 / ROS 重构 / CUDA 重写
- 不允许引入 `SB3`；当前 PPO 后端必须继续沿用仓库内自定义 Torch 实现

### 1.2 接口红线

- 决策层公共接口已经冻结为：
  - `ModeDecision(mode, theta, source, confidence)`
- controller 公共入口已经冻结为：
  - `compute_actions(observation, mode, theta=None)`
- 必须保持：
  - `theta=None` 时精确退化到当前白盒基线
  - FSM 和 RL supervisor 共用同一 `ModeDecision` 输出形状

### 1.3 工程红线

- 每次改动后必须运行：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 所有实验必须可复现，seed 不能漂
- 不要在没有 resume 的前提下继续硬跑 `200000` 步长训

---

## 2. 已完成工作

### 2.1 决策层

- 已统一决策接口：
  - `src/apflf/utils/types.py` 中已有 `ModeDecision(mode, theta, source, confidence)`
- FSM 已接入统一接口：
  - `src/apflf/decision/fsm_mode.py` 返回 `ModeDecision`
  - FSM 默认输出确定性 `theta`
- RL supervisor 已落地：
  - `src/apflf/decision/rl_mode.py`
  - 已具备 fallback FSM、theta 投影、rate-limit、confidence gate、OOD gate

### 2.2 控制层

- controller 入口已统一为：
  - `compute_actions(observation, mode, theta=None)`
- 当前 RL 只通过 `theta` 调度 nominal 层的有界参数：
  - 不直接改 `accel`
  - 不直接改 `steer`
- 这条链路已在 controller / world / runner 中打通

### 2.3 RL 基础设施

- 以下 RL 文件已经在树上，且可导入、可运行：
  - `src/apflf/decision/rl_mode.py`
  - `src/apflf/rl/features.py`
  - `src/apflf/rl/env.py`
  - `src/apflf/rl/policy.py`
  - `src/apflf/rl/ppo.py`
  - `scripts/train_rl_supervisor.py`
  - `scripts/benchmark_s5_rl.py`
  - `tests/test_rl_supervisor.py`
- 当前 Torch GPU 环境已可用：
  - `torch 2.5.1+cu121`
  - CUDA 可见

### 2.4 诊断与产物

- summary / traj artifact 已持久化 RL 诊断字段，包括：
  - `rl_fallback_count`
  - `rl_confidence_mean`
  - `rl_confidence_min`
  - `theta_delta_linf_mean`
  - `theta_delta_linf_max`
  - `theta_clip_events`
- traj 中已落盘：
  - `decision_sources`
  - `decision_thetas`
  - `decision_theta_deltas`
  - `decision_confidences`
  - `decision_rl_fallbacks`
  - `decision_theta_clipped`

### 2.5 验证结果

- 非侵入式验证已通过：
  - `python -m pytest -q` -> `105 passed`
  - `python -m compileall src tests scripts` -> 通过
- `no_rl` 基线已冻结：
  - 文件：`outputs/s5_rl_stage1_baseline__no_rl/summary.csv`
  - `seed0 leader_final_x = 26.634774055574823`
  - `fallback_events = 203`
  - `safety_interventions = 238`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
- `500` 步 CUDA smoke 已成功：
  - checkpoint：`outputs/rl_train_s5_param_only/checkpoints/smoke.pt`
  - eval：`outputs/s5_rl_smoke_eval_cuda/summary.csv`
  - `leader_final_x = 26.634774055574823`
  - `fallback_events = 203`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
- smoke eval 中 RL 字段有效，但这只是链路验证，不是性能突破：
  - `rl_fallback_count = 220`
  - `rl_confidence_mean = 0.08346643664620139`
  - `theta_clip_events = 220`

---

## 3. 当前真实实验状态

- 当前 S5 论文级主问题还没有被 RL 打穿
- 当前能确认的只有：
  - RL 路径已接上
  - GPU smoke 训练可跑
  - checkpoint 保存链路至少能在训练结束时写出 `smoke.pt`
  - smoke eval 无碰撞/越界回归
- 当前还不能确认：
  - `rl_param_only` 是否能在 `seed 0/1/2` 上优于 `no_rl`
- 当前正式训练状态：
  - 曾经启动过 `200000` 步 CUDA 长训
  - 当前没有 `main.pt`
  - `main_stdout.log` 为空
  - `main_stderr.log` 仅有运行时 fallback 日志，不能替代 checkpoint
  - 由于 trainer 只在训练结束后保存，关机即丢失进度

---

## 4. 下一步指令

### 4.1 下一个工程师启动后，第一件事要写的代码

请直接修改：

- `src/apflf/rl/ppo.py`
- `scripts/train_rl_supervisor.py`
- `tests/test_rl_supervisor.py`

目标：

- 为 `S5-only rl_param_only` 的长时 PPO 训练加入：
  - 周期性 checkpoint
  - rollout-boundary resume

### 4.2 必须满足的数学 / 算法约束

设在第 `k` 个 rollout 完成采样与 PPO update 后保存训练态，记该训练态为：

`S_k = {network, optimizer, obs_stats(count, mean, m2), timesteps_done, rollout_seed_next, numpy_rng, torch_cpu_rng, torch_cuda_rng}`

必须满足：

1. 只允许在完整 rollout 边界保存
- 必须在“完整 rollout 收集 + 完整 PPO update”之后保存
- 严禁 mid-rollout 保存
- 严禁 mid-rollout 恢复

2. resume 必须恢复完整训练态
- 从 `S_k` 恢复后，后续 rollout 次序必须与未中断训练一致
- 后续 PPO update 次序必须与未中断训练一致
- 恢复后训练不能只恢复 `network`，必须同时恢复：
  - `optimizer`
  - `obs_stats.count`
  - `obs_stats.mean`
  - `obs_stats.m2`
  - `timesteps_done`
  - `rollout_seed_next`
  - `numpy_rng`
  - `torch_cpu_rng`
  - `torch_cuda_rng`

3. 不带 `--resume-from` 时必须精确退化
- 若 CLI 未提供 `--resume-from`
- 训练行为必须与当前版本精确一致
- 不能改变现有 smoke / training 默认行为

4. 周期性保存频率
- 默认每 `1` 个 rollout 保存一次
- 因当前 `steps_per_rollout = 512`
- 所以崩溃 / 关机时最大损失步数必须 `< 512`

5. 不改变 PPO 数学本体
- checkpoint 只保存训练态
- 不改 reward
- 不改 controller
- 不改 safety
- 不改 `theta` 空间定义

### 4.3 需要新增的训练行为

- `scripts/train_rl_supervisor.py` 必须新增：
  - `--resume-from`
- 训练目录中必须周期性写出：
  - `latest.pt`
- 建议行为：
  - 正式输出 `main.pt`
  - 周期性覆盖 `latest.pt`
  - 训练正常结束后，`main.pt` 仍作为最终 checkpoint

### 4.4 必须补的测试

在 `tests/test_rl_supervisor.py` 中新增 / 扩展测试，至少覆盖：

1. 训练态保存测试
- 一个短训练后能生成 `latest.pt`
- checkpoint 中包含完整 resume 必需字段

2. resume 等价性测试
- 同 seed 下：
  - 直接短训 `N` rollout
  - 先训 `k` rollout 保存，再从 `latest.pt` resume 到 `N` rollout
- 二者最终关键训练态必须对齐
- 至少核对：
  - `timesteps_done`
  - network 参数
  - optimizer 状态
  - obs stats

3. CLI 回归测试
- 不带 `--resume-from` 时行为与旧版一致
- 带 `--resume-from` 时可以继续推进 timesteps

### 4.5 下一阶段验收门槛

实现 resume 后，按以下顺序验收：

1. `python -m pytest -q`
2. `python -m compileall src tests scripts`
3. 新的最小 smoke：
  - 先生成 `latest.pt`
  - 再从 `latest.pt` resume 一次
  - 验证 checkpoint 会更新，timesteps 会继续推进
4. 只有 resume 路径通过后，才重新发起 `200000` 步 CUDA 正式训练
5. `main.pt` 真正生成后，再运行：
  - `python scripts/benchmark_s5_rl.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --rl-checkpoint outputs/rl_train_s5_param_only/checkpoints/main.pt --exp-id-prefix s5_rl_stage1_cuda --deterministic-eval`

阶段性成功判据仍是：

- `seed 0/1/2` 至少一组 `leader_final_x` 优于 `no_rl`
- `collision_count` 不回归
- `boundary_violation_count` 不回归
- 改进 seed 的 `fallback_events` 不恶化

---

## 5. 下次开机后的计划

1. 不依赖本轮后台长训的任何进度
- 默认视为 `main.pt` 不存在
- 默认视为当前长训进度无效

2. 第一件事不是重跑 `200000` 步
- 先实现 `PPO checkpoint + resume`

3. 写完后立刻做基础验证
- `python -m pytest -q`
- `python -m compileall src tests scripts`

4. 再做 resume smoke
- 短训生成 `latest.pt`
- 从 `latest.pt` 恢复一次
- 确认 timesteps 继续增长
- 确认 checkpoint 被刷新

5. resume 验证通过后，再重启正式训练
- 用 CUDA
- 用 `main.pt` 作为最终输出
- 用 `latest.pt` 作为周期性恢复点

6. 正式训练结束后再做 benchmark
- 对照 `outputs/s5_rl_stage1_baseline__no_rl/summary.csv`
- 跑 `seed 0/1/2`

7. benchmark 通过后再决定是否扩到 `S4`
- 在此之前，不允许扩到 `S4`
- 在此之前，不允许多场景混训

---

## 6. 不要做的事

- 不要再把下一步写回 nominal governor release
- 不要在没有 resume 的前提下继续硬跑 `200000` 步长训
- 不要修改：
  - `src/apflf/safety/safety_filter.py`
  - `src/apflf/safety/cbf.py`
  - `src/apflf/safety/qp_solver.py`
- 不要把 RL 扩到：
  - `mode-only`
  - `full supervisor`
- 不要现在扩到：
  - `S4`
  - 多场景混训
- 不要改 `ModeDecision` 的字段形状
- 不要改 `compute_actions(observation, mode, theta=None)` 的公共签名

---

## 7. 一句话交接

当前仓库已经从“是否引入 RL”推进到了“RL 第一阶段已接上、GPU smoke 已跑通”，但真正的工程阻塞点已经变成了“长时 PPO 训练无法跨关机恢复”。下一位工程师不要再回头修 nominal governor release，也不要碰 safety；请先把 `ppo.py + train_rl_supervisor.py + test_rl_supervisor.py` 的 `periodic checkpoint + rollout-boundary resume` 做完，再重启正式训练。
