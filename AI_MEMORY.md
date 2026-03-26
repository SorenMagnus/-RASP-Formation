# AI_MEMORY - 当前周期交接文档

> 致下一位 AI / 工程师：
> 请先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动手改代码。
> 当前项目已经进入 `S5-only rl_param_only` 第一阶段。主阻塞点已经不是 nominal governor release，而是 RL 长训的稳定执行、checkpoint 落盘和后续 benchmark 闭环。

---

## 0. 当前开发游标

- 当前扫描基线：
  - `HEAD = 1ccb1132629bea5e0aff45d6069854cd655b50b3`
  - 本次同步前，`git status --short` 显示未提交改动在：
    - `src/apflf/rl/ppo.py`
    - `tests/test_rl_supervisor.py`
  - 这些改动对应的是：`progress logging` 补充与相关测试
- 当前论文主线：
  - 白盒主链仍是 `FSM -> nominal controller -> CBF-QP safety`
  - RL 当前仅作为第一阶段 `param-only supervisor` 接入，不控制 `mode`，不控制 `accel/steer`
- 当前有效成果：
  - `outputs/s5_rl_stage1_baseline__no_rl/summary.csv`
  - `outputs/rl_train_s5_param_only/checkpoints/smoke.pt`
  - `outputs/s5_rl_smoke_eval_cuda/summary.csv`
  - `outputs/rl_resume_smoke/checkpoints/latest.pt`
  - `outputs/rl_resume_smoke/checkpoints/main.pt`
- 当前未完成成果：
  - `outputs/rl_train_s5_param_only/checkpoints/main.pt` 仍不存在
  - `outputs/rl_train_s5_param_only/checkpoints/latest.pt` 仍不存在
  - 最近一次 `200000` 步 CUDA 长训在首个 rollout 保存点之前异常退出，因此没有留下可恢复进度
- 当前真实阻塞：
  - `checkpoint + resume` 已实现
  - `progress logging` 已实现
  - 下一步不再是继续写训练器功能，而是重新发起长训，确认首个 `latest.pt` 落盘，然后等待 `main.pt`

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
- 长训只能在已经支持 `latest.pt` 周期保存和 `--resume-from` 的前提下运行

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

### 2.4 Checkpoint / Resume

- `PPOTrainer` 已支持：
  - 每个完整 rollout + PPO update 后保存 `latest.pt`
  - 从 `--resume-from` 在 rollout 边界恢复
- 训练态已写入 richer checkpoint payload，包括：
  - `optimizer_state_dict`
  - `obs_stats_count`
  - `obs_stats_mean`
  - `obs_stats_m2`
  - `timesteps_done`
  - `rollout_seed_next`
  - `initial_seed`
  - `numpy_rng_state`
  - `torch_cpu_rng_state`
  - `torch_cuda_rng_state_all`
  - `trainer_config`
- 保存方式已使用“临时文件 + 原子替换”，避免 `latest.pt` 被半写坏
- 推理兼容性保持不变：
  - `main.pt` / `latest.pt` 仍包含原推理所需字段
  - `policy.py` 无需修改即可加载推理所需内容

### 2.5 Progress Logging

- 训练器已支持 `stdout` 进度打印，并且每条都 `flush=True`
- 当前会打印的事件：
  - `[ppo] start`
  - `[ppo] resume`
  - `[ppo] rollout_start`
  - `[ppo] rollout_done`
  - `[ppo] complete`
- 每条日志包含的关键信息：
  - `device`
  - `seed`
  - `timesteps_done=current/total`
  - `progress=...%`
  - `elapsed_s`
  - `rollout_index`
  - `rollout_seed`
  - `batch_steps`
  - `policy_loss`
  - `value_loss`
  - `entropy`
  - `checkpoint=...`（在 rollout 保存后打印）
- 因此后台训练时，应优先查看：
  - `outputs/rl_train_s5_param_only/logs/main_stdout.log`
- safety fallback 仍会继续进入：
  - `outputs/rl_train_s5_param_only/logs/main_stderr.log`

### 2.6 诊断与产物

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

### 2.7 验证结果

- 当前验证结果：
  - `python -m pytest -q` -> `109 passed`
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
- `resume smoke` 已成功：
  - 路径：`outputs/rl_resume_smoke/checkpoints/`
  - 首次短训可生成 `latest.pt`
  - 从 `latest.pt` 恢复后可把 `timesteps_done` 从 `4` 续到 `8`
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
  - checkpoint / resume 路径可跑
  - progress logging 已可见
  - smoke eval 无碰撞/越界回归
- 当前还不能确认：
  - `rl_param_only` 是否能在 `seed 0/1/2` 上优于 `no_rl`
- 当前长训状态：
  - 最近一次 `200000` 步 CUDA 长训已异常退出
  - 首个 rollout 完成前未生成 `latest.pt`
  - 因此没有可恢复的长训进度
- 这意味着：
  - 现在不是继续写训练器功能
  - 而是重新发起长训，并利用新的 progress logging 观察它是否成功跨过首个 rollout 保存点

---

## 4. 下一步指令

### 4.1 下一个工程师启动后，第一件事要做什么

不要再继续改 `checkpoint + resume` 代码。  
请直接重新发起 `S5-only rl_param_only` 的 `200000` 步 CUDA 长训：

```bash
python scripts/train_rl_supervisor.py --config configs/scenarios/s5_dense_multi_agent.yaml --seed 0 --total-timesteps 200000 --steps-per-rollout 512 --learning-rate 3e-4 --device cuda --output outputs/rl_train_s5_param_only/checkpoints/main.pt
```

### 4.2 第一阶段运行验收

重新启动长训后，优先盯以下两处：

1. `main_stdout.log`
- 命令：
```powershell
Get-Content outputs/rl_train_s5_param_only/logs/main_stdout.log -Wait
```
- 预期能看到：
  - `[ppo] start`
  - `[ppo] rollout_start`
  - `[ppo] rollout_done ... checkpoint=...latest.pt`

2. `latest.pt`
- 命令：
```powershell
Test-Path outputs/rl_train_s5_param_only/checkpoints/latest.pt
```
- 一旦返回 `True`，说明首个可恢复保存点已建立

### 4.3 如果需要关机，如何回档

只有当下面文件存在时，才允许放心关机：

- `outputs/rl_train_s5_param_only/checkpoints/latest.pt`

下次开机后，必须用这个命令续跑：

```bash
python scripts/train_rl_supervisor.py --config configs/scenarios/s5_dense_multi_agent.yaml --seed 0 --total-timesteps 200000 --steps-per-rollout 512 --learning-rate 3e-4 --device cuda --resume-from outputs/rl_train_s5_param_only/checkpoints/latest.pt --output outputs/rl_train_s5_param_only/checkpoints/main.pt
```

注意：

- 不要用旧的 `smoke.pt` 续跑长训
- 长训回档点必须是 `latest.pt`

### 4.4 如果长训再次异常退出

不要立刻重写控制器或 safety。  
先做最小排查：

1. 看 `main_stdout.log`
- 确认是否至少打印到了 `[ppo] rollout_start`

2. 看 `main_stderr.log`
- 确认是否只有 fallback 输出
- 还是出现了 Python traceback / OOM / CUDA error

3. 看 `latest.pt` 是否存在
- 若不存在，说明仍死在首个 rollout 保存点之前
- 此时下一步应该补“异常 traceback 输出 / 首轮 heartbeat 更细粒度日志”，而不是改论文算法

### 4.5 长训完成后的实验闭环

只有 `main.pt` 真正生成后，才运行：

```bash
python scripts/benchmark_s5_rl.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --rl-checkpoint outputs/rl_train_s5_param_only/checkpoints/main.pt --exp-id-prefix s5_rl_stage1_cuda --deterministic-eval
```

阶段性成功判据仍是：

- `seed 0/1/2` 至少一组 `leader_final_x` 优于 `no_rl`
- `collision_count` 不回归
- `boundary_violation_count` 不回归
- 改进 seed 的 `fallback_events` 不恶化

---

## 5. 下次开机后的计划

1. 先确认当前长训是否仍在运行
- 若没有运行，则直接重新发起长训

2. 发起长训后立刻观察 `stdout`
- 用 `main_stdout.log` 看 `[ppo] start / rollout_start / rollout_done`

3. 首个关键目标不是 `main.pt`
- 而是先看到 `latest.pt` 成功落盘

4. 一旦 `latest.pt` 出现
- 以后就允许关机
- 以后就允许用 `--resume-from latest.pt` 续跑

5. 只有 `main.pt` 生成后，才进入 benchmark

6. benchmark 通过前，不允许扩到 `S4`

---

## 6. 不要做的事

- 不要再把下一步写回 nominal governor release
- 不要再回头重复实现 `checkpoint + resume`
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
- 不要用 `smoke.pt` 当长训恢复点

---

## 7. 一句话交接

当前仓库已经从“是否引入 RL”推进到了“RL 第一阶段已接上、checkpoint + resume 已实现、progress logging 已实现”。下一个工程师不要再回头修 governor release，也不要碰 safety；请直接重新发起 `200000` 步 CUDA 长训，盯住 `main_stdout.log` 与 `latest.pt`，先确保首个可恢复保存点建立，再等待 `main.pt` 和后续 benchmark。
