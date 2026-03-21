# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标
- 目标不是做演示 demo，而是持续收敛到“论文级 artifact”。
- 主架构必须始终保持三层闭环：
  - `Nominal Controller`
  - `Safety Filter (CBF-QP + OSQP)`
  - `Mode Decision (FSM 为主，RL 仅允许作为后续可选扩展)`

### 1.2 技术栈约束
- 仅允许 Python 技术栈，当前基线：
  - `Python 3.10+`
  - `numpy`
  - `scipy`
  - `PyYAML`
  - `matplotlib`
  - `osqp`
  - `pytest`
- 禁止引入：
  - ROS
  - CUDA 依赖
  - 端到端黑盒控制器替代现有三层结构
- 安全层必须保持：
  - `CBF-QP`
  - `OSQP`
  - preview verification
  - fallback 机制
- 实验链必须保持：
  - headless CLI
  - YAML 配置驱动
  - 全链路 seed 可复现
  - 输出包含 `config_resolved.yaml / summary.csv / traj/*.npz`

### 1.3 验证红线
- 每次代码改动后至少通过：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`
- 任何时候都不允许带着已知回归进入下一轮开发。
- 行为调优类改动必须按顺序验证：
  - `s1`
  - `s2`
  - `s3`
- 只要 `s1` 任一 seed 退化，就必须立即回退，不能继续扩展到其他场景。

### 1.4 仓库现实约束
- 当前目录仍然不是 git 仓库，所以实验输出里的 `git_commit` 仍然是 `nogit`。
- 结论必须以当前代码和 `outputs/` 中的真实产物为准。

## 2. 当前开发游标

### 2.1 当前阶段
- 已完成 `Phase A -> Phase E` 主体实现。
- 当前处于：
  - `Phase E 行为稳定性收口`
  - `hard scenes 审计与调优`

### 2.2 当前稳定 checkpoint
- 当前工作区最终停在一条“已验证安全、可继续研究”的稳定线。
- 当前稳定线只保留了今天真正通过验证的新增逻辑：
  - `leader post-goal recovery speed floor`
- 这条逻辑的作用是：
  - 在 `recover` 模式下，leader 过了 `goal_x` 之后，不再因为 `target_speed=0` 被直接钉停
  - 如果队形还没有恢复完成，leader 保留一个按 `lag/error` 缩放的低速前行上限
- 当前稳定代码已重新通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
  - 结果：`65 passed`

## 3. 本次工作区扫描结果

### 3.1 今天扫描到的源码/测试改动文件
- `src/apflf/controllers/base.py`
- `src/apflf/controllers/apf_lf.py`
- `src/apflf/controllers/adaptive_apf.py`
- `src/apflf/decision/fsm_mode.py`
- `tests/test_modes.py`
- `AI_MEMORY.md`

### 3.2 今天扫描到的重要实验产物
- 保留并认可的稳定产物：
  - `outputs/stage18_tune2_s1_seed0`
  - `outputs/stage18_tune2_s1_seed1`
  - `outputs/stage18_tune2_s1_seed2`
  - `outputs/stage18_tune2_s2_seed0`
  - `outputs/stage18_tune2_s2_seed1`
  - `outputs/stage18_tune2_s2_seed2`
  - `outputs/stage18_tune2_s3_seed0`
  - `outputs/stage18_tune2_s3_seed1`
  - `outputs/stage18_tune2_s3_seed2`
- 已扫描但判定为失败尝试、必须忽略的产物：
  - `outputs/stage18_tune1_*`
  - `outputs/stage18_tune3_*`
  - `outputs/stage18_tune4_*`
  - 这些目录只用于失败试验留痕，不代表当前代码应达到的状态

### 3.3 扫描结论
- `apf_lf.py` 和 `adaptive_apf.py` 今天被碰过，但最终没有留下新的激进前向目标点补丁。
- `fsm_mode.py` 今天被试过更激进的 recover 退出修复，但最终都已回退，当前不保留新的退出逻辑。
- 当前真正新增且被保留的核心改动，只有 `base.py` 中的 post-goal recovery speed floor。

## 4. 已完成工作

### 4.1 三层架构与实验链
- 已实现并接通：
  - `APF / ST-APF / APF-LF / adaptive APF`
  - `CBF-QP + OSQP + preview + fallback`
  - `FSM mode decision`
- 环境与仿真主链完整：
  - `src/apflf/env/geometry.py`
  - `src/apflf/env/road.py`
  - `src/apflf/env/dynamics.py`
  - `src/apflf/env/obstacles.py`
  - `src/apflf/sim/world.py`
  - `src/apflf/sim/runner.py`
- 论文实验与导出链可用：
  - `src/apflf/analysis/metrics.py`
  - `src/apflf/analysis/stats.py`
  - `src/apflf/analysis/export.py`
  - `scripts/reproduce_paper.py`
  - `scripts/export_figures.py`
  - `src/apflf/sim/replay.py`

### 4.2 当前保留的有效修复

#### 修复 A：leader 过终点后仍保持前向目标
- 文件：
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/controllers/adaptive_apf.py`
  - `tests/test_modes.py`
- 作用：
  - 避免 leader 一过 `goal_x` 就把吸引点钉死在终点位置，导致 `force_x` 退化

#### 修复 B：leader recovery speed relief
- 文件：
  - `src/apflf/controllers/base.py`
  - `tests/test_modes.py`
- 作用：
  - 当队友已经明显超前、且没有明显掉队时，leader 在 recover 末段允许继续向前拉开

#### 修复 C：leader post-goal recovery speed floor
- 文件：
  - `src/apflf/controllers/base.py`
  - `tests/test_modes.py`
- 当前保留公式：
  - 若 `leader.x >= goal_x` 且仍未恢复完成，则
  - `v_cap = min(v_target_max, 0.35 + 0.35 * v_target_max * speed_scale)`
  - `speed_scale = max(0.18, min(lag_scale, error_scale))`
  - `lag_scale = spacing / (spacing + max_lag)`
  - `error_scale = spacing / (spacing + max_error)`
- 作用：
  - 修掉 recover 模式下 leader 一过 `goal_x` 就被钉停的问题

### 4.3 今天已验证但已回退的失败尝试

#### 失败尝试 1：更远的 leader recovery forward target
- 试验线：
  - `stage18_tune1_*`
- 结论：
  - 未能改善 `team_goal_reached`
  - 使 `s3` 的安全裕度恶化
  - 已完全回退，不在当前代码中

#### 失败尝试 2：FSM recover lead 阈值微调
- 试验线：
  - `stage18_tune3_*`
  - `stage18_tune4_*`
- 结论：
  - 没有形成安全可保留解
  - `stage18_tune4_s3_seed1` 明确退化到 `collision_count = 6`
  - 已完全回退，不在当前代码中

## 5. 当前稳定验证结果

### 5.1 代码级验证
- 当前稳定代码通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
- 当前结果：
  - `65 passed`

### 5.2 场景级验证

#### s1：继续保持全绿
- 对应产物：
  - `outputs/stage18_tune2_s1_seed0/summary.csv`
  - `outputs/stage18_tune2_s1_seed1/summary.csv`
  - `outputs/stage18_tune2_s1_seed2/summary.csv`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - 3/3 `team_goal_reached = True`

#### s2：安全保持，全组结论不变
- 对应产物：
  - `outputs/stage18_tune2_s2_seed0/summary.csv`
  - `outputs/stage18_tune2_s2_seed1/summary.csv`
  - `outputs/stage18_tune2_s2_seed2/summary.csv`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - `team_goal_reached = 2/3`
- 说明：
  - `seed2` 仍未收口，但这轮没有退化

#### s3：全组保持安全，恢复质量部分改善
- 对应产物：
  - `outputs/stage18_tune2_s3_seed0/summary.csv`
  - `outputs/stage18_tune2_s3_seed1/summary.csv`
  - `outputs/stage18_tune2_s3_seed2/summary.csv`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - `team_goal_reached = 0/3`
- 相比上一稳定基线 `stage14_recovery_speed_relief_s3`：
  - seed0：
    - `terminal_formation_error: 9.4357 -> 7.0569`
    - `min_obstacle_clearance: 0.3476 -> 0.5117`
    - `fallback_ratio: 0.4817 -> 0.4383`
  - seed1：
    - `terminal_formation_error: 3.9424 -> 4.9425`
    - 仍安全，但恢复质量略差
  - seed2：
    - `terminal_formation_error: 15.4365 -> 14.6423`
    - `min_obstacle_clearance: 0.4775 -> 0.5233`

## 6. 当前真实问题

### 6.1 当前主矛盾
- 当前主矛盾已经不是“缺模块”。
- 当前主矛盾是：
  - `s3` 虽然已经全组安全，但 `team_goal_reached` 仍然是 `0/3`
  - `seed1` 在 recover 末段会过早回到 `follow`
  - `seed0/seed2` 在 recover 末段依然存在“队友逐步超前 leader”的现象

### 6.2 今天已经确认的事实
- 不要再改任何 follower recovery 横向 bias。
- 不要再尝试更远的 leader 前向目标点。
- leader recovery 纵向门控是有效方向，但收益不是单调的，必须小步验证。
- FSM recover 退出逻辑确实是下一步瓶颈，但今天尝试过的“直接下调 lead 阈值”没有形成可保留解。

## 7. 下一步指令

### 7.1 下一位工程师启动 AI 后，应该马上写哪段代码
- 优先目标文件：
  - `src/apflf/decision/fsm_mode.py`
  - 必要时联动：
    - `tests/test_modes.py`
- 下一刀要做的不是降低 recover 退出硬阈值，而是：
  - 给 `recover -> follow` 的退出逻辑增加“退出迟滞”或“连续满足条件若干步后才退出 recover”的机制

### 7.2 下一刀必须满足的数学约束
- 在 `preview horizon H` 内仍必须保持：
  - 边界安全：
    - `h_b(x_{t+k}) >= 0`
  - 障碍/车间安全：
    - `h_obs(x_{t+k}) >= 0`
- nominal/safety 的最小干预原则仍必须保持：
  - 如果 `u_nom` 已满足所有安全约束，则 `u_safe = u_nom`
- 新的 recover 退出逻辑必须满足：
  - 只允许在 `recover` 模式内部生效
  - 不能降低现有恢复硬阈值去换行为
  - 退出 recover 前，必须要求恢复条件连续满足 `N_exit` 步，而不是单步满足就退出
  - 建议 `N_exit >= decision.hysteresis_steps`
  - 恢复条件至少显式使用以下量中的一种或多种：
    - `max_teammate_lag`
    - `team_formation_error`
    - `max_centerline_offset`

### 7.3 下一刀的实现建议
- 建议在 `FSMModeDecision` 内增加 recover-exit counter，而不是改风险阈值或 lead 阈值。
- 推荐思路：
  - 进入 `recover` 后，只有当“恢复完成条件”连续满足 `N_exit` 步，才允许候选模式切回 `follow`
  - 一旦中间任何一步不满足，则 counter 归零
- 这样可以解决今天已经观察到的真实问题：
  - `seed1` 在 recover 末段因为一两步短暂满足条件而过早退回 `follow`

### 7.4 下一刀的验证顺序
1. `python -m compileall src tests scripts`
2. `python -m pytest -q`
3. `s1`
4. `s2`
5. `s3`
- 只要 `s1` 任一 seed 退化，立即回退
- 只要出现：
  - `collision_count > 0`
  - `boundary_violation_count > 0`
  - `min_obstacle_clearance = 0.0`
  就必须立即回退

## 8. 当前稳定结论
- 当前工作区最终停在一个“已验证安全、对 s3 有部分改善、但还没彻底收口”的 checkpoint。
- 当前最可信的结论是：
  - `s1` 已继续保持全绿
  - `s2` 保持安全且结论不变
  - `s3` 在不破坏安全的前提下，`seed0/seed2` 的恢复质量有所改善
  - 但 `s3 team_goal_reached = 0/3` 仍未解决
- 下一位工程师明天继续时，主线必须是：
  - `recover 退出迟滞`
  - 而不是更激进的 leader 前向目标点
  - 更不是任何 follower recovery 横向 bias
