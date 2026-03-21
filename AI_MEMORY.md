# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标
- 目标不是做演示 demo，而是持续收敛到"论文级 artifact"。
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
- 当前目录仍然不是 git 仓库，所以实验输出里的 `git_commit` 仍然是 `nogit` 或自动生成的 hash。
- 结论必须以当前代码和 `outputs/` 中的真实产物为准。

## 2. 当前开发游标

### 2.1 当前阶段
- 已完成 `Phase A -> Phase E` 主体实现。
- 当前处于：
  - `Phase E 行为稳定性收口`
  - `hard scenes 审计与调优`
  - 刚完成 `recover 退出迟滞` 机制的实现与验证

### 2.2 当前稳定 checkpoint
- 当前工作区停在"已验证安全、可继续研究"的稳定线：**`stage19_recover_hysteresis`**。
- 本次新增的核心改动：
  - `recover-exit hysteresis counter`（recover 退出迟滞计数器）
  - 在 `FSMModeDecision` 中新增 `_recover_exit_count` 计数器
  - recover→follow 退出条件必须**连续满足 `recover_exit_steps` 步**才允许退出
  - 任何一步不满足恢复条件，计数器立即归零
- 当前稳定代码已通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
  - 结果：`67 passed`

## 3. 本次工作区扫描结果

### 3.1 本次改动的源码/测试/配置文件
- `src/apflf/utils/types.py`
  - `DecisionConfig` 新增字段 `recover_exit_steps: int = 4`
- `src/apflf/utils/config.py`
  - `_load_decision()` 新增解析 `recover_exit_steps`
- `src/apflf/decision/fsm_mode.py`
  - `FSMModeDecision.__init__` 新增 `_recover_exit_count: int = 0`
  - `_candidate_mode()` 新增 recover 退出迟滞逻辑（lines 105–137）
  - 当 `needs_recovery=True` 时重置计数器
  - 当 `needs_recovery=False` 且当前在 recover 模式时，累加计数器
  - 仅当计数器达到 `recover_exit_steps` 时才允许退出 recover
  - 退出到 follow/default 时也重置计数器
- `configs/default.yaml`
  - decision 节新增 `recover_exit_steps: 4`
- `tests/test_modes.py`
  - `_decision_config()` 增加 `recover_exit_steps` 参数
  - 新增测试 `test_fsm_recover_exit_requires_sustained_satisfaction`
  - 新增测试 `test_fsm_recover_exits_after_consecutive_satisfaction`
- `AI_MEMORY.md`
  - 本文件

### 3.2 本次扫描到的重要实验产物
- 当前保留并认可的稳定产物（**stage19 系列**）：
  - `outputs/stage19_recover_hysteresis_s1` (seeds 0,1,2)
  - `outputs/stage19_recover_hysteresis_s2` (seeds 0,1,2)
  - `outputs/stage19_recover_hysteresis_s3` (seeds 0,1,2)
- 上一轮保留的参考基线（**stage18_tune2 系列**）：
  - `outputs/stage18_tune2_s1_seed0..2`
  - `outputs/stage18_tune2_s2_seed0..2`
  - `outputs/stage18_tune2_s3_seed0..2`

### 3.3 扫描结论
- 本次改动仅涉及 FSM 决策层的 recover 退出逻辑，没有改动：
  - 任何名义控制器（controllers/）
  - 安全滤波层（safety/）
  - 环境模型（env/）
  - 仿真主循环（sim/）
- 改动范围最小化，符合"只改 fsm_mode.py + 联动 test_modes.py"的指令。

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

### 4.2 当前保留的有效修复（累积）

#### 修复 A：leader 过终点后仍保持前向目标
- 文件：`src/apflf/controllers/apf_lf.py`, `src/apflf/controllers/adaptive_apf.py`
- 作用：避免 leader 一过 `goal_x` 就把吸引点钉死在终点位置

#### 修复 B：leader recovery speed relief
- 文件：`src/apflf/controllers/base.py`
- 作用：队友明显超前且没有明显掉队时，leader 允许继续向前拉开

#### 修复 C：leader post-goal recovery speed floor
- 文件：`src/apflf/controllers/base.py`
- 公式：若 `leader.x >= goal_x` 且未恢复完成：
  - `v_cap = min(v_target_max, 0.35 + 0.35 * v_target_max * speed_scale)`
  - `speed_scale = max(0.18, min(lag_scale, error_scale))`
- 作用：修掉 recover 模式下 leader 一过 goal_x 就被钉停的问题

#### 修复 D（本次新增）：recover-exit hysteresis counter
- 文件：`src/apflf/decision/fsm_mode.py`, `src/apflf/utils/types.py`, `src/apflf/utils/config.py`, `configs/default.yaml`, `tests/test_modes.py`
- 数学约束：
  - recover→follow 退出条件必须连续满足 `N_exit = recover_exit_steps` 步
  - 当前默认 `N_exit = 4`（`>= hysteresis_steps = 3`，定义满足 AI_MEMORY §7.2）
  - 不降低任何恢复硬阈值
  - 不改动 follower 横向 bias
  - 安全层最小干预原则不受影响
- 作用：防止 s3 seed1 类场景中 recover 模式因一两步短暂满足条件就过早退出

## 5. 当前稳定验证结果

### 5.1 代码级验证
- 当前稳定代码通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
- 当前结果：
  - `67 passed`（原 65 + 新增 2 个 recover 退出迟滞测试）

### 5.2 场景级验证

#### s1：继续保持全绿 ✅
- 对应产物：`outputs/stage19_recover_hysteresis_s1`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - 3/3 `team_goal_reached = True`
- 与 stage18_tune2 对比：无退化

#### s2：安全保持，结论不变 ✅
- 对应产物：`outputs/stage19_recover_hysteresis_s2`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - `team_goal_reached = 2/3`（seed0 ✅, seed1 ✅, seed2 ✗）
- 与 stage18_tune2 对比：无退化

#### s3：全组保持安全，recover 退出时机改善 ⚠️
- 对应产物：`outputs/stage19_recover_hysteresis_s3`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - `team_goal_reached = 0/3`
- 关键指标对比（stage18_tune2 → stage19）：
  - seed0：
    - `terminal_formation_error: 7.057 → 7.057`（持平）
    - `min_obstacle_clearance: 0.512 → 0.512`（持平）
    - `fallback_ratio: 0.438 → 0.438`（持平）
  - seed1：
    - `terminal_formation_error: 4.942 → 4.937`（微改善）
    - `min_obstacle_clearance: 0.287 → 0.291`（微改善）
    - `time_to_recover_formation: 18.0 → 18.3`（recover 持续更久，符合预期）
  - seed2：
    - `terminal_formation_error: 14.642 → 14.642`（持平）
    - `min_obstacle_clearance: 0.523 → 0.523`（持平）

## 6. 当前真实问题

### 6.1 当前主矛盾
- 当前主矛盾已经不是"缺模块"，也不再是"recover 过早退出"。
- 当前主矛盾是：
  - `s3` 虽然已经全组安全，但 `team_goal_reached` 仍然是 `0/3`
  - recover 退出迟滞机制已就位（本次实现），seed1 的 `time_to_recover_formation` 从 18.0 上升到 18.3 证明机制生效
  - 但 s3 的根本问题是：**recover 模式下编队恢复本身效率不足**，即使给了更多时间也无法在仿真结束前完成恢复
  - seed0、seed2 的 `time_to_recover_formation = nan` 说明根本没有达到恢复完成状态

### 6.2 已确认的事实（累积）
- 不要再改任何 follower recovery 横向 bias。
- 不要再尝试更远的 leader 前向目标点。
- leader recovery 纵向门控是有效方向，但收益不是单调的，必须小步验证。
- FSM recover 退出逻辑已经加入迟滞机制（本次），不需要再改退出阈值。
- s3 的 `team_goal_reached = 0/3` 问题根源在 recover 模式下**编队收敛速度**，不是退出时机。

## 7. 下一步指令

### 7.1 下一位工程师启动 AI 后，应该马上分析的问题
- 优先分析方向：
  - s3 的 3 个 seed 中，recover 模式下编队为什么不能在仿真时间内恢复完成？
  - 需要读取 `outputs/stage19_recover_hysteresis_s3/traj/*.npz` 轨迹数据
  - 关注 recover 模式时段内各车辆的横向偏移和纵向间距变化趋势

### 7.2 下一刀可能的方向（优先级排列）

#### 方向 A（推荐）：提高 recover 模式下的编队收敛速度
- 目标文件：
  - `src/apflf/controllers/base.py`（recover 模式下 follower 的控制增益）
  - 可能联动：`src/apflf/controllers/apf_lf.py`
- 思路：
  - 在 recover 模式下，适当增大 `formation_gain` 和 `lateral_gain` 的权重
  - 或者减小 `consensus_gain` 以降低编队内部耦合对横向恢复的阻力
  - 必须保持参数有界（不能突破 `ControllerConfig` 中的上下界）

#### 方向 B（次选）：增加仿真步数让 s3 有更多恢复时间
- 目标文件：`configs/scenarios/s3_narrow_passage.yaml`
- 思路：将 `steps` 从当前值适当增加（例如 200 → 240 或 260）
- 风险：这只是治标不治本，但可以先验证"给足够时间是否能恢复"

#### 方向 C（实验性）：recover 模式下动态调整 leader 速度上限
- 目标文件：`src/apflf/controllers/base.py`
- 思路：在 recover 模式下，根据 `max_teammate_lag` 动态调低 leader 速度，让队友追赶更快
- 风险：可能与修复 C（post-goal speed floor）冲突，需仔细测试

### 7.3 下一刀必须满足的数学约束
- 在 `preview horizon H` 内仍必须保持：
  - 边界安全：`h_b(x_{t+k}) >= 0`
  - 障碍/车间安全：`h_obs(x_{t+k}) >= 0`
- nominal/safety 的最小干预原则仍必须保持：
  - 如果 `u_nom` 已满足所有安全约束，则 `u_safe = u_nom`
- 任何控制增益调整必须有界且在配置文件中可配置
- 不允许在 recover 模式中引入硬编码常数

### 7.4 下一刀的验证顺序
1. `python -m compileall src tests scripts`
2. `python -m pytest -q`（必须 ≥ 67 passed）
3. `s1`（3 seeds）
4. `s2`（3 seeds）
5. `s3`（3 seeds）
- 只要 `s1` 任一 seed 退化，立即回退
- 只要出现：
  - `collision_count > 0`
  - `boundary_violation_count > 0`
  - `min_obstacle_clearance = 0.0`
  就必须立即回退

## 8. 当前稳定结论
- 当前工作区停在 **stage19_recover_hysteresis** checkpoint。
- recover 退出迟滞机制已实现、测试通过、场景验证通过。
- 当前最可信的结论是：
  - `s1` 继续保持全绿（3/3 team_goal_reached）
  - `s2` 保持安全且结论不变（2/3 team_goal_reached）
  - `s3` 在不破坏安全的前提下，recover 退出时机得到改善（seed1 时间 +0.3s）
  - 但 `s3 team_goal_reached = 0/3` 仍未解决
- 下一位工程师继续时，主线必须是：
  - 分析 s3 recover 模式下编队恢复效率的瓶颈
  - 而不是再改 FSM 退出逻辑（已就位）
  - 更不是任何 follower recovery 横向 bias
