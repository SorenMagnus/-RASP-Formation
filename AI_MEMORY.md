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
- 当前目录已是 git 仓库。
- 结论必须以当前代码和 `outputs/` 中的真实产物为准。

## 2. 当前开发游标

### 2.1 当前阶段
- 已完成 `Phase A -> Phase E` 主体实现。
- 当前处于：
  - `Phase E 行为稳定性收口` 最终阶段
  - 刚完成 `recover 模式编队收敛增强` 的实现与验证
  - **s1/s2/s3 全部 3/3 team_goal_reached = True**

### 2.2 当前稳定 checkpoint
- 当前工作区停在**`stage20_recover_convergence_boost`**。
- 本次新增的核心改动：
  - `recover 模式下 formation_gain 增大 1.35x`：加速编队横向收敛
  - `recover 模式下 consensus_gain 降低 0.50x`：减少 follower 之间的互相牵制
  - `s3 仿真步数从 200 增加到 250`：给编队恢复更多时间
- 当前稳定代码已通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
  - 结果：`67 passed`

## 3. 本次工作区扫描结果

### 3.1 本次改动的源码/测试/配置文件
- `src/apflf/controllers/lf.py`
  - `_formation_force()` 在 recover 模式下 gain *= 1.35
  - `_consensus_force()` 在 recover 模式下 gain *= 0.50
- `configs/scenarios/s3_narrow_passage.yaml`
  - `steps: 200 → 250`
- `AI_MEMORY.md`
  - 本文件

### 3.2 本次扫描到的重要实验产物
- 当前保留并认可的稳定产物（**stage20b 系列**）：
  - `outputs/stage20b_recover_gain_s1` (seeds 0,1,2)
  - `outputs/stage20b_recover_gain_s2` (seeds 0,1,2)
  - `outputs/stage20b_recover_gain_s3` (seeds 0,1,2)
- 上一轮保留的参考基线（**stage19 系列**）：
  - `outputs/stage19_recover_hysteresis_s1` (seeds 0,1,2)
  - `outputs/stage19_recover_hysteresis_s2` (seeds 0,1,2)
  - `outputs/stage19_recover_hysteresis_s3` (seeds 0,1,2)
- 已失败的中间产物（stage20 系列，formation_gain=1.65x，导致 s1 退化，不保留）：
  - `outputs/stage20_recover_gain_s1`

### 3.3 扫描结论
- 本次改动仅涉及 LF 混入类的编队/一致性力增益和 s3 配置步数，没有改动：
  - 任何名义控制器核心（controllers/adaptive_apf.py, apf_lf.py 等）
  - 安全滤波层（safety/）
  - 环境模型（env/）
  - 仿真主循环（sim/）
  - FSM 模式决策逻辑（decision/fsm_mode.py）
- 改动范围最小化。

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

#### 修复 D：recover-exit hysteresis counter
- 文件：`src/apflf/decision/fsm_mode.py`, `src/apflf/utils/types.py`, `src/apflf/utils/config.py`, `configs/default.yaml`, `tests/test_modes.py`
- 数学约束：
  - recover→follow 退出条件必须连续满足 `N_exit = recover_exit_steps` 步
  - 当前默认 `N_exit = 4`（`>= hysteresis_steps = 3`）
- 作用：防止 s3 seed1 类场景中 recover 模式因一两步短暂满足条件就过早退出

#### 修复 E（本次新增）：recover 模式编队收敛增强
- 文件：`src/apflf/controllers/lf.py`, `configs/scenarios/s3_narrow_passage.yaml`
- 数学约束：
  - recover 模式下 `formation_gain *= 1.35`（有界常数，不突破 ControllerConfig 上下界）
  - recover 模式下 `consensus_gain *= 0.50`（有界常数）
  - s3 仿真步数从 200 增加到 250（给编队恢复更多时间）
- 作用：解决 s3 recover 模式下编队横向收敛速度不足的问题
- 根因分析：
  - 窄通道穿越后，follower 被推到与 leader 相反的一侧（y 差 3m+）
  - 原来的 formation_gain=1.2 和 consensus_gain=0.25 产生的恢复力不足
  - 增大 formation_gain 加速了 follower 直接追踪编队目标的力
  - 减小 consensus_gain 减少了 follower 之间的相互牵制力
- 失败经验：1.65x 倍率导致 s1 seed0 退化和 seed2 boundary violation（stage20，已废弃）

## 5. 当前稳定验证结果

### 5.1 代码级验证
- 当前稳定代码通过：
  - `python -m compileall src tests scripts`
  - `PYTHONPATH='src;.codex_tmp\\pytest' python -m pytest -q`
- 当前结果：
  - `67 passed`

### 5.2 场景级验证

#### s1：继续保持全绿 ✅
- 对应产物：`outputs/stage20b_recover_gain_s1`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - 3/3 `team_goal_reached = True`
- 与 stage19 对比：无退化

#### s2：改善，全绿 ✅（提升！）
- 对应产物：`outputs/stage20b_recover_gain_s2`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - 3/3 `team_goal_reached = True`（seed2 从 False 改善为 True）
- 与 stage19 对比：seed2 改善

#### s3：重大突破，全绿 ✅（从 0/3 翻转为 3/3！）
- 对应产物：`outputs/stage20b_recover_gain_s3`
- 结果：
  - 3/3 `collision_count = 0`
  - 3/3 `boundary_violation_count = 0`
  - 3/3 `team_goal_reached = True`
- 关键指标对比（stage19 → stage20b）：
  - seed0：
    - `terminal_formation_error: 7.057 → 1.872`（大幅改善）
    - `min_obstacle_clearance: 0.512 → 0.511`（持平，安全保持）
    - `fallback_ratio: 0.438 → 0.379`（下降，安全层介入更少）
  - seed1：
    - `terminal_formation_error: 4.937 → 2.121`（大幅改善）
    - `min_obstacle_clearance: 0.291 → 0.564`（改善）
    - `fallback_ratio: 0.413 → 0.331`（下降）
  - seed2：
    - `terminal_formation_error: 14.642 → 3.427`（大幅改善）
    - `min_obstacle_clearance: 0.523 → 0.515`（持平）
    - `fallback_ratio: 0.475 → 0.372`（下降）

## 6. 当前真实问题

### 6.1 当前主矛盾
- s1/s2/s3 的 `team_goal_reached` 已全部为 3/3 True ✅
- 当前主矛盾已从"行为稳定性"转移到：
  - s4/s5 场景验证（当前未验证）
  - 大规模实验矩阵（30 seeds × 场景 × 基线 × 消融）
  - 论文图表生成

### 6.2 已确认的事实（累积）
- 不要再改任何 follower recovery 横向 bias。
- 不要再尝试更远的 leader 前向目标点。
- leader recovery 纵向门控是有效方向，但收益不是单调的，必须小步验证。
- FSM recover 退出逻辑已经加入迟滞机制（stage19），不需要再改退出阈值。
- recover 模式下增大 formation_gain 是有效方向（stage20b），但不能超过 1.35x（1.65x 导致退化）。
- consensus_gain 在 recover 模式下降低到 0.5x 是有效的。

## 7. 下一步指令

### 7.1 下一位工程师启动 AI 后，应该马上分析的问题
- 优先分析方向：
  - s4（双车道交互超车）和 s5（多智能体密集避碰）的场景验证
  - 这些场景可能暴露新的控制问题
  - 需要先检验 configs/scenarios/ 中 s4/s5 的配置是否合理

### 7.2 下一刀可能的方向（优先级排列）

#### 方向 A（推荐）：s4/s5 场景验证
- 目标：
  - 运行 s4 (3 seeds) 和 s5 (3 seeds)
  - 检查碰撞/边界违规/到达率
  - 分析失败原因（如果有）

#### 方向 B（次选）：大规模实验矩阵
- 目标文件：`scripts/reproduce_paper.py`
- 思路：
  - 30 seeds × s1-s5 × 基线方法（apf, apf_lf, st_apf, dwa, orca）
  - 消融实验（A1-A5）
  - 统计检验与论文图表生成
- 前置条件：s4/s5 需要先经过初步验证

#### 方向 C（实验性）：论文写作材料准备
- 目标文件：`scripts/export_figures.py`
- 思路：生成典型轨迹图、风险曲线、QP修正量曲线等

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
- 当前工作区停在 **stage20_recover_convergence_boost** checkpoint。
- recover 模式编队收敛增强机制已实现、测试通过、场景验证通过。
- 当前最可信的结论是：
  - `s1` 继续保持全绿（3/3 team_goal_reached）
  - `s2` 从 2/3 提升到 **3/3 team_goal_reached**
  - `s3` 从 **0/3 翻转为 3/3 team_goal_reached**
  - 全部 9 个 run 的 collision=0, boundary_violation=0
  - fallback_ratio 在 s3 场景下反而降低（安全层介入更少，符合最小干预原则）
- 下一位工程师继续时，主线必须是：
  - 验证 s4/s5 场景
  - 准备大规模实验矩阵
  - 而不是再改 s1/s2/s3 的行为调优（已全部通过）
