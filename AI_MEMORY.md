# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标红线
- 目标不是做一个能跑的 demo，而是持续收敛到可投稿 IEEE 级别论文的可复现 artifact。
- 主架构必须始终保持三层闭环，禁止退化成单层启发式或黑盒端到端：
  - `Nominal Controller`
  - `Safety Filter (CBF-QP + OSQP)`
  - `Mode Decision (FSM 为主，RL 仅可作为离散决策可选增强)`

### 1.2 技术栈红线
- 仅允许 Python 技术栈，当前仓库基线：
  - `Python 3.10+`
  - `numpy`
  - `scipy`
  - `PyYAML`
  - `matplotlib`
  - `osqp`
  - `pytest`
- 禁止默认引入：
  - ROS
  - CUDA 依赖
  - 端到端黑盒连续控制器

### 1.3 安全层红线
- `Safety Filter` 必须保留：
  - `CBF-QP`
  - `OSQP`
  - `preview verification`
  - `fallback`
- 禁止通过删除 `preview verification`、削弱 `fallback`、放松 one-step safety 约束来“刷过”场景。
- 当前主要矛盾仍在 nominal 几何与速度调制，不在 safety 层。
- 除非用户明确要求，禁止直接改 `src/apflf/safety/safety_filter.py` 去换结果。

### 1.4 工程与复现红线
- 必须保持：
  - headless CLI
  - YAML 配置驱动
  - 全链路 seed 可复现
  - 结果导出可重算
- 每次有效代码改动后至少验证：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 实验输出至少应包含：
  - `config_resolved.yaml`
  - `summary.csv`
  - `traj/*.npz`

## 2. 当前开发游标

### 2.1 仓库与工作树状态
- 当前 `HEAD`：
  - commit: `789fa3c9f79d8a447661c338ad90d9ab1a03d6bf`
  - 时间: `2026-03-23 23:30:54 +08:00`
  - 提交说明: `第十一次更新`
- 当前工作树相对 `HEAD` 仅有一个修改文件：
  - `AI_MEMORY.md`
- 也就是说：
  - 所有 `src/` 与 `tests/` 代码当前都已经回到 clean-HEAD 行为
  - 本轮 boundary gate 试探已经完整回退，没有遗留未提交代码

### 2.2 当前代码主线
- 当前 nominal 主线已经包含并保留：
  - `src/apflf/controllers/apf_lf.py`
    - `_leader_side_target_y(...)`
    - `_leader_hazard_target_x(...)`
    - `_leader_preflip_target_blend(...)`
  - 其中 `preflip_blend` 当前稳定版本仍是：
    - `blend = commitment_activation * gap_activation`
    - 上界 `0.55`
- 当前 adaptive nominal 还包含：
  - `src/apflf/controllers/adaptive_apf.py`
    - `_leader_hazard_speed_limit(...)`
    - `_leader_low_speed_braking_cap(...)`
- 当前安全层主线：
  - `src/apflf/safety/safety_filter.py`
- 当前模式决策主线：
  - `src/apflf/decision/fsm_mode.py`
  - `rl_mode.py` / `game_heuristic.py` 仍是可选扩展，不是当前 blocker

### 2.3 当前验证状态
- 最近一次完整验证：
  - `python -m pytest -q` -> `86 passed in 184.27s`
  - `python -m compileall src tests scripts` -> 通过

### 2.4 当前最可信参考输出
- 当前 stable nominal 逻辑的实验参考主要看两类输出：
  - 主仓库已有历史单点结果：
    - `outputs/context_sync_s5_preflip_blend_v1/summary.csv`
  - 本轮为 clean-HEAD 对照额外生成的临时工作树结果：
    - `D:\11111\-RASP-Formation__baseline_head\outputs\context_sync_boundary_gate_baseline__s1_local_minima`
    - `D:\11111\-RASP-Formation__baseline_head\outputs\context_sync_boundary_gate_baseline__s2_dynamic_crossing`
    - `D:\11111\-RASP-Formation__baseline_head\outputs\context_sync_boundary_gate_baseline__s5_dense_multi_agent`
- 说明：
  - 主仓库里 `context_sync_boundary_gate_smoke__*` 是 boundary-gate 试探阶段产物，只能用于诊断，不应再当作主线结果
  - 当前代码已经回退，所以若要复现实验，应以 clean-HEAD 逻辑重新跑主仓库输出，或参考上面的 baseline worktree 结果

## 3. 已完成工作

### 3.1 架构级工作
- 三层主架构已经打通：
  - nominal controller
  - safety filter
  - mode decision
- 可复现实验闭环已经具备：
  - `scripts/run_experiment.py`
  - `scripts/reproduce_paper.py`
  - `src/apflf/sim/runner.py`
  - `src/apflf/sim/replay.py`
- 分析与论文导出链路已经具备基础版本：
  - `src/apflf/analysis/metrics.py`
  - `src/apflf/analysis/stats.py`
  - `src/apflf/analysis/export.py`

### 3.2 nominal 控制层已完成的关键工作
- `APF-LF` 与 `Adaptive APF` 已完成主线融合。
- leader hazard 几何已经不再只是“贴边走”：
  - 已显式区分 passing side
  - 已引入 corridor centerline blend
  - 已引入 flip overshoot
  - 已引入 hazard-local target_x
  - 已引入 pre-flip target_y 提前偏转
- 当前最重要的 nominal 收获不是新模块，而是：
  - pre-flip 几何修正已经正式进入主线
  - leader 在 hard local flip 之前就能开始向 alternate corridor 漂移

### 3.3 safety 与 decision 已完成工作
- `CBF-QP + preview verification + fallback` 已工作，并有对应测试。
- `FSM + hysteresis + recover` 已工作。
- 当前主版本依赖 FSM，不依赖 RL。

### 3.4 测试与工程质量
- 当前完整测试通过数为 `86`。
- `tests/test_adaptive_apf.py` 已包含与当前 nominal 主线一致的关键回归：
  - `test_adaptive_apf_keeps_full_leader_reference_speed_before_local_flip`
  - `test_adaptive_apf_preflip_target_y_starts_shifting_before_hard_local_flip`
  - `test_adaptive_apf_leader_reference_speed_throttles_during_staggered_hazard_reorientation`
  - `test_adaptive_apf_caps_low_speed_hazard_braking_before_self_stop`

### 3.5 本轮额外完成的实验工作
- 已完成 `S1/S2/S5 × seeds 0 1 2` 小范围 sweep。
- 已完成 clean-HEAD 基线对照：
  - 在临时工作树 `D:\11111\-RASP-Formation__baseline_head` 中运行同样的 `S1/S2/S5 × seeds 0 1 2`
- 已完成 boundary-aware preflip gate 的完整验证与回退：
  - 试探版代码曾加入 `boundary-aware preflip gating`
  - 小范围对照后确认其不具备稳健收益
  - 已从主线完全回退

## 4. 当前最可信实验事实

### 4.1 S1 local minima
- `S1` 当前代码与 clean-HEAD 对照完全一致，说明这组结果不是 boundary gate 引入的。
- seed 级摘要：
  - seed0:
    - `leader_final_x = 111.31568626878654`
    - `reached_goal = True`
    - `team_goal_reached = True`
    - `boundary_violation_count = 0`
  - seed1:
    - `leader_final_x = 126.91361401785045`
    - `reached_goal = True`
    - `team_goal_reached = False`
    - `boundary_violation_count = 1`
  - seed2:
    - `leader_final_x = 118.11507950033955`
    - `reached_goal = True`
    - `team_goal_reached = False`
    - `boundary_violation_count = 0`
- 结论：
  - `S1` 仍有 preexisting 难点，尤其是 seed1 的边界与编队恢复质量
  - 但这不是本轮 boundary gate 的副作用

### 4.2 S2 dynamic crossing
- `S2` 当前代码与 clean-HEAD 对照完全一致。
- 三个 seed 全部满足：
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `reached_goal = True`
  - `team_goal_reached = True`
- 结论：
  - `S2` 当前主线已相对稳定，不是当前 blocker

### 4.3 S5 dense multi-agent
- `S5` 是当前论文主结果最关键也最未完成的难场景。
- 对应当前 stable 代码逻辑，可信参考应看 clean-HEAD baseline worktree 输出：
  - seed0:
    - `leader_final_x = 26.674232747808983`
    - `fallback_events = 205`
    - `safety_interventions = 238`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
  - seed1:
    - `leader_final_x = 26.708307520909788`
    - `fallback_events = 66`
    - `safety_interventions = 85`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
  - seed2:
    - `leader_final_x = 26.72258686357844`
    - `fallback_events = 71`
    - `safety_interventions = 92`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
- 结论：
  - 当前主线在 `S5` 上已经守住了安全红线
  - 但仍未形成“到达目标”的论文级闭环
  - 真正的问题已从“是否会撞/越界”转为“如何减少 fallback 与 safety 托底，同时让 leader 更有效穿过 staggered multi-blocker corridor”

### 4.4 boundary-aware preflip gate 的最终结论
- 该方向已经被完整验证，不应再继续盲调。
- 诊断结论：
  - `S1/S2` 与 clean-HEAD 完全一致
  - `S5` 真正受影响
  - `S5 seed1/2` 中，boundary gate 让 leader 在后段更容易出现 nominal accel 过激，随后被 safety 托底
- 因而最终工程决策是：
  - 回退 boundary gate
  - 保留 preflip blend 主线
  - 下一步改方向，不再围绕 boundary gate 做 guard/release band 微调

## 5. 下一步指令

### 5.1 下一个工程师启动 AI 后，应该立刻写的代码
- 第一优先文件：
  - `src/apflf/controllers/adaptive_apf.py`
- 第一优先切入点：
  - 在现有 `_leader_hazard_speed_limit(...)` 上继续做 nominal 纵向调制增强
- 建议新增 helper：
  - `_leader_staggered_hazard_activation(...)`
  - 或 `_leader_dual_blocker_speed_scale(...)`

### 5.2 要解决的真实问题
- 当前 `S5` 的主要矛盾不是 lateral target 不会提前翻，而是：
  - leader 在 staggered multi-blocker bypass 几何里，纵向 aggressiveness 仍然偏高
  - 最终表现为 fallback 与 safety interventions 偏多
  - 结果是虽然安全，但长期 `reached_goal = False`

### 5.3 必须满足的具体数学约束
- 这次修改必须仍然是 nominal 层，不是 safety 层。
- 必须满足以下约束：
  - `leader-only`
  - `hazard-only`
  - 平滑有界
  - 不引入新的硬切换
  - 不修改 `src/apflf/safety/safety_filter.py`
  - 不通过硬改 `force_x` 或直接钳死 accel 来“刷指标”
- 建议的几何量必须同时使用三类信息：
  - nominal-side blocker 的 rear gap
  - alternate-side blocker 的 rear gap
  - 当前 lateral reorientation 未完成程度
- 一个推荐的数学形式是构造三个 `activation ∈ [0, 1]`：
  - `a_nominal_gap`
    - nominal blocker 越接近，值越大
  - `a_alternate_gap`
    - alternate blocker 越进入 lookahead window，值越大
  - `a_lateral`
    - `|target_y - state.y|` 越大，值越大
- 然后构造一个平滑有界的综合量：
  - `a_staggered = a_nominal_gap * a_alternate_gap * a_lateral`
- 最终只允许用它去压低 leader hazard reference speed，而不是改 safety：
  - `speed_cap = f(base_target_speed, gap_speed, a_staggered)`
  - 必须满足：
    - 当 `a_staggered = 0` 时，输出必须退化为当前已有逻辑
    - 当 `a_staggered` 增大时，`speed_cap` 单调不增
    - `speed_cap` 必须保留下界，不能把 leader 直接钉死
    - 不能影响 follower reference speed

### 5.4 下一个工程师必须补的测试
- 文件：
  - `tests/test_adaptive_apf.py`
- 至少补两个回归：
  - 一个 `S5` 风格测试，锁住“当 nominal/alternate 双 blocker 同时进入竞争几何，且 lateral reorientation 尚未完成时，leader reference_speed 必须进一步下降”
  - 一个非目标场景保护测试，锁住“在非 staggered case 或 preflip 很早阶段，不能平白压低巡航速度”
- 现有测试必须继续通过：
  - `test_adaptive_apf_keeps_full_leader_reference_speed_before_local_flip`
  - `test_adaptive_apf_preflip_target_y_starts_shifting_before_hard_local_flip`
  - `test_adaptive_apf_leader_reference_speed_throttles_during_staggered_hazard_reorientation`
  - `test_adaptive_apf_caps_low_speed_hazard_braking_before_self_stop`

### 5.5 下一轮验收标准
- 代码验收：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 实验验收：
  - 先跑：
    - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --exp-id <new_exp_id>`
  - 然后至少补跑：
    - `S1/S2/S5 × seeds 0 1 2`
- 下一轮最低 no-regression 目标：
  - seed0:
    - `leader_final_x >= 26.674232747808983`
    - `fallback_events <= 205`
    - `safety_interventions <= 238`
  - seed1:
    - `leader_final_x >= 26.708307520909788`
    - `fallback_events <= 66`
    - `safety_interventions <= 85`
  - seed2:
    - `leader_final_x >= 26.72258686357844`
    - `fallback_events <= 71`
    - `safety_interventions <= 92`
  - 所有 seed 都必须保持：
    - `collision_count = 0`
    - `boundary_violation_count = 0`

## 6. 明确不要做的事
- 不要再继续调这版 `boundary-aware preflip gating`
- 不要碰 `src/apflf/safety/safety_filter.py`
- 不要把主精力拉回 late-stage braking cap / crawl floor 的重复微调
- 不要直接硬改 `force_x`、硬钳 accel、或靠 mode 名字硬分支“刷过” `S5`
- 不要把 RL 拉成主版本

## 7. 交接备注
- 当前最重要的新事实不是“又多了一个 patch”，而是：
  - boundary gate 这条线已经被完整试错并排除
  - 当前主线应重新聚焦到 staggered multi-blocker 几何下的 nominal 纵向调制
- 如果下一位工程师严格沿着“`adaptive_apf.py` 中 leader hazard speed governor 的几何增强 + 对应回归测试 + `S1/S2/S5 × seeds 0 1 2` 小范围验证”这条线继续走，是当前最有希望把 `S5` 从“安全但到不了”推进到论文主结果可用状态的路径。
