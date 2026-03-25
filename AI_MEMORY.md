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
- 禁止通过删除 `preview verification`、削弱 `fallback`、放松 one-step safety 约束来"刷过"场景。
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
  - commit: `1353311cd519ace65f00408e1c297a264325c6ac`
  - 时间: `2026-03-24 23:42:13 +08:00`
  - 提交说明: `第十二次更新`
- 当前工作树相对 `HEAD` 有三个修改文件：
  - `AI_MEMORY.md`（本文件）
  - `src/apflf/controllers/adaptive_apf.py`（+134 行：新增 `_leader_staggered_hazard_speed_cap` 方法 + 修改 `_reference_speed` 流水线）
  - `tests/test_adaptive_apf.py`（+74 行：新增 2 个 staggered governor 回归测试）
- 也就是说：
  - 本轮新增了 staggered dual-blocker speed governor，代码尚未提交
  - 所有修改仅限 nominal 层 + 测试，未触碰 safety 层

### 2.2 当前代码主线
- 当前 nominal 主线已经包含并保留：
  - `src/apflf/controllers/apf_lf.py`
    - `_leader_side_target_y(...)`
    - `_leader_hazard_target_x(...)`
    - `_leader_preflip_target_blend(...)`
  - 其中 `preflip_blend` 当前稳定版本仍是：
    - `blend = commitment_activation * gap_activation`
    - 上界 `0.55`
- **本轮新增** — `src/apflf/controllers/adaptive_apf.py`：
  - `_leader_staggered_hazard_speed_cap(...)` — staggered dual-blocker 纵向调制
    - 构造三路激活：`a_nominal_gap × a_alternate_gap × a_lateral`
    - gap-speed blend：`blend = clip(a_staggered * 4.5, 0, 0.65)`
    - 保留 crawl floor = 0.45
    - 横向 dead-zone = `0.40 * vehicle_width`
  - `_reference_speed(...)` 修改为链式调用：`super() → _leader_hazard_speed_limit → _leader_staggered_hazard_speed_cap`
- 当前 adaptive nominal 还包含（上轮已有）：
  - `_leader_hazard_speed_limit(...)`
  - `_leader_low_speed_braking_cap(...)`
- 当前安全层主线：
  - `src/apflf/safety/safety_filter.py`（未修改）
- 当前模式决策主线：
  - `src/apflf/decision/fsm_mode.py`（未修改）

### 2.3 当前验证状态
- 最近一次完整验证：
  - `python -m pytest -q` → `88 passed in 85.55s`
  - `python -m compileall src tests scripts` → 通过

### 2.4 当前最可信参考输出
- 本轮 staggered governor 实验输出：
  - `outputs/staggered_governor_s5/summary.csv`
  - `outputs/staggered_governor_s1/summary.csv`
  - `outputs/staggered_governor_s2_v2/summary.csv`

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
- leader hazard 几何已经不再只是"贴边走"：
  - 已显式区分 passing side
  - 已引入 corridor centerline blend
  - 已引入 flip overshoot
  - 已引入 hazard-local target_x
  - 已引入 pre-flip target_y 提前偏转
- **本轮新增**：staggered dual-blocker speed governor
  - 联合感知 nominal-side / alternate-side / lateral 三路信息
  - 已集成到 `_reference_speed` 流水线
  - 在 S5 seed0 上实现了 fallback 201（vs baseline 205）、safety 232（vs 238）
  - S1/S2 无回归

### 3.3 safety 与 decision 已完成工作
- `CBF-QP + preview verification + fallback` 已工作，并有对应测试。
- `FSM + hysteresis + recover` 已工作。
- 当前主版本依赖 FSM，不依赖 RL。

### 3.4 测试与工程质量
- 当前完整测试通过数为 `88`。
- `tests/test_adaptive_apf.py` 已包含与当前 nominal 主线一致的关键回归：
  - `test_adaptive_apf_keeps_full_leader_reference_speed_before_local_flip`
  - `test_adaptive_apf_preflip_target_y_starts_shifting_before_hard_local_flip`
  - `test_adaptive_apf_leader_reference_speed_throttles_during_staggered_hazard_reorientation`
  - `test_adaptive_apf_caps_low_speed_hazard_braking_before_self_stop`
  - **本轮新增**：
    - `test_adaptive_apf_staggered_dual_blocker_further_throttles_leader_speed`
    - `test_adaptive_apf_staggered_governor_inactive_without_dual_blockers`

### 3.5 本轮额外完成的实验工作
- 已完成 `S1/S2/S5 × seeds 0 1 2` 小范围 sweep（staggered governor 版本）。
- 历史遗留：上一轮 boundary-aware preflip gate 已被完整回退，不在当前代码中。

## 4. 当前最可信实验事实

### 4.1 S1 local minima（staggered governor 版本）
- 与 clean-HEAD 基线完全一致，staggered governor 在 S1 单侧障碍几何中不激活：
  - seed0: `leader_final_x = 111.316`, `reached_goal = True`, `team_goal_reached = True`, `boundary_violation_count = 0`
  - seed1: `leader_final_x = 126.914`, `reached_goal = True`, `team_goal_reached = False`, `boundary_violation_count = 1`
  - seed2: `leader_final_x = 118.115`, `reached_goal = True`, `team_goal_reached = False`, `boundary_violation_count = 0`

### 4.2 S2 dynamic crossing（staggered governor 版本）
- 三个 seed 全部满足：
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `reached_goal = True`
  - `team_goal_reached = True`

### 4.3 S5 dense multi-agent（staggered governor 版本）
- S5 仍是论文主结果最关键也最未完成的难场景。
- 当前 staggered governor 版本实验结果：
  - seed0:
    - `leader_final_x = 26.670`
    - `fallback_events = 201`（baseline 205，**改善 -4**）
    - `safety_interventions = 232`（baseline 238，**改善 -6**）
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
  - seed1:
    - `leader_final_x = 26.709`
    - `fallback_events = 66`（baseline 66，持平）
    - `safety_interventions = 84`（baseline 85，**改善 -1**）
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
  - seed2:
    - `leader_final_x = 26.723`
    - `fallback_events = 71`（baseline 71，持平）
    - `safety_interventions = 91`（baseline 92，**改善 -1**）
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `reached_goal = False`
- 结论：
  - staggered governor 在 S5 上实现了方向正确但幅度有限的改善
  - 安全红线全部守住（零碰撞、零越界）
  - `reached_goal` 仍然为 False，`leader_final_x ≈ 26.7`
  - 真正的 blocker 已从"是否会撞/越界"转为"如何让 leader 在 staggered multi-blocker corridor 中持续推进而不被 safety filter 反复托底"

### 4.4 staggered governor 的效果分析
- staggered governor 的三路激活 `a_staggered = a_nominal_gap × a_alternate_gap × a_lateral` 在 S5 staggered 几何中确实激活了（seed0 fallback 从 205 降到 201）
- 但三路乘积天然偏保守（三个 [0,1] 相乘后绝对值偏小），导致 blend 幅度有限
- 在 S1/S2 单侧或无交叉几何中，staggered governor 完全静默（dead-zone guard + alternate_relevant 为空 → 直接返回 base），验证了最小干预性

## 5. 下一步指令

### 5.1 下一个工程师启动 AI 后，应该立刻写的代码
- 第一优先文件：
  - `src/apflf/controllers/adaptive_apf.py`
- 当前 staggered governor 的横向 dead-zone（`0.40 * vehicle_width = 0.76`）和 blend 放大系数（`4.5`）的组合需要进一步调优，但以下三条路线应该被同时评估（选择最有效的一条推进）：

#### 路线 A：加强 staggered governor 参数
- 切入点：`_leader_staggered_hazard_speed_cap` 中的 `blend = clip(a_staggered * 4.5, 0, 0.65)`
- 具体尝试：
  - 降低 `lateral_dead_zone` 从 `0.40` 到 `0.25` 倍 `vehicle_width`
  - 提高 blend 放大系数从 `4.5` 到 `6.0` 或 `8.0`
  - 降低 `crawl_floor` 从 `0.45` 到 `0.30`
- 风险：可能导致 leader 在某些 seed 中过度减速而无法推进
- 约束：必须保持 `S1/S2` 无回归

#### 路线 B：横向调制增强——在 staggered 几何中加速横向过渡
- 切入点：`_leader_bypass_force` 或新增 `_leader_staggered_lateral_boost`
- 核心思路：当 staggered geometry 被检测到时，增大 `guidance_gain` 以加速横向偏转，缩短 leader 在 hazard 暴露窗口内的时间
- 具体数学形式建议：
  - 构造 `staggered_lateral_boost = 1.0 + boost_gain * a_staggered`
  - 将 `guidance_gain` 乘以 `staggered_lateral_boost`
  - `boost_gain ∈ [0.5, 2.0]`，需实验确定
- 约束：
  - boost 必须有界（不能让 guidance 力超过 road_gain）
  - 只在 leader (index=0) 且 hazard 模式下生效
  - 不修改 safety_filter.py

#### 路线 C：mode decision 层配合——FSM 在 staggered 检测时主动切换
- 切入点：`src/apflf/decision/fsm_mode.py` 的 `_candidate_mode`
- 核心思路：当 FSM 检测到 staggered dual-blocker 几何时，从 `yield` 切换到一个新的 `staggered_yield` 行为，该行为映射到更保守的 gain profile
- 这条路线改动最大，但也最有可能产生质变效果
- 约束：不引入新的硬切换，必须通过 hysteresis

### 5.2 要解决的真实问题
- 当前 S5 的主要矛盾是：
  - leader 在 staggered multi-blocker corridor（`dense_static_left` @ (30, 2.0) 与 `dense_static_right` @ (32, -2.1)）中，即使有了 staggered governor，fallback 仍然频繁（seed0: 201 次）
  - leader 被反复"刹停 → 微动 → 再刹停"循环困住
  - 需要让 leader 更积极地完成横向过渡，然后以合理速度直行穿过
- S5 的物理瓶颈是：两个 blocker 之间只有约 4m 纵向间距（x=30 vs x=32），横向间距约 4.1m（y=2.0 vs y=-2.1），形成一个 staggered chicane，leader 必须做一个 S 形机动才能穿过

### 5.3 必须满足的具体数学约束
- 与上轮一致的不变约束：
  - `leader-only`
  - `hazard-only`
  - 平滑有界
  - 不引入新的硬切换
  - 不修改 `src/apflf/safety/safety_filter.py`
  - 不通过硬改 `force_x` 或直接钳死 accel 来"刷指标"
- 新增约束：
  - 如走路线 A（参数调优），必须验证降低 dead-zone 不导致 S1/S2 速度回归
  - 如走路线 B（横向 boost），boost 必须有界且在非 staggered 场景退化为 1.0
  - 如走路线 C（FSM 增强），必须保持 hysteresis 和不影响其他模式转换

### 5.4 下一个工程师必须补的测试
- 文件：
  - `tests/test_adaptive_apf.py`
- 如走路线 A/B：
  - 修改现有 `test_adaptive_apf_staggered_dual_blocker_further_throttles_leader_speed` 的阈值（如果参数变化导致速度更低）
  - 补一个 "staggered lateral boost 有界" 测试（如走路线 B）
- 如走路线 C：
  - 在 `tests/test_modes.py` 中补一个 staggered mode transition 测试
- 现有测试必须继续通过（88 个）

### 5.5 下一轮验收标准
- 代码验收：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 实验验收：
  - 先跑：
    - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --exp-id <new_exp_id>`
  - 然后至少补跑：
    - `S1/S2/S5 × seeds 0 1 2`
- 下一轮最低 no-regression 目标（基于本轮 staggered governor 基线）：
  - seed0:
    - `leader_final_x >= 26.670`
    - `fallback_events <= 201`
    - `safety_interventions <= 232`
  - seed1:
    - `leader_final_x >= 26.709`
    - `fallback_events <= 66`
    - `safety_interventions <= 84`
  - seed2:
    - `leader_final_x >= 26.723`
    - `fallback_events <= 71`
    - `safety_interventions <= 91`
  - 所有 seed 都必须保持：
    - `collision_count = 0`
    - `boundary_violation_count = 0`
- 质变目标（论文可用）：
  - 至少 1 个 seed 的 `leader_final_x > 35.0`（穿过第一组 staggered blockers）
  - 最终目标：`reached_goal = True`

## 6. 明确不要做的事
- 不要再继续调上上轮的 `boundary-aware preflip gating`（已被完整试错并排除）
- 不要碰 `src/apflf/safety/safety_filter.py`
- 不要把主精力拉回 late-stage braking cap / crawl floor 的重复微调
- 不要直接硬改 `force_x`、硬钳 accel、或靠 mode 名字硬分支"刷过" S5
- 不要把 RL 拉成主版本
- 不要盲目降低 staggered governor 的 dead-zone / crawl_floor 而不做 S1/S2 回归验证

## 7. 交接备注
- 本轮最重要的新事实：
  - staggered dual-blocker speed governor 已实现并通过测试（88 passed）
  - 其效果方向正确但幅度有限（seed0 fallback 205→201）
  - 三路乘积 `a_nominal_gap * a_alternate_gap * a_lateral` 的天然保守性是幅度有限的主因
  - S1/S2 完全无回归，验证了最小干预性
- 当前最有希望的方向判断：
  - 路线 B（横向调制增强）或路线 A+B 混合是最轻量且最可能有效的下一步
  - 路线 C（FSM 增强）是最有可能产生质变的方向，但改动量最大
  - 纯参数调优（路线 A）天花板较低，不建议作为唯一方向
- 如果下一位工程师严格沿着"在 staggered 几何中同时加强纵向减速和横向过渡速度 + 对应回归测试 + S1/S2/S5 × seeds 0 1 2 小范围验证"这条线继续走，是当前最有希望把 S5 从"安全但到不了"推进到论文主结果可用状态的路径。
