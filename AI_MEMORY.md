# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标红线
- 目标不是做一个能跑的 demo，而是持续收敛到可投稿 IEEE 级别论文的可复现 artifact。
- 主架构必须始终保持三层闭环，禁止退化成单层启发式或黑盒端到端：
  - `Nominal Controller`
  - `Safety Filter (CBF-QP + OSQP)`
  - `Mode Decision (FSM 为主，RL 只允许作为可选离散决策扩展)`

### 1.2 技术栈红线
- 仅允许 Python 技术栈，当前仓库基线：
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
  - 用端到端黑盒控制器替换现有三层结构

### 1.3 安全层红线
- `Safety Filter` 必须保持：
  - `CBF-QP`
  - `OSQP`
  - `preview verification`
  - `fallback` 机制
- 禁止通过删除 `preview verification`、删除 `fallback`、或直接放松 exact one-step safety 约束去“刷过” `s4/s5`。
- 当前主矛盾仍然在 nominal 层几何，不在 safety 层。

### 1.4 实验与复现红线
- 必须保持 headless CLI、YAML 配置驱动、全链路 seed 可复现。
- 每个实验输出至少包含：
  - `config_resolved.yaml`
  - `summary.csv`
  - `traj/*.npz`
- 每次代码修改后至少通过：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`

## 2. 当前开发游标

### 2.1 当前 HEAD 与工作树
- 当前 `HEAD`：
  - `761648956209e43332c6c8e5ca2d0ce346bf325e`
  - 提交说明：`第十次更新`
  - 提交时间：`2026-03-23 21:50:17 +0800`
- 当前工作树不是干净的，存在 2 个未提交改动：
  - `src/apflf/controllers/apf_lf.py`
  - `tests/test_adaptive_apf.py`
- 这两个未提交改动正是本轮新增的 pre-flip 几何优化与其回归测试。
- 注意：
  - `outputs/context_sync_s5_preflip_blend_v1/summary.csv` 记录的 `git_commit` 仍是当前 `HEAD`
  - 但该结果实际来自“基于 `HEAD`、且带上述本地未提交修改”的脏工作树
  - 下一位工程师如需严格复现，必须保留这两处改动再重跑

### 2.2 当前已验证状态
- 本轮定向验证：
  - `python -m pytest tests/test_adaptive_apf.py -q` -> `26 passed in 0.90s`
- 本轮完整验证：
  - `python -m pytest -q` -> `86 passed in 208.16s`
  - `python -m compileall src tests scripts` -> 通过

### 2.3 当前最可信的场景级结果
- 当前最可信的 `S5` 单 seed 结果在：
  - `outputs/context_sync_s5_preflip_blend_v1/summary.csv`
- 关键指标：
  - `leader_final_x = 26.674232747808983`
  - `fallback_events = 205`
  - `safety_interventions = 238`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `min_ttc = 0.6468118018741192`
  - `min_boundary_margin = 0.03984901892907544`
  - `min_obstacle_clearance = 0.500995095301306`

### 2.4 相对上一稳定基线的变化
- 上一版最可信基线在：
  - `outputs/context_sync_s5_braking_cap_v1/summary.csv`
- 本轮 pre-flip 几何改动相对上一版的净变化：
  - `leader_final_x`：`26.64380778642643 -> 26.674232747808983`
  - `fallback_events`：`205 -> 205`
  - `safety_interventions`：`239 -> 238`
  - `collision_count`：保持 `0`
  - `boundary_violation_count`：保持 `0`
- 结论：
  - 这是一次真实的净正推进
  - `S5` leader 终于明显往前走了一截
  - 但 `min_ttc` 与 `min_boundary_margin` 比上一版略紧，说明 pre-flip 机制下一步要补“边界裕度感知”

### 2.5 当前核心诊断
- 本轮之前的关键诊断已经被验证：
  - `step 70-80` 不是根因，晚段 `target_y` 已经足够正确
  - 真正应该动刀的位置是 `step 53-56`
  - 问题是 hard local flip 之前缺少平滑预偏转
- 本轮落地后，结论进一步收敛为：
  - `pre-flip target geometry` 是正确方向
  - 继续做 late-stage braking cap / crawl floor 不是主线
  - 下一刀不再是“要不要做 pre-flip”，而是“怎么让 pre-flip 更边界感知、更稳”

## 3. 已完成工作

### 3.1 已在当前基线中保留的既有 nominal 机制
- `src/apflf/controllers/apf_lf.py`
  - `_leader_nonrelevant_clearance_activation()`
    - 对 nonrelevant obstacle 的局部清障状态提供平滑 activation
  - `_leader_side_channel_center_y()`
    - 计算当前绕行侧通道中心线
  - `_leader_channel_centerline_blend()`
    - 将 `target_y` 从障碍边缘目标平滑 blend 向通道中心线
  - `_leader_hazard_target_x()`
    - hazard 阶段引入 near-preview，避免 leader 永远盯远端 `goal_x`
  - `_leader_bypass_force()`
    - 对 road push 做 leader-specific 的横向补偿
- `src/apflf/controllers/adaptive_apf.py`
  - nonrelevant obstacle 横向削弱已经统一复用 clearance activation
  - `_leader_low_speed_braking_cap()` 已保留
    - 只在 low-speed hazard near-stop 窗口中避免 nominal 自己把车刹死

### 3.2 本轮新完成的核心代码工作
- 文件：
  - `src/apflf/controllers/apf_lf.py`
- 已新增 `_leader_side_target_y(...)`
  - 将“在指定绕行侧上计算 leader `target_y`”抽成独立 helper
  - 支持显式 `side_sign`
  - 支持 `apply_flip_overshoot=False`
  - 这一步是后续所有 pre-flip 几何修正的接口基础
- 已新增 `_leader_preflip_target_blend(...)`
  - 在 hard local flip 之前，提前把 `target_y` 从 nominal-side target 平滑拉向 alternate-side corridor target
  - 触发条件依赖：
    - nominal-side lateral commitment 已接近现有 flip threshold
    - nominal anchor rear gap 已接近 flip 区间
    - alternate relevant obstacle 已进入 lookahead window
  - 当前 blend 系数上界是 `0.55`
  - 该机制是：
    - `leader-only`
    - `hazard-only`
    - 平滑有界
    - 不引入新的 hard switch
- 已重构 `_leader_behavior_target_y()`
  - 改为：
    - 先调用 `_leader_side_target_y(...)`
    - 再调用 `_leader_preflip_target_blend(...)`
  - 这样当前的 pre-flip 逻辑与 post-flip 逻辑已经被正式串进主 nominal 链路

### 3.3 本轮新完成的测试工作
- 文件：
  - `tests/test_adaptive_apf.py`
- 已新增回归：
  - `test_adaptive_apf_preflip_target_y_starts_shifting_before_hard_local_flip`
- 该测试锁定的行为是：
  - 在 `step 55` 左右、仍处于 `yield_right` 且 hard flip 尚未发生时
  - leader 的 `target_y` 必须已经开始从 nominal-side target 漂向 alternate corridor
  - 不能等到 hard local flip 真的发生之后才突然抬升

### 3.4 本轮已完成的实验结论
- `S5` 单 seed pre-flip 探针已经跑通：
  - 输出目录：`outputs/context_sync_s5_preflip_blend_v1`
- 当前新版本相对上一版是净正：
  - `leader_final_x` 明显提升
  - `fallback_events` 没有增加
  - `safety_interventions` 反而小降
  - 安全指标保持零碰撞、零边界越界
- 这意味着：
  - “把 pre-flip 提前到 target geometry 层”是正确的
  - 当前改动值得保留

## 4. 下一步指令

### 4.1 下一个工程师启动 AI 后，应立即写哪段代码
- 第一优先文件：
  - `src/apflf/controllers/apf_lf.py`
- 立即要写的内容：
  - 在现有 `_leader_preflip_target_blend(...)` 上加入 **boundary-aware preflip gating**
  - 让 pre-flip blend 不仅看 commitment 与 gap，还要看“alternate corridor 会不会把 leader 过早推到边界附近”

- 第二优先文件：
  - `tests/test_adaptive_apf.py`
- 立即要补的测试：
  - 新增一个回归测试，锁住“当 alternate-side corridor 的边界剩余裕度不足时，pre-flip blend 必须被抑制或显著削弱”

### 4.2 下一段代码必须满足的具体数学约束
- 这仍然是 nominal geometry 修正，不是 safety 修正。
- 必须继续满足：
  - `leader-only`
  - `hazard-only`
  - 平滑有界
  - 不能引入新的二元硬切换
  - 不能直接修改 `safety_filter.py`
  - 不能通过硬改 `force_x` 猛踩油门

- 推荐加入的新约束项：
  - 在 `_leader_preflip_target_blend(...)` 中引入第三个 activation：
    - `boundary_activation`
  - 该量应由 alternate-side target 或 alternate corridor 对应的边界剩余裕度构造
  - 建议使用平滑有界形式，而不是硬阈值开关

- 一个可接受的实现方向：
  - 先计算 `alternate_target_y`
  - 再计算该目标对应的道路边界剩余裕度：
    - `boundary_margin_alt = road_half_width - abs(alternate_target_y - lane_center_y) - 0.5 * vehicle_width`
  - 构造一个平滑 activation：
    - 当 `boundary_margin_alt` 小于安全 guard band 时，`boundary_activation -> 0`
    - 当 `boundary_margin_alt` 足够宽松时，`boundary_activation -> 1`
  - 最终 blend 系数改为：
    - `blend = commitment_activation * gap_activation * boundary_activation`
    - 仍然保留上界，不超过当前量级 `0.55`

- 必须保持的行为约束：
  - 在 `step 53-56` 的 S5 几何里，pre-flip 仍要比当前 nominal-side target 更早向 alternate corridor 偏移
  - 但不允许为了换取更大的 `leader_final_x` 而继续明显压缩 `min_boundary_margin`
  - 目标是“保住当前推进，同时把边界裕度拉回一点”

### 4.3 下一轮验收标准
- 先跑 `S5` 单 seed 探针
- 新版本至少应满足：
  - `leader_final_x >= 26.674232747808983`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `fallback_events <= 205`
  - `safety_interventions <= 238`
  - `min_boundary_margin` 不低于当前值，最好比 `0.03984901892907544` 更宽松

- 然后再做一个小范围鲁棒性检查：
  - 推荐先跑 `S1/S2/S5`
  - 每个场景至少 `seeds = 0 1 2`
  - 先看 pre-flip 机制有没有把别的场景带坏，再决定是否扩到完整论文矩阵

- 验证命令：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
  - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 --exp-id <new_exp_id>`

### 4.4 明确不要做的事
- 不要碰 `src/apflf/safety/safety_filter.py`
- 不要把主精力重新拉回 late-stage braking cap / crawl floor
- 不要再次简单地只调 relock threshold
- 不要直接硬改 `force_x` 去“推着车冲过去”

## 5. 交接备注

- 当前仓库的最好理解方式是：
  - 既有 nominal 主线已经从“只会 late-stage 微修”推进到了“会在 hard flip 前做几何预偏转”
  - 这一步是本轮最重要的真正进展

- 当前最重要的新事实不是“又多了一个 patch”，而是：
  - `pre-flip target geometry` 已经落地
  - 它在 `S5` 上已经拿到了真实净提升
  - 下一步应该从“是否做 pre-flip”切换到“如何让 pre-flip 更边界感知、更鲁棒”

- 如果下一位工程师严格沿着“boundary-aware preflip gating + 对应回归测试 + 小规模多 seed 验证”这条线继续走，最有希望在不破坏安全红线的前提下，把当前 `S5` 的改善变成稳定、可扩展、可写进论文主结果的提升。
