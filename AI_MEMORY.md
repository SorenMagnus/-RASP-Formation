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
  - `numpy`, `scipy`, `PyYAML`, `matplotlib`, `osqp`, `pytest`
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
- 每个实验输出必须至少包含：
  - `config_resolved.yaml`
  - `summary.csv`
  - `traj/*.npz`
- 每次代码修改后至少通过：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`

## 2. 当前开发游标

### 2.1 当前 HEAD 与工作树
- 当前 `HEAD`：
  - `5e16920205528afb11eb0a08e42de978f77dd9d9`
  - 提交说明：`第九次更新`
  - 提交时间：`2026-03-23 21:26:45 +0800`
- 当前工作树是干净的，没有未提交改动。
- 当前稳定代码状态主要体现在：
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/controllers/adaptive_apf.py`
  - `tests/test_adaptive_apf.py`

### 2.2 当前已验证状态
- 最近一次完整验证结果：
  - `python -m pytest -q` -> `85 passed`
  - `python -m compileall src tests scripts` -> 通过
- 由于当前工作树干净，可以把上面的验证视为当前 `HEAD` 的有效基线。

### 2.3 当前最可信的场景级结果
- 当前最可信的 `S5` 单 seed 结果在：
  - `outputs/context_sync_s5_braking_cap_v1/summary.csv`
- 关键指标：
  - `leader_final_x = 26.64380778642643`
  - `fallback_events = 205`
  - `safety_interventions = 239`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `min_obstacle_clearance = 0.5008002674527248`
- 结论：
  - 安全性守住了。
  - 效率只获得了很小的净正改进。
  - `S5` 仍未被真正打穿。

### 2.4 当前关键诊断
- `step 70-80` 已经不是 `target_y` 设计错误的问题：
  - leader 的 `target_y` 已经稳定在约 `1.499`
  - nominal steer 也已经长时间打满到左侧
- late-stage 真正主导停滞的是 relevant obstacle `dense_static_right` 的纵向负排斥：
  - 在 `step 75` 附近，leader 的 `obstacle_force` 量级大约是 `[-226, +37]`
  - 也就是说，横向引导已经到位，但纵向负排斥仍在主导闭环
- 但继续盯 `step 70-80` 做 speed floor / braking cap 微调，不是主线：
  - 当前 nominal accel 已经被 `_leader_low_speed_braking_cap()` 限成轻微负加速度
  - 再继续在 late-stage 补速度，收益很小，且安全风险高
- 真正的决定点在 `step 53-56`：
  - 在 `yield_right` 的最后阶段，`target_y` 仍停留在约 `-0.55`
  - 到 `step 56` 才突然跳到约 `1.39`
  - 这说明真正缺的不是“翻边后继续加速”，而是“翻边前的平滑预偏转几何”
- 先前已经证伪的方向：
  - 更早的硬阈值 relock / local flip
  - 直接补 speed floor / crawl floor
  - 单纯继续修 late-stage braking
- 当前最有价值的新结论：
  - 下一刀应该是 **pre-flip target geometry**，不是 safety filter，不是 late-stage speed hack。

## 3. 已完成工作

### 3.1 已进入当前 HEAD 的名义层几何机制
- `src/apflf/controllers/apf_lf.py`
  - 已加入 `_leader_nonrelevant_clearance_activation()`：
    - 对 nonrelevant obstacle 的“已局部清障”状态提供平滑 activation
  - 已加入 `_leader_side_channel_center_y()`：
    - 计算当前绕行侧通道中心线
  - 已加入 `_leader_channel_centerline_blend()`：
    - 将 leader 的 `target_y` 从障碍边缘目标平滑 blend 向通道中心线
  - 已保留 `_leader_hazard_target_x()` 的 near-preview 机制：
    - hazard 阶段不再始终直接看远端 `goal_x`
  - 已保留 `_leader_bypass_force()` 的 road compensation：
    - 在 road push 存在时，对 leader 的横向绕行引导做非对称补偿

- `src/apflf/controllers/adaptive_apf.py`
  - 已将 nonrelevant obstacle 横向削弱复用到统一的 clearance activation 上
  - 已加入 `_leader_low_speed_braking_cap()`：
    - 在 low-speed hazard near-stop 窗口中，避免 nominal 自己把 leader 刹成完全停死
  - 该 braking cap 是窄窗口、leader-only、hazard-only 的修正，当前已保留在 `HEAD` 中

### 3.2 已完成并保留下来的回归测试
- `tests/test_adaptive_apf.py`
  - `test_adaptive_apf_tapers_nonrelevant_obstacle_lateral_push_after_local_relock`
  - `test_adaptive_apf_keeps_full_leader_reference_speed_before_local_flip`
  - `test_adaptive_apf_leader_reference_speed_throttles_during_staggered_hazard_reorientation`
  - `test_adaptive_apf_caps_low_speed_hazard_braking_before_self_stop`
  - `test_adaptive_apf_overtake_guidance_overcomes_road_push_during_incomplete_lane_shift`

### 3.3 已完成并验证的实验结论
- nonrelevant obstacle shaping + channel centerline blend 之后，`S5` 的 safety interventions 已经显著低于早期失控版本。
- 当前 `HEAD` 保留下来的 braking cap 相比之前的稳定基线，带来了一个很小但净正的提升：
  - `leader_final_x` 从 `26.64118188325594` 提升到 `26.64380778642643`
  - `fallback_events` 保持 `205`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
- 已明确失败并回退的方向：
  - 更早 hard relock / hard flip
  - hazard crawl floor
  - 继续追着 late-stage 去补速度

### 3.4 本轮未落地但非常重要的诊断结论
- 本轮做了 `S5` replay 切片诊断，确认：
  - `step 70-80`：问题不是 `target_y` 不够居中
  - `step 53-56`：问题是 local flip 前缺少平滑预偏转
- 本轮没有把新的 pre-flip 代码写进仓库。
- 也就是说，当前 `HEAD` 是稳定的，但 **pre-flip 几何修正仍然没做**。

## 4. 下一步指令

### 4.1 下一位工程师启动 AI 后，应该马上写哪段代码
- 第一优先文件：
  - `src/apflf/controllers/apf_lf.py`
- 立即要写的内容：
  - 先把当前 `_leader_behavior_target_y()` 中“按指定侧计算 target_y”的逻辑抽成独立 helper
    - 建议命名：`_leader_side_target_y(...)`
    - 这个 helper 必须接受显式 `side_sign`
    - 它只负责计算“如果当前选择该侧绕行，那么 leader 的 target_y 应该是多少”
  - 然后在 `_leader_behavior_target_y()` 中加入一个新的 **smooth pre-flip target blend**
    - 仅在 `leader-only`
    - 仅在 `hazard-only`
    - 仅在 `side_sign == nominal_side_sign`，也就是“真正 local flip 还没发生”的窗口内触发
    - 触发条件必须同时利用：
      - nominal 侧 lateral commitment 已接近现有 flip threshold
      - nominal anchor 的 rear gap 已经足够小
      - alternate relevant obstacle 已经进入 lookahead window
    - 该 blend 的目标不是直接 hard flip，而是把 `target_y` 从 nominal-side target 平滑拉向 alternate-side corridor target

- 第二优先文件：
  - `tests/test_adaptive_apf.py`
- 立即要补的测试：
  - 新增一个 `step 55` 左右的 staggered-blocker 回归测试
  - 目标是锁住 pre-flip 几何修正：
    - 在 still-`yield_right` 但即将 local flip 的窗口，`target_y` 必须已经开始从 `-0.55` 往中心/alternate corridor 漂移
    - 不能等到 hard flip 发生后才突然抬升

### 4.2 下一段代码必须满足的具体数学约束
- 这是 **guidance / attraction geometry** 修正，不是 safety 修正。
- 必须满足：
  - `leader-only`
  - `hazard-only`
  - 平滑有界
  - 不能引入新的二元硬切换
  - 不能直接去改 `safety_filter.py`
  - 不能靠暴力调大 `force_x` 或猛踩油门硬冲
- 推荐的 pre-flip activation 结构：
  - 一项来自 lateral commitment 接近现有 threshold 的平滑 activation
  - 一项来自 nominal anchor rear gap 接近 flip 区间的平滑 activation
  - 两者相乘后作为 blend 系数
- blend 系数必须有上界：
  - 建议上界在 `0.5 ~ 0.6`
  - 目的是“提前预偏转”，不是“提前整段强行换边”
- 修改后的几何应满足的直觉约束：
  - `step 53-55` 类几何中，leader 的 `target_y` 不应继续死锁在约 `-0.55`
  - 它应该在 hard flip 前就开始向中心或 alternate corridor 偏移
  - `step 70-80` 的 late-stage 逻辑不应成为继续调参主战场

### 4.3 下一轮验收标准
- 先做 `S5` 单 seed 探针
- 新版本至少要满足：
  - `leader_final_x > 26.64380778642643`
  - `collision_count = 0`
  - `boundary_violation_count = 0`
  - `fallback_events <= 205`
  - `safety_interventions` 不能出现明显爆炸式回升
- 验证命令：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 实验输出仍然必须落到新的 `outputs/...` 目录，并保留：
  - `config_resolved.yaml`
  - `summary.csv`
  - `traj/seed_0000.npz`

### 4.4 明确不要做的事
- 不要碰 `src/apflf/safety/safety_filter.py`
- 不要继续把主要精力放在 late-stage braking cap / crawl floor 上
- 不要再次简单地只调 relock threshold
- 不要直接改 `force_x` 去“推着车冲过去”

## 5. 交接备注
- 当前仓库已经处于一个稳定、可交接的状态。
- 这轮最重要的新增认知不是“又多了一条补丁”，而是：
  - `S5` 的下一刀必须前移到 `step 53-56`
  - 必须通过 **pre-flip target geometry** 解决
  - 不能再靠 safety 或 late-stage speed hack 硬顶
- 下一位工程师如果严格沿着上面的 helper 抽取 + pre-flip smooth blend 去做，最有希望拿到真正能过 `S5` 的下一次净提升。
