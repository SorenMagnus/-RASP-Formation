# AI_MEMORY - 当前周期交接文档

> 致下一位 AI / 工程师：
> 请先完整读完本文件，再动手改代码。
> 当前项目不是脚手架阶段，而是论文原型后期收敛阶段。
> 当前唯一真正卡住论文主结果的瓶颈，仍然是 `S5 dense_multi_agent` 中 leader 无法穿过第一组 staggered chicane。

---

## 0. 强制阅读顺序
1. `AI_MEMORY.md`（本文件）
2. `PROMPT_SYSTEM.md`
3. `RESEARCH_GOAL.md`
4. `src/apflf/controllers/adaptive_apf.py`
5. `src/apflf/controllers/apf_lf.py`
6. `src/apflf/decision/fsm_mode.py`
7. `tests/test_adaptive_apf.py`
8. `tests/test_modes.py`

---

## 1. 技术栈红线

### 1.1 架构红线
- 必须保持三层闭环：Nominal -> Safety Filter -> Mode Decision。
- 禁止修改：
  - `src/apflf/safety/safety_filter.py`
  - `src/apflf/safety/cbf.py`
  - `src/apflf/safety/qp_solver.py`
- 禁止把问题改写成 ROS / CUDA / 端到端黑盒 / RL 主方案。
- 禁止为了过 S5 而写场景硬编码分支。

### 1.2 工程红线
- Python 技术栈保持不变：Python 3.10+, numpy, scipy, PyYAML, matplotlib, osqp, pytest。
- 每次改动后必须运行：
  - `python -m pytest -q`
  - `python -m compileall src tests scripts`
- 所有实验必须可复现，seed 不能漂。

### 1.3 编码红线
- 禁止直接硬改 `accel` 或硬钳 `force_x` 做 S5 特判。
- 允许做的是：几何目标、参考速度、平滑有界的 nominal shaping。
- 所有新增函数必须有 docstring。
- 所有新激活函数必须满足平滑、有界、可退化回原逻辑。
- 当新逻辑“未激活”时，必须精确退化为旧逻辑，而不是近似退化。

---

## 2. 当前开发游标

### 2.1 当前分支状态
- 当前 `HEAD`：`0c48fde7488771e5cd92a1b32ad72e9960434d8e`
- 当前工作区有未提交改动，集中在：
  - `src/apflf/controllers/adaptive_apf.py`
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/decision/fsm_mode.py`
  - `tests/test_adaptive_apf.py`
  - `tests/test_modes.py`

### 2.2 当前稳定开发游标
- 当前稳定保留的最新思路不是“更早翻边”，也不是“更强 x-force 衰减”。
- 当前保留下来的稳定改动是：
  - `post-relock target_x hold`
  - `post-relock target_y edge-hold`
  - 以及此前已经通过回归的 staggered governor / lateral boost / steer bias / FSM relock 优化
- 当前**不要**继续沿着“直接额外衰减 obstacle force_x”这条线走；这条线已经试过，完整 seed0 会把 `fallback_events` 从 `203` 拉高到 `342`，副作用过大，已判定为坏方向。

### 2.3 当前稳定验证状态
- `python -m pytest -q` -> **99 passed**
- `python -m compileall src tests scripts` -> **通过**

---

## 3. 已完成工作

### 3.1 `adaptive_apf.py` 已完成
以下改动都已经保留在当前工作区，并通过回归：

- 增加平滑激活基础设施：
  - `_smoothstep01`
  - `_rising_activation`
  - `_falling_activation`
- 增加 leader hazard 诊断：
  - `_leader_nearest_rear_gap`
  - `_leader_staggered_hazard_activation`
- 增加 staggered dual-blocker 纵向 shaping：
  - `_shape_leader_staggered_obstacle_force`
  - `_leader_staggered_longitudinal_relief`
- 增强 leader hazard 速度治理：
  - `_leader_hazard_speed_limit` 现在显式考虑 staggered activation
  - `_leader_staggered_hazard_speed_cap` 已采用更激进但仍有界的参数
  - near-stop 状态下加了 `state.speed <= 0.35` 保护，不再盲目继续压速
- 增加横向 nominal 杠杆：
  - `_leader_staggered_lateral_boost`
  - `_leader_staggered_steer_bias`
- `compute_actions()` 已集成：
  - staggered lateral boost
  - leader steer bias
  - 低速 hazard braking cap

### 3.2 `apf_lf.py` 已完成
以下几何逻辑已经落地并保留：

- 局部翻边阈值从原始版本下调到：
  - `commitment_threshold = max(0.33 * vehicle_width, 0.17 * half_width)`
- 增加 relock 后的局部 `target_x` 保持：
  - `_leader_relocked_target_x_hold`
- 增加 relock 后的 edge-hold 几何激活：
  - `_leader_relocked_edge_hold_activation`
- 增加 relock 后 `target_y` 的 centerline blend 抑制：
  - `_leader_relocked_centerline_blend_hold`
- 当前保留的几何结论：
  - 在 `yield_left` 且 dual-blocker 仍存在时，`target_y` 不应过早抬向 channel centerline；
  - 应先沿 `dense_static_right` 的安全边缘线附近走，即 `target_y ≈ 0.45`。

### 3.3 `fsm_mode.py` 已完成
- 局部 relock commitment threshold 已同步到：
  - `max(0.33 * vehicle_width, 0.17 * half_width)`
- 增加 hazard-side relock 快速通过机制：
  - `_is_hazard_side_relock`
- `_apply_hysteresis()` 现在允许 hazard side relock 绕过通用 hysteresis。

### 3.4 测试已完成
- `tests/test_adaptive_apf.py`
  - 补了 staggered activation、staggered longitudinal relief、lateral boost、steer bias、preflip dual-blocker throttling 等测试
  - 低速 braking cap 断言已对齐当前浮点行为
- `tests/test_modes.py`
  - 补了 FSM hazard relock 测试
  - 补了 APF-LF local flip / relocked target_x / relocked target_y hold 测试
- 当前全量测试数：**99 passed**

---

## 4. 本周期关键实验结论

### 4.1 当前保留的稳定实验结果
保留版本对应实验：
- `outputs/s5_seed0_targety_hold_v1/summary.csv`

结果：
- `leader_final_x = 26.630563706474664`
- `fallback_events = 203`
- `safety_interventions = 229`
- `reached_goal = False`

和上一版相比：
- `leader_final_x` 基本没破平台
- `fallback_events` 没变
- 但 `safety_interventions` 从 `253` 降到了 `229`

这说明：
- `target_y edge-hold` 让 nominal 与 safety 的耦合更顺了
- 但还没有把主结果打穿

### 4.2 已证实的正确信号
最重要的新结论：

- 在旧版本中，step 59 以后 nominal preview margin 对 `dense_static_right` 为负，safety 需要持续纠正。
- 在当前保留的 `target_y hold` 几何下，局部回放表明：
  - step 56-65 期间 leader 经常达到 nominal = safe
  - 说明问题已不再主要是“safety 反复否决 nominal”
  - 而转移成了“nominal 自己过于保守，把自己刹停”

这一步非常关键。
它把问题从“几何不可行”推进成了“几何已基本可行，但 governor 释放不足”。

### 4.3 已证伪的坏方向
以下分支已经试过，结论是不要保留：

1. 更早 local flip / 更早 FSM relock
- 实验：`outputs/s5_seed0_early_relock_v1/summary.csv`
- 结果：
  - `leader_final_x = 26.618069427623887`
  - `fallback_events = 215`
- 结论：
  - 提前两拍翻边会把系统带进另一套停滞盆地，方向错误

2. 额外 `x-force` 衰减分支
- 实验：`outputs/s5_seed0_targety_xrelief_v1/summary.csv`
- 结果：
  - `leader_final_x = 26.64872172142302`
  - `fallback_events = 342`
- 结论：
  - 虽然 `leader_final_x` 略升，但 safety/fallback 指标严重恶化，不能留

---

## 5. 当前真实瓶颈

当前稳定代码下，S5 的主要瓶颈已经非常明确：

- `post-relock target_y` 过早抬向 centerline 这个问题，已经被部分修正。
- 当前剩余核心瓶颈不是 safety filter 本身。
- 当前剩余核心瓶颈是：
  - leader 在 edge-hold 几何已经成立后，
  - nominal 的速度治理仍然过于保守，
  - reference speed / crawl floor 释放太慢，
  - 导致 leader 在 `x ≈ 26.6` 左右再次自停。

一句话：
**下一步该做的是 governor release，不是继续做 obstacle-force shaping。**

---

## 6. 下一步指令

### 6.1 下一个工程师启动后，第一件事要写的代码
请直接修改：
- `src/apflf/controllers/adaptive_apf.py`

目标：
- 新增一个 **leader-only / relock-edge-hold-only / smooth-bounded** 的
  `post-relock governor release` 辅助函数，
- 并在 `_leader_staggered_hazard_speed_cap()` 内调用它，
- 让 leader 在 edge-hold 已建立时，不再把参考速度继续压到“自停”。

### 6.2 必须满足的数学约束
设：
- `a_hold = self._leader_relocked_edge_hold_activation(...)`
- `staggered_cap` 为当前 `_leader_staggered_hazard_speed_cap()` 已算出的旧输出
- `base_target_speed` 为进入 staggered governor 之前的输入

要求你实现的新函数满足：

1. 零激活精确退化
- 若 `a_hold <= 1e-6`，新逻辑必须**精确返回旧的** `staggered_cap`
- 不能是“接近旧值”，必须是完全一样

2. 平滑有界释放
- 定义一个平滑有界的 floor，例如：
  - `release_floor = clip(0.85 + 0.35 * a_hold, 0.85, 1.20)`
- 最终输出示例形式：
  - `released_cap = min(base_target_speed, max(staggered_cap, release_floor))`

3. 单调性
- 当 `a_hold` 增大时，release 只能单调增强，不能出现反向压速

4. 有界性
- 释放后速度上界不得超过 `base_target_speed`
- release floor 必须保持在一个小速度区间内，建议 `[0.85, 1.20]`
- 禁止重新放回 cruise speed

5. 作用域约束
- 只对 leader 生效
- 只在 dual-blocker + relocked edge-hold 几何下生效
- 不得改 safety 层
- 不得直接写死 `accel`
- 不得再做额外的 obstacle force x 特判

### 6.3 代码落点建议
建议新增：
- `_leader_relocked_edge_speed_release(...)`

建议调用位置：
- `_leader_staggered_hazard_speed_cap()` 最后得到 `staggered_cap` 后、`return` 前

建议复用已有几何激活：
- `self._leader_relocked_edge_hold_activation(...)`

### 6.4 你必须马上补的测试
在 `tests/test_adaptive_apf.py` 增加两类测试：

1. 正向激活测试
- dual-blocker
- `yield_left`
- step 60 左右的 relocked edge-hold slice
- 断言：
  - 新 cap > 旧 `staggered_cap`
  - 新 cap <= `base_target_speed`
  - 新 cap <= 1.20

2. 非激活退化测试
- single-blocker 或 `a_hold == 0`
- 断言新 cap == 旧 `staggered_cap`

### 6.5 下一步验收门槛
最小验收：
- `python -m pytest -q`
- `python -m compileall src tests scripts`
- 重新跑：
  - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --exp-id <new_id>`

阶段性成功判据：
- `seed0` 的 `leader_final_x > 26.630563706474664`
- `seed0` 的 `fallback_events <= 203`
- 不出现 collision / boundary violation 回归

更高目标：
- 任一 seed 达到 `leader_final_x > 35.0`

终局目标仍然是：
- `reached_goal = True`

---

## 7. 不要做的事
- 不要改 `safety_filter.py / cbf.py / qp_solver.py`
- 不要继续保留或恢复“额外 x-force 衰减”分支
- 不要再走“更早 local flip / 更早 FSM relock”这条线
- 不要直接硬改 `accel`
- 不要写基于 mode 名字的场景特判
- 不要降低 `safe_distance`
- 不要放松 `barrier_decay`
- 不要引入新依赖
- 不要新开场景，集中火力只打 S5

---

## 8. 关键文件索引
- `src/apflf/controllers/adaptive_apf.py`
  - 当前 nominal speed governor、lateral boost、steer bias 主文件
- `src/apflf/controllers/apf_lf.py`
  - 当前 relocked target_x hold / target_y edge-hold 主文件
- `src/apflf/decision/fsm_mode.py`
  - 当前 hazard-side relock hysteresis 优化
- `tests/test_adaptive_apf.py`
  - 当前 adaptive nominal 回归保护
- `tests/test_modes.py`
  - 当前 APF-LF / FSM 几何回归保护

---

## 9. 一句话交接
当前代码已经把 S5 问题从“safety 否决 nominal”推进到了“nominal governor 自己过于保守”。
下一位工程师不要再碰 safety，也不要再碰 obstacle-force 特判；请直接在 `adaptive_apf.py` 里实现 **基于 relocked edge-hold activation 的 smooth governor release**。
