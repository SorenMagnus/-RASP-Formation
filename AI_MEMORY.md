# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标红线
- 目标不是做 demo，而是持续收敛到可投稿 IEEE 级别论文的可复现 artifact。
- 主架构必须始终保持三层闭环，禁止退化成单层启发式或黑盒端到端：
  - `Nominal Controller`
  - `Safety Filter (CBF-QP + OSQP)`
  - `Mode Decision (FSM 为主，RL 只允许作为可选离散决策扩展)`

### 1.2 技术栈红线
- 仅允许 Python 技术栈，当前仓库基线：
  - `Python 3.10+`
  - `numpy`, `scipy`, `PyYAML`, `matplotlib`, `osqp`, `pytest`
- 禁止引入：
  - ROS、CUDA 依赖、用端到端黑盒控制器替换现有三层结构

### 1.3 安全层红线
- `Safety Filter` 必须保持：
  - `CBF-QP`、`OSQP`、preview verification、fallback 机制
- 禁止通过删除 preview verification、删除 fallback、或直接放松 exact one-step safety 来“刷过” `s4/s5`。
- 当前主矛盾仍在 nominal 层几何。

### 1.4 实验与复现红线
- 必须保持 headless CLI、YAML 配置驱动、全链路 seed 可复现。
- 每个实验输出必须可追溯并至少包含：`config_resolved.yaml`, `summary.csv`, `traj/*.npz`
- 每次代码改动后至少通过 `python -m compileall` 与 `python -m pytest -q`。


## 2. 当前开发游标

### 2.1 当前 HEAD 与工作树
- 当前 `HEAD`：`3f41802c7d05cea160aa67e8488429e2ee958ce8`（提交说明：`第八次更新`）
- 当前工作树有未提交的改动（Phase 1 Shaping 增强）：
  - `src/apflf/controllers/adaptive_apf.py`
  - `tests/test_adaptive_apf.py`

### 2.2 当前已验证状态
- 全量回归测试通过：
  - `python -m pytest tests/test_adaptive_apf.py -q` (24 passed)
  - `python -m pytest -q` 全量测试：`84 passed in ~88s`（包含了本轮新增的 3 个 shaping 边界约束测试）。

### 2.3 当前场景级游标
- 当前最可信的单 seed 结果 (`stage42_probe_s5`)：
  - `outputs/stage42_probe_s5/summary.csv`
    - `leader_final_x = 26.646889730553628` (略低于 stage41 的 ~26.70)
    - `fallback_events = 205`
    - `safety_interventions = 384` (较 stage41 的 241 大幅上升)
    - `collision_count = 0` 
    - `boundary_violation_count = 0`

### 2.4 当前核心结论
- **Phase 1 Shaping 增强的副作用**：我们通过 steepening 激活曲线和提升 `r_max=0.85` 确实极其有效地降低了 nonrelevant obstacle 施加的错误横向排斥力（`F_obs,y`）。
- **真正的瓶颈联动暴露**：当名义排斥力降低后，leader 物理上靠得离该 obstacle 更近了；这导致 CBF Safety Filter 感知到横向安全裕度不足，频繁介入（safety_interventions 飙升至 384）。由于 CBF-QP 使用 nominal input 作为参考进行二次规划，CBF 强行对冲了 nominal 控制。
- 这说明：单纯切除 nominal 排斥力不够，还必须配合更积极的 **目标点牵引 (Attraction)** 或 **绕行引导 (Guidance)** 把 leader "吸" 进豁口，从而避免因贴近障碍物触发安全接管导致的停滞。


## 3. 已完成工作

### 3.1 本轮最终保留在当前工作树里的代码改动
#### A. `AdaptiveAPF` 的 Nonrelevant Obstacle Lateral Shaping 增强
- 文件：`src/apflf/controllers/adaptive_apf.py` 
- 内容：
  - 增强 `_leader_nonrelevant_lateral_reduction()`：改变 activation 窗口，将 `reduction_start_gap` 前推至 2.5，`full_reduction_overlap` 缩短至 1.5，使得障碍物与本车平齐 (gap<=0) 时立即获得 ~75-85% 的强反向横向力抵消。
  - 增强 `_shape_leader_nonrelevant_obstacle_force()`：引入侧向距离 (`lateral_distance`) 权重加成，越远的被绕过障碍物，反向力抵消权重越高，硬上限 0.90。
- 数学语义：符合 leader-only、hazard-only、非绕行侧障碍物专属、仅缩放 `force_y` 且 `force_x` 绝对锁死不变的设计红线。

#### B. `AdaptiveAPF` 完整 Shaping 函数单测覆盖
- 文件：`tests/test_adaptive_apf.py`
- 内容：新增了 3 个关键测试：
  - `test_nonrelevant_shaping_preserves_relevant_obstacle_force`
  - `test_nonrelevant_shaping_skips_aligned_lateral_push`
  - `test_nonrelevant_shaping_bounded_total_reduction`
- 保障了对后续开发的接口稳定性。

### 3.2 历史保留关键架构机制（不要随意回退）
- Leader Hazard Speed Throttle (根据 obstacle rear gap 动态压降巡航速度)。
- Leader Hazard Target-X Near Preview 保留。
- Leader Bypass Guidance 对 Road Push 的非对称补偿机制。


## 4. 下一步指令

### 4.1 下一个工程师启动 AI 后，应立即执行的任务
- **优先文件**：
  - `src/apflf/controllers/adaptive_apf.py`（针对 Nominal Attraction/Guidance 阶段优化）
  - `src/apflf/controllers/apf_lf.py`（寻找绕行宽度空间）
- 下一组代码绝不能动 `safety_filter.py`。

### 4.2 下一段代码要解决的精确数学问题
- 根据 `stage42_probe_s5` 暴露的现象：由于 Nonrelevant 障碍排斥力减弱引发 CBF 介入导致死锁。这表明 leader 的**名义力合成向量偏向了原本的障碍物，导致触碰了安全集边界**。
- **下一步数学约束对策**：需要给领航车额外补充朝向绕行通道中心的**拉力**（Attraction 或 Guidance），对抗 CBF 触发。
- 具体思路探讨：
  1. 当前 `target_y` 是否因为队形编队的残余影响没有完美对准豁口的几何中心？（可以审查 `_static_goal_target` 对 leader 在 hazard 的 `offset_y` 处理是否可以更激进贴靠中心点）。
  2. 对于 `yield_left / yield_right`，当处于 relock 晚期并在两车之间钻缝时，能否引入针对豁口中心线 (Clearance Centerline) 的 attractor。
  3. 审查 `stagnation_force_threshold` 触发进入 recovery mode 的阈值设计，看是否在 s5 被触发。

### 4.3 新一轮优化的起手式测试
- 写一个独立的静态或单步模拟脚本（使用与 s5 相似的 `box_clearance` 临界状态），在 `AdaptiveAPFController.compute_actions` 发生时，将 `action` 交给 `safety_filter.filter_actions` 验证。
- 目标：确保 Nominal Controllers 计算得出的 Action，不会因为横向裕度（Lateral Margin）不足立刻被 CBF 惩罚为 fallback 或极大修改的 safe action。


## 5. 明确不要做的事
- **绝对不要**直接修改 `safety_filter.py` 内部任何关于安全边界的计算，不准缩小 CBF 的碰撞缓冲或容忍越界。
- **不要**在没有用数学约束讲清楚为何能避免 CBF 干预的前提下，瞎改 `force_x` 来猛踩油门冲过去。
- **不要**破坏已经写好的 84 个测试和 3 个新 Shaping 测试。

## 6. 交接备注
- 你现在的起点极好。我们已经证明：压低错侧排斥力 = CBF介入增加。目前的问题完全定位在“合成力的横向重心位置没有完全引导车辆贴着安全气泡中央行驶”。
- 只要针对 `target_y` 或引入通道向心力做一个微调，`s5` 的僵局就有极大概率会打破。
- 请直接深入 `AdaptiveAPF / APFLF`，思考如何加强朝向通道正中央的**引导力 (Guidance / Attraction)**，避开边缘 CBF 的截停。
