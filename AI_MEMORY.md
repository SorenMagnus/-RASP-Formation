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
  - preview verification
  - fallback 机制
- 禁止通过删除 preview verification、删除 fallback、或直接放松 exact one-step safety 来“刷过” `s4/s5`。
- 当前阶段禁止优先改 `safety_filter.py` 作为主战场；近几轮已经证明 safety 继续微调收益很低，主瓶颈在 nominal 几何。

### 1.4 实验与复现红线
- 必须保持 headless CLI、YAML 配置驱动、全链路 seed 可复现。
- 每个实验输出必须可追溯并至少包含：
  - `config_resolved.yaml`
  - `summary.csv`
  - `traj/*.npz`
- 每次代码改动后至少通过：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`
- 不允许带着已知碰撞回归、边界越界回归进入下一轮开发。


## 2. 当前开发游标

### 2.1 当前 HEAD 与工作树
- 当前 `HEAD`：`596c173e37447b66e912410cd537905cb6fe4e86`（提交说明：`第六次更新`）
- 当前工作树有未提交改动：
  - `src/apflf/controllers/adaptive_apf.py`
  - `src/apflf/controllers/apf_lf.py`
  - `tests/test_adaptive_apf.py`

### 2.2 当前已验证状态
- 当前代码已通过：
  - `python -m pytest -q`
  - 结果：`81 passed in 94.72s`
  - `python -m compileall src tests scripts`
- 当前交接基准应以 **当前工作树 + stage41 输出** 为准，不要再参考更早 AI_MEMORY 里那条“优先改 safety fallback”的旧路线。

### 2.3 当前场景级游标
- 当前最可信的单 seed 结果：
  - `outputs/stage41_probe_s5/summary.csv`
    - `leader_final_x = 26.703600337341292`
    - `fallback_events = 206`
    - `safety_interventions = 241`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `team_goal_reached = False`
  - `outputs/stage41_probe_s4/summary.csv`
    - `leader_final_x = 53.16337668710457`
    - `fallback_events = 127`
    - `safety_interventions = 161`
    - `collision_count = 0`
    - `boundary_violation_count = 0`
    - `team_goal_reached = False`

### 2.4 当前核心结论
- `s1/s2` 之前的回归是健康的，本轮没有发现单测级退化。
- `s4/s5` 仍未打穿，当前系统仍然是“安全但过不去”。
- 目前已经基本确认：
  - `leader_bypass_force` 只能做局部纠偏，不是主瓶颈。
  - `target_x` preview 放得太远也不是主瓶颈。
  - 真正卡住 `s5` 的，是 `AdaptiveAPF` 中 **非当前绕行侧 front obstacle** 对 leader 施加的反向横向分量仍然过强。


## 3. 已完成工作

### 3.1 本轮最终保留在当前工作树里的代码改动

#### A. `AdaptiveAPF` 的 hazard 期 leader 速度节流
- 文件：
  - `src/apflf/controllers/adaptive_apf.py`
- 已完成内容：
  - 新增 leader-only 的 `_leader_hazard_speed_limit(...)`
  - 重写 `_reference_speed(...)`，使其在 hazard/relock 几何里不再长期输出巡航级前推
- 数学语义：
  - 只影响 `index == 0`
  - 只影响 hazard mode
  - 依据 `nearest relevant obstacle rear gap` 与 `lateral_error` 有界地下调 `reference_speed`
  - 不触碰 follow/recover
- 已验证：
  - 对 `s5` 的 nominal-vs-safety 纵向对抗有明显收敛作用
  - 但单靠速度节流不足以打穿 `s5`

#### B. `APFLF` 的 leader hazard `target_x` 近端 preview 保留
- 文件：
  - `src/apflf/controllers/apf_lf.py`
  - `tests/test_modes.py`（注意：这些测试已在当前 `HEAD` 或历史修改中固化，当前工作树未再改它）
- 已完成内容：
  - `APFLFController._leader_hazard_target_x(...)` 已支持在 hazard 阶段、即使 nominal side 已 re-lock，只要横向重定位还未完成且 relevant obstacle 未清空，就保留近端 preview，不立刻退回远端 `goal_x` 吸引
- 数学语义：
  - preview 保持有界
  - 只在 leader hazard 局部几何内生效
  - follow/recover 不受影响
- 已验证：
  - 轨迹诊断确认 `target_x` 确实被拉近
  - 但闭环 throughput 几乎未改善，说明 preview 深度不是主瓶颈

#### C. `leader_bypass_force` 的最终保留版本：只做“road opposition compensation”
- 文件：
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/controllers/adaptive_apf.py`
  - `tests/test_adaptive_apf.py`
- 已完成内容：
  - `APFLFController._leader_bypass_force(...)` 新增 `road_gain` 入口
  - 最终保留的逻辑不是整体放大 bypass force，而是：
    - 仍以固定基线 gain 计算 `Fy_guidance`
    - 仅当 `road_force` 与当前 `lateral_error` 方向对抗时，增加一项有界 `road_compensation`
  - `AdaptiveAPFController.compute_actions()` 已把当前实际 `road_gain` 传入该 helper
  - 增加了 `S4` 静态切片回归测试，确保 leader 在 overtake lane shift 未完成时不会再被 road push 顶回错误侧
- 数学语义：
  - 若 `sign(road_force_y) == -sign(lateral_error)`，才允许补偿
  - 补偿项是有界的，且只作用于 lateral guidance
  - `follow/recover` 不变
- 当前效果：
  - `s5` 基本回到 `stage38` 水平，没有再被这条线显著拉坏
  - `s4` 只有极小提升，尚不足以构成场景级突破

### 3.2 本轮做过但已明确放弃的路线

#### 已放弃路线 1：继续强化 `leader_bypass_force` 的近障碍 boost
- 试验结果：
  - `stage39_probe_s5` 回退到 `leader_final_x = 26.640952116137782`
  - `safety_interventions = 382`
- 结论：
  - 强 boost 会把 `S5` 搅坏，不能保留

#### 已放弃路线 2：继续把主要精力放在 `safety_filter.py`
- 结论来自多轮尝试：
  - safety fallback 微调已经接近上限
  - 即使更会“保安全”，也不会自动生成可通过 nominal 轨迹
  - 旧 AI_MEMORY 中那条“下一步先改 safety fallback”的指令已经过时

### 3.3 本轮新增/保留的关键测试
- `tests/test_adaptive_apf.py`
  - 保留：
    - stage58 staggered-blocker relock 横向 shaping 回归
    - stage48 pre-flip 不应提前 throttle 的回归
    - stage56 reorientation 应触发 leader speed throttle 的回归
  - 新增：
    - `test_adaptive_apf_overtake_guidance_overcomes_road_push_during_incomplete_lane_shift()`
- 全量测试当前为：
  - `81 passed`

### 3.4 场景级实验时间线（本轮最重要）
- `stage37_probe_s5`
  - 速度节流首次落地
  - `leader_final_x = 26.703611552662704`
  - `fallback_events = 206`
  - `safety_interventions = 241`
- `stage38_probe_s5`
  - 加入近端 preview 保留
  - 结果几乎与 `stage37` 重合，说明 `target_x` 不是主瓶颈
- `stage39_probe_s5`
  - 强近障碍 boost，已废弃
  - `leader_final_x = 26.640952116137782`
  - `safety_interventions = 382`
- `stage40_probe_s5`
  - 部分回退
  - `leader_final_x = 26.678859159710015`
- `stage41_probe_s5`
  - 当前保留版本
  - `leader_final_x = 26.703600337341292`
  - 与 `stage37/38` 基本等价


## 4. 下一步指令

### 4.1 下一个工程师启动 AI 后，应该立刻写哪段代码
- **优先文件**：
  - `src/apflf/controllers/adaptive_apf.py`
- **优先函数**：
  - `AdaptiveAPFController._shape_leader_nonrelevant_obstacle_force(...)`
  - 如有必要，可新增一个更明确的 helper，例如：
    - `_leader_nonrelevant_obstacle_lateral_scale(...)`
    - 或 `_leader_hazard_obstacle_decomposition(...)`
- **不要先改**：
  - `safety_filter.py`
  - `fsm_mode.py`
  - `apf_lf.py` 的 bypass 幅值

### 4.2 下一段代码要解决的精确数学问题
- 当前 `s5` 的核心问题不是 `target_x`，也不是 `leader_bypass_force`，而是：
  - 在 stage58 一类几何里，leader 已经选对侧、`target_y` 也正确，
  - 但 **非当前绕行侧** 的 front obstacle 仍然提供过大的反向 lateral force，
  - 导致总横向力 `F_total,y` 仍可能被压回错误方向。

- 下一个工程师要直接瞄准的数学约束是：
  - 对于 leader hazard 模式，定义
    - `F_total,y = F_att,y + F_guidance,y + F_road,y + F_obs,y + F_peer,y`
  - 在 `s5` 的典型 relock 切片（stage58 类几何）下，必须让
    - `F_total,y > 0`
  - 同时严格满足：
    - **不改变** nonrelevant obstacle 的 `force_x`
    - **只**缩放与当前绕行侧相反的 `force_y`
    - 缩放因子 `r` 必须有界且连续：
      - `0 <= r <= r_max`
      - 推荐 `0.65 <= r_max <= 0.90`
    - 若 obstacle 属于 `relevant_obstacles`，则 `r = 0`
    - 若该 obstacle 的 lateral force 已与当前绕行方向同号，则 `r = 0`
    - follow/recover 模式下，shape 必须完全退化为 identity

### 4.3 下一段代码的形式约束
- 必须保持：
  - `leader-only`
  - `hazard-only`
  - `nonrelevant-front-obstacle-only`
  - `lateral-only shaping`
- 严禁：
  - 改动 `force_x`
  - 在普通 obstacle 上全局放大/减小 repulsion
  - 再次回到 `safety_filter.py` 做主路径微调

### 4.4 下一个工程师写完后必须补的测试
- 在 `tests/test_adaptive_apf.py` 继续加 regression：
  - 基于 stage58 类 observation，验证 nonrelevant obstacle shaping 后：
    - `force_x` 与 raw 相同
    - `total lateral force` 相比当前 raw/shaped 基线明显增大
    - `actions[0].steer > 0.05`
- 如新增 helper，测试必须直接覆盖：
  - `relevant obstacle -> no shaping`
  - `aligned lateral push -> no shaping`
  - `nonrelevant adverse lateral push -> bounded shaping`

### 4.5 下一个工程师的验收顺序
1. 先跑：
   - `python -m compileall src tests scripts`
   - `python -m pytest -q`
2. 再跑：
   - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 --exp-id stage42_probe_s5`
3. `stage42_probe_s5` 的最低门槛：
   - `leader_final_x > 26.703600337341292`
   - `collision_count = 0`
   - `boundary_violation_count = 0`
   - 不允许显著拉高 `fallback_events`
4. 若 `s5` 单 seed 有实质提升，再跑：
   - `s4` 单 seed
   - `s1/s2/s3` 回归


## 5. 明确不要做的事
- 不要再把主要时间投入到 `safety_filter.py`。
- 不要通过放松 exact one-step safety、放松 boundary/obstacle collision 检查来换取 `leader_final_x`。
- 不要再继续调大 `leader_bypass_force` 的整体增益。
- 不要把 `target_x` 再推得更近或更远作为主策略；这条线已经证明不是主矛盾。
- 不要在没有 `s5` 单 seed 实质提升之前就跑大矩阵。


## 6. 交接备注
- 当前最可信的结论，以 `summary.csv` 为准，不要只看控制台 fallback 日志。
- 旧 AI_MEMORY 里那条“下一步先写 safety fallback near-stop creep”的指令已经失效，必须忽略。
- 当前工作树虽然未提交，但已经处于可验证、可继续接力的状态：
  - 单测与编译均通过
  - `stage41` 已给出当前最可信场景游标
  - 下一位工程师应直接从 `AdaptiveAPF` 的 nonrelevant obstacle lateral decomposition 开始
