# AI_MEMORY

## 1. 技术栈红线

### 1.1 研究目标
- 目标不是做 demo，而是持续收敛到可投稿 IEEE 级别论文的 artifact。
- 主架构必须始终保持三层闭环，不允许退化成单层或黑盒：
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
- 行为调优类改动必须最终回归验证 `s1 -> s2 -> s3`，只要 `s1` 任一 seed 退化，就必须立即停手回滚，不允许继续扩展到其他场景。

### 1.4 架构红线
- 不允许为了通过 s4/s5 直接删除 safety preview、删除 fallback、改成纯启发式停车器。
- 不允许把 leader/follower 编队语义删掉，只能做有界增强。
- 不允许用“放宽安全约束”替代“低速精细 maneuver”；任何 relax 只能发生在 near-stop creep 区间，并且必须保持 exact one-step safety。
- 论文 artifact 红线不变：
  - 同一轨迹的 metrics 必须可重算
  - summary / traj / config 必须一致可追溯
  - 不能依赖手动 UI 调参数作为实验来源

## 2. 当前开发游标

### 2.1 当前代码游标
- 当前工作区已修改但未提交的 tracked 文件：
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/controllers/adaptive_apf.py`
  - `src/apflf/controllers/base.py`
  - `src/apflf/decision/fsm_mode.py`
  - `src/apflf/safety/safety_filter.py`
  - `tests/test_modes.py`
  - `tests/test_cbf_filter.py`
- 当前工作区未跟踪的日志文件：
  - `stage23_probe_s5_run.log`
  - `stage24_probe_s5_run.log`
  - `stage25_probe_s5_run.log`
- 当前开发所处位置：
  - 已完成 `s4/s5` 的失败复现
  - 已完成一轮 nominal/FSM 侧的最小增强
  - 已完成一轮 safety fallback 的 low-speed creep 修补
  - **尚未完成最新 safety patch 之后的场景级重新验证**

### 2.2 当前稳定代码级状态
- 已恢复并确认开发环境：
  - `python -m pip install -e ".[dev]"`
- 当前代码通过：
  - `python -m compileall src tests scripts`
  - `python -m pytest -q`
- 当前结果：
  - `71 passed`

### 2.3 当前已验证的实验游标
- 下面这些输出已经真实落盘，可作为交接依据：
  - `outputs/stage21_validate_s4`
  - `outputs/stage21_validate_s5`
  - `outputs/stage22_validate_s4`
  - `outputs/stage23_probe_s4`
  - `outputs/stage23_probe_s5`
  - `outputs/stage24_probe_s5`
- `stage25_probe_s5` 在本轮对话中被中断，**不要把它视为有效结论**。

### 2.4 当前最重要的结论
- `s1/s2/s3`：
  - 上一轮记忆中的 `stage20b` 结论仍然是历史稳定结论
  - 但在本轮针对 `s4/s5` 的新 patch 之后，**还没有重新跑 `s1/s2/s3` 回归**
- `s4`：
  - 仍未通过
  - leader 已经能更明显地偏到右侧，但仍卡在慢车后方，未形成真正超车
- `s5`：
  - 仍未通过
  - 但 failure mode 已明显收缩：从最初的错误侧选择 `yield_left`，修到 `yield_right`
  - leader_final_x 已从 `24.68` 提升到约 `26.70`
  - 当前卡点已收缩到：
    - near-stop 状态下需要“向左打角 + 小正加速度”的低速重定向
    - safety fallback 仍然过于保守，导致车速被锁死

## 3. 已完成工作

### 3.1 基线恢复
- 重新安装了开发依赖，恢复了本机可运行环境。
- 重新确认了基线状态：
  - `python -m compileall src tests scripts` 通过
  - 初始全量测试重新跑通时为 `67 passed`
  - 本轮新增回归测试后，当前为 `71 passed`

### 3.2 已完成的场景级验证

#### stage21：首次正式验证 s4/s5
- `outputs/stage21_validate_s4`
  - seeds `0/1/2` 全部失败
  - `leader_final_x` 约 `53.09 / 53.15 / 53.37`
  - `fallback_ratio` 约 `0.193 / 0.189 / 0.207`
  - 无碰撞、无边界越界，但 leader 被 safety layer 长期压在慢车后方
- `outputs/stage21_validate_s5`
  - seeds `0/1/2` 全部失败
  - `leader_final_x` 约 `24.68 / 24.68 / 24.69`
  - `fallback_ratio` 约 `0.353 / 0.332 / 0.550`
  - 初始 failure mode 是 `yield_left -> escape_left`，明显选错侧

#### 诊断结论：问题不只是 FSM，也不只是 adaptive controller
- 额外做过两组对照诊断：
  - 把 `decision.kind` 暂时改成 `static + follow`
  - 把 `controller.kind` 暂时改成 `apf_lf`
- 结论：
  - `s4/s5` 失败不是“仅仅 FSM 错了”
  - 也不是“仅仅 adaptive_apf 错了”
  - 根因更接近：
    - leader 缺少真正的侧向绕行目标
    - staggered blockers 下 side preference 规则不够局部
    - near-stop 时 safety fallback 无法完成低速转向蠕动

### 3.3 本轮已完成的源码改动

#### 改动 A：FSM 的 `preferred_side` 从粗糙通道判断改成“最近不对称阻塞物优先”
- 文件：
  - `src/apflf/decision/fsm_mode.py`
- 具体内容：
  - 新增 `_passing_side_margins()`
  - `_preferred_side()` 不再只看整条道路外缘通道，而是：
    - 先按 longitudinal gap + lateral proximity 排序 front obstacles
    - 优先根据最近且有明显不对称余量的 obstacle 选侧
    - 若单个 obstacle 不够 decisive，再做带 longitudinal weight 的 margin 汇总
- 作用：
  - 把 `s5` 的初始错误策略从 `yield_left` 修正为 `yield_right`

#### 改动 B：leader hazard 模式不再依赖固定横向 bias，而是拥有显式 bypass target
- 文件：
  - `src/apflf/controllers/apf_lf.py`
  - `src/apflf/controllers/adaptive_apf.py`
  - `src/apflf/controllers/base.py`
- 具体内容：
  - `BaseNominalController._mode_behavior_force()` 对 leader 在 hazard mode 下改为 `0`
  - 新增 leader 局部几何 helper：
    - `_leader_passing_side_margins()`
    - `_leader_front_obstacles()`
    - `_leader_behavior_side_sign()`
    - `_leader_behavior_target_y()`
    - `_leader_bypass_force()`
  - `_leader_goal_target()` 现在允许根据当前 mode 和前方 obstacle 生成显式横向绕行目标
  - `AdaptiveAPFController.compute_actions()` / `APFLFController.compute_actions()` 都已接入 `leader_guidance_force`
- 数学语义：
  - leader 的 hazard 语义不再是常数 `±Fy`
  - 而是 `target_y(mode, obstacles)` 与 `Fy_guidance ∝ (target_y - y_leader)` 的有界引导

#### 改动 C：leader 在 staggered blockers 下允许局部 side flip
- 文件：
  - `src/apflf/controllers/apf_lf.py`
- 具体内容：
  - `_leader_behavior_side_sign()` 增加局部切边逻辑
  - 当 nominal mode 给定的 side 对当前“仍在前方的 anchor obstacle”已不可行，而对侧明显更可行时，leader 可局部翻转 side sign
  - 该判据只在 leader 的局部目标层发生，不改动 FSM 全局 mode label
- 作用：
  - 让 `s5` 从“始终粘在错误侧”推进到“开始尝试右绕后再左转”

#### 改动 D：safety fallback 对 near-stop creep 的 preview deficit 容忍从 `0.03` 放宽到 `0.08`
- 文件：
  - `src/apflf/safety/safety_filter.py`
- 具体内容：
  - `_fallback_action()` 中：
    - `creep_margin_tolerance = 0.03 -> 0.08`
  - 语义限制没有变：
    - 仅在 `state.speed <= 0.25`
    - `candidate.accel > 0`
    - `verification_error is None`
    - nominal 仍想前进时才允许
- 这是一个**受限 relax**，不是全局放松 preview safety

### 3.4 本轮新增/更新的测试

#### `tests/test_modes.py`
- 新增：
  - `test_fsm_preferred_side_prioritizes_the_nearest_asymmetric_blocker()`
  - `test_apf_lf_leader_goal_target_builds_a_true_bypass_offset()`
  - `test_apf_lf_leader_goal_target_can_flip_locally_when_staggered_blocker_changes_side()`
- 更新：
  - `test_apf_lf_controller_consumes_mode_topology_and_behavior()` 现在使用真实 blocker 场景，不再假设“无障碍也必须横摆”

#### `tests/test_cbf_filter.py`
- 新增：
  - `test_fallback_allows_small_preview_deficit_for_one_step_safe_creep()`
- 这个测试直接固化了 `s5` 停滞点的一个真实诊断快照，保证 near-stop creep relax 只发生在：
  - one-step exact safety 成立
  - preview deficit 很小
  - 正加速度确实能帮助重定向

## 4. 当前实验事实与量化进度

### 4.1 s4 量化进度
- `stage21_validate_s4`：
  - seed0 `leader_final_x = 53.09`
- `stage22_validate_s4`：
  - seed0 `leader_final_x = 53.12`
- `stage23_probe_s4`：
  - seed0 `leader_final_x = 53.39`
- 结论：
  - leader 横向偏置略增强，但没有质变
  - 当前仍是“跟车偏移”，不是“形成可持续超车轨迹”

### 4.2 s5 量化进度
- `stage21_validate_s5`：
  - seed0 `leader_final_x = 24.68`
  - mode 主要是 `yield_left -> escape_left`
- `stage23_probe_s5`：
  - seed0 `leader_final_x = 26.70`
  - mode 已变成 `yield_right -> escape_right`
- `stage24_probe_s5`：
  - seed0 `leader_final_x = 26.71`
  - 仍失败，但 leader 的停滞点已经从 `y ≈ +1.5` 收缩到 `y ≈ -0.9`
- 结论：
  - side selection 和 nominal bypass 确实有增益
  - 但当前真正卡住的是 near-stop safety fallback

### 4.3 已完成的关键诊断
- 在 `stage24_probe_s5` 的停滞步附近，已做过精确诊断：
  - 代表性状态：
    - `step = 80`
    - leader `x = 26.6805`
    - leader `y = -0.9103`
    - leader `speed = 0.0995`
    - nominal action `= (accel=2.0, steer=0.4363)`
- 对 candidate `(accel > 0, steer = 0.436)` 的诊断结果：
  - `(0.1, 0.436)`：
    - preview margin `≈ -0.0354`
    - `verification_error is None`
  - `(0.2, 0.436)`：
    - preview margin `≈ -0.0453`
    - `verification_error is None`
  - `(0.5, 0.436)`：
    - preview margin `≈ -0.0752`
    - `verification_error is None`
- 含义：
  - 这些动作在 exact one-step sense 上是安全的
  - 但在 preview margin 上略负，因此原本被 fallback 拒绝
  - 这就是 `creep_margin_tolerance` 调整的直接依据

## 5. 下一步指令

### 5.1 下一位工程师启动 AI 后，应该马上写哪段代码
- **优先文件**：
  - `src/apflf/safety/safety_filter.py`
- **优先函数**：
  - `CBFQPSafetyFilter._fallback_action()`
- **马上要写的代码**：
  - 在 `_fallback_action()` 中新增一个专用的 `near-stop guided creep` 选择分支，优先于“原地打角停车”型 fallback。
  - 推荐新增独立 helper，例如：
    - `_select_near_stop_guided_creep(...)`
  - 不要继续先调 `fsm_mode.py`
  - 不要继续先调 `adaptive_apf.py`
  - 眼下最该写的是 low-speed fallback 逻辑，而不是再加新的 discrete mode

### 5.2 必须满足的具体数学约束
- 该新分支只能在以下条件同时满足时激活：
  - `state.speed <= 0.5`
  - `nominal_vector[0] > 0.0`
  - `verification_error is None`
  - `candidate.accel > 0.0`
  - `candidate.steer` 与 nominal/guided 方向同号
- 允许的 preview deficit 必须有界：
  - 设 `epsilon_creep_local`
  - 要求 `0.08 <= epsilon_creep_local <= 0.12`
  - 只允许对 near-stop creep 使用这个有界 relax
  - 不允许把普通 fallback 的全局 preview margin 放松到这个范围
- exact one-step safety 绝不能放松：
  - 必须保持 `_verify_safe_action(...) is None`
  - 也就是：
    - 不允许 boundary violation
    - 不允许一步后 obstacle collision
    - 不允许一步后 peer collision
- 候选动作选择顺序应明确写死：
  - 先最大化 `candidate.accel`
  - 再最大化 `margin`
  - 再最小化 `steer_delta`
- 这个分支必须只服务于“低速重定向蠕动”，不能影响高速正常 fallback

### 5.3 下一位工程师写完代码后的验收顺序
1. 先跑：
   - `python -m compileall src tests scripts`
   - `python -m pytest -q`
2. 再跑单 seed probe：
   - `python scripts/run_experiment.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 --exp-id stage26_probe_s5`
3. 单 seed 的最低验收门槛：
   - `leader_final_x > 30.0`
   - `collision_count = 0`
   - `boundary_violation_count = 0`
4. 如果 `stage26_probe_s5` 通过，再跑：
   - `s5` 的 3 seeds
   - `s4` 的 1 seed probe，再视结果扩成 3 seeds
5. 只要 `s5` 或 `s4` 有改善，就必须回归验证：
   - `s1`
   - `s2`
   - `s3`

### 5.4 明确不要做的事情
- 不要把 `safe_distance` 直接调小来“穿过去”。
- 不要删除 preview verification。
- 不要把 fallback 改成纯停车策略。
- 不要先去做大规模 `30 seeds × s1-s5` 矩阵；当前阶段离论文矩阵还早。

## 6. 交接备注
- 当前最可信的最新结论，以 `summary.csv` 为准，不要只看 `.log`。
- `stage25_probe_s5_run.log` 是中断文件，不是有效结果。
- 当前 `AI_MEMORY.md` 已同步到：
  - 当前改动文件状态
  - 当前通过的测试状态
  - 当前有效实验产物
  - 下一刀该写的代码和数学约束
