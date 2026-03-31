# AI_MEMORY - 当前周期交接文档

> 下一个 AI / 工程师启动后，先完整阅读本文件，再阅读 `PROMPT_SYSTEM.md` 与 `RESEARCH_GOAL.md`，然后再动代码。
> 本文件已按 `2026-03-31` 的真实仓库状态重写。旧版 AI_MEMORY 中关于“`latest.pt` / `main.pt` 尚不存在、下一步先重新发起长训”的描述已经失效。

---

## 0. 当前开发游标

- 日期：
  - `2026-03-31`
- Git 游标：
  - `HEAD = 8951c5db785ab2c6217abf387bfe154ef2961b02`
  - `git status --short` 当前为空，工作树干净
- 当前论文主线：
  - 正文主方法已经重新锚定为白盒主链 `FSM + adaptive_apf + CBF-QP`
  - `rl_param_only` 当前只保留为可选增强 / 附录分支，在它没有稳定优于 `no_rl` 之前，不进入正文主方法
- 当前训练状态：
  - `outputs/rl_train_s5_param_only/checkpoints/latest.pt` 已存在
  - `outputs/rl_train_s5_param_only/checkpoints/main.pt` 已存在
  - `outputs/rl_train_s5_param_only/logs/main_stdout.log` 已出现完整训练结束记录：
    - `[ppo] complete device=cuda seed=0 timesteps_done=200192/200000 progress=100.00%`
- 当前 S5 benchmark 的真实结论：
  - 白盒 reference：`outputs/s5_rl_stage1_cuda__no_rl/summary.csv`
  - RL 结果：`outputs/s5_rl_stage1_cuda__rl_param_only/summary.csv`
  - `leader_final_x_delta_mean = -0.0503404628473092 m`
  - 也就是说，现有 `rl_param_only` 在 `seed 0/1/2` 上没有稳定优于 `no_rl`
  - 两边都保持 `collision_count = 0` 与 `boundary_violation_count = 0`
- 当前 S5 RL 归因的真实结论：
  - 归因结果目录：`outputs/s5_rl_stage1_cuda__rl_param_only/analysis/rl_attribution/`
  - 聚合文件：`aggregate.json`
  - 关键指标：
    - `dominant_bottleneck = supervisor_gating`
    - `rl_fallback_ratio_mean = 0.8136363636363636`
    - `fallback_low_confidence_steps_mean = 173.33333333333334`
    - `fallback_ood_steps_mean = 0.0`
    - `theta_change_ratio_mean = 0.18636363636363637`
    - `nominal_layer_changed_mean = 1.0`
    - `safety_intervention_ratio_mean = 0.5287878787878788`
- 当前阶段的直接判断：
  - 主阻塞已经不是训练器基础设施
  - 主阻塞也不是 OOD
  - 当前最值得立刻动手的瓶颈，是 `rl_param_only` 的 `confidence gate` 过于保守，导致 RL 大部分时间被 `rl_fallback` 吞掉

---

## 1. 已完成工作

### 1.1 论文主线与 artifact 入口已重新收口

- `README.md`
  - 已重写为当前真实项目状态
  - 已修复旧文档链接
  - 已加入 canonical paper matrix、offline RL attribution 的命令入口
- `docs/reproducibility.md`
  - 已补齐可复现实验矩阵与图表导出流程
- `docs/development.md`
  - 已同步当前分析层、归因层和验证约束

### 1.2 论文复现实验入口已改为白盒主线优先

- `scripts/reproduce_paper.py`
  - `PRIMARY_METHOD = "no_rl"`
  - 默认场景已扩成 `S1-S5`
  - 默认方法集已收口到白盒主线 + baseline：
    - `no_rl`
    - `apf`
    - `apf_lf`
    - `st_apf`
    - `dwa`
    - `orca`
  - 已新增 `--canonical-matrix`
    - 作用：一键展开 `30 seeds + S1-S5 + baselines + all ablations`
  - 结论：
    - 论文主链现在可以先不依赖 RL，独立跑出 canonical matrix

### 1.3 统计口径已升级为论文级默认配置

- `src/apflf/analysis/stats.py`
  - 已新增 `aggregate_metric_with_ci(...)`
  - group summary 的默认置信区间方法已从 `t-interval` 切到 deterministic bootstrap
  - 仍保留 `ci_method="t"` 的兼容能力
  - pairwise comparison 仍保留：
    - bootstrap delta CI
    - Wilcoxon
    - effect size

### 1.4 导出层已升级到论文图表层

- `scripts/export_figures.py`
  - reference method 默认值已统一成 `no_rl`
- `src/apflf/analysis/export.py`
  - 主表已纳入 comfort / runtime 指标：
    - `longitudinal_jerk_rms`
    - `steer_rate_rms`
    - `mean_step_runtime_ms`
    - `qp_solve_time_mean_ms`
  - 已补齐论文图表导出：
    - `trajectory_overview.pdf`
    - `risk_clearance_timeseries.pdf`
    - `qp_correction_timeline.pdf`
    - `mode_timeline.pdf`
    - `runtime_histogram.pdf`
    - `failure_case_panel.pdf`
  - 旧版总览图仍保留：
    - `metric_overview.pdf`
    - `safety_efficiency_tradeoff.pdf`

### 1.5 S5 RL 误差归因链路已落地

- 新增 `src/apflf/analysis/rl_attribution.py`
  - 已能对单个 seed 归因：
    - RL active / fallback 比例
    - 低置信度 fallback 次数
    - OOD fallback 次数
    - theta 变化比例
    - theta clipping 比例
    - safety intervention / safety fallback / QP engagement 比例
    - dominant bottleneck 判定
  - 已能与白盒 reference 回放进行逐 seed 对比：
    - `leader_target_speed_delta_abs_mean`
    - `leader_force_total_delta_norm_mean`
    - nominal / safe accel, steer 差异
    - `mode_mismatch_ratio`
    - `nominal_layer_changed`
- 新增 `scripts/analyze_s5_rl_attribution.py`
  - 已能从现有 benchmark 输出自动生成：
    - `seed_attribution.csv`
    - `aggregate.json`
    - `attribution_overview.pdf`

### 1.6 当前真实实验结果已确认

- 训练完成：
  - `outputs/rl_train_s5_param_only/checkpoints/main.pt`
  - `outputs/rl_train_s5_param_only/checkpoints/latest.pt`
- S5 benchmark 已存在：
  - `outputs/s5_rl_stage1_cuda__no_rl/summary.csv`
  - `outputs/s5_rl_stage1_cuda__rl_param_only/summary.csv`
- RL 归因结果已存在：
  - `outputs/s5_rl_stage1_cuda__rl_param_only/analysis/rl_attribution/aggregate.json`
- 当前 `main_stderr.log` 仍显示较多 safety fallback / solver 告警：
  - `preview_violation_after_qp`
  - `solver_status=maximum iterations reached`
  - `solver_status=primal infeasible`

### 1.7 当前验证状态

- 本周期已重新验证：
  - `python -m compileall src tests scripts` 通过
  - `python -m pytest -q tests/test_stats_export.py::test_export_paper_artifacts_writes_tables_and_figures tests/test_rl_attribution.py::test_analyze_s5_rl_attribution_script_writes_outputs` 通过
- 说明：
  - 本周期我没有再次完整跑完全量 `pytest`
  - 但本轮新增的 paper export 与 RL attribution 关键路径已经重新验证可用

---

## 2. 当前研究结论

- 当前仓库已经不是“缺实验脚手架”
  - 而是“工程闭环已成型，论文闭环还没有完全收口”
- 白盒主链已经足够支持正文：
  - `FSM + adaptive_apf + CBF-QP`
- RL 当前不应主导论文叙事：
  - 现有 `rl_param_only` 没有稳定优于 `no_rl`
  - 真实瓶颈是 `supervisor_gating`
  - 不是 OOD
  - 不是缺 checkpoint
  - 也不是 RL 完全没有影响 nominal controller
- 因此下一步不应该先去做大规模 RL 长训
  - 更不应该先跑昂贵的 `30 seeds` RL sweep
  - 应该先修正 `confidence gate` / `confidence calibration`

---

## 3. 下一步指令

### 3.1 第一优先级

下一个工程师启动 AI 后，第一件事不是重新训练，也不是继续改导出层，而是立刻写 `RL 置信度校准 + 滞回门控` 代码。

### 3.2 必须修改的文件

- `src/apflf/rl/policy.py`
- `src/apflf/decision/rl_mode.py`
- `src/apflf/utils/types.py`
- `tests/test_rl_supervisor.py`
- `tests/test_rl_attribution.py`

### 3.3 立刻要写的代码与数学约束

#### A. 在 `src/apflf/rl/policy.py` 中，重写 `TorchBetaPolicy.infer()` 的 confidence 定义

不要再把当前 entropy-based confidence 直接当成最终 gate 置信度。

请改成基于 Beta 分布方差的校准置信度。对每个动作维度 `i`，若策略输出的 Beta 参数为 `alpha_i, beta_i`，则定义：

```text
var_i = alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1))
```

令：

```text
v_uniform = 1 / 12
confidence_raw = clip(1 - mean_i(var_i) / v_uniform, 0, 1)
```

这里 `v_uniform = 1/12` 是 `Beta(1, 1)` 的方差，要求它对应“零置信度基线”。

必须满足：

- `confidence_raw ∈ [0, 1]`
- 当分布越集中时，`confidence_raw` 单调不减
- 当分布接近 `Beta(1,1)` 时，`confidence_raw` 应接近 `0`

#### B. 在 `src/apflf/decision/rl_mode.py` 中，实现两阈值滞回门控

新增两个阈值：

- `tau_enter`
- `tau_exit`

要求严格满足：

```text
0 <= tau_exit < tau_enter <= 1
```

门控规则必须写成：

```text
若 ||z_t||_inf > ood_threshold，则强制 fallback

否则：
  - 如果上一步没有接受 RL，则仅当 confidence_raw >= tau_enter 时接受 RL
  - 如果上一步已经接受 RL，则当 confidence_raw >= tau_exit 时继续保持 RL
  - 其余情况一律 fallback 到 FSM
```

其中 `z_t` 是 normalized observation，`||z_t||_inf` 是其无穷范数。

#### C. 必须保持的硬约束

不管怎么改 gate，以下数学约束不能破：

```text
theta_lower[j] <= theta_t[j] <= theta_upper[j]
|theta_t[j] - theta_{t-1}[j]| <= rate_limit[j]
```

并且：

- `mode_t` 仍然必须来自 FSM，不允许 RL 直接改 mode
- RL 仍然只允许输出 `theta`
- 不允许 RL 直接输出 `accel`
- 不允许 RL 直接输出 `steer`
- gate 拒绝时，必须 exact fallback 到白盒 decision：
  - `mode = fallback_decision.mode`
  - `theta = fallback_decision.theta`
  - `source = "rl_fallback"`

#### D. 强烈建议顺手补的诊断字段

在不改 `ModeDecision` 结构的前提下，优先给 `DecisionDiagnostics` 增补如下字段：

- `confidence_raw`
- `gate_open`
- `gate_reason`

目的：

- 下一个 benchmark 周期要能区分“raw confidence 低”与“滞回后仍被关掉”
- 不能只看 `confidence` 一个标量继续猜

### 3.4 为什么下一步必须是这段代码

因为当前真实归因已经说明：

- `fallback_ood_steps_mean = 0.0`
- `fallback_low_confidence_steps_mean = 173.33333333333334`
- `rl_fallback_ratio_mean = 0.8136363636363636`
- `theta_change_ratio_mean = 0.18636363636363637`
- `nominal_layer_changed_mean = 1.0`

这说明：

- OOD 不是主因
- RL 不是完全没产生 nominal 影响
- 真正卡住的是 confidence gate 太保守

所以现在最合理的工程动作，是先修 gate，而不是盲目继续长训

### 3.5 写完这段代码后的立即验收

先不要重训。

先直接用现有 `main.pt` 做 gatefix 后的 deterministic benchmark：

```bash
python scripts/benchmark_s5_rl.py --config configs/scenarios/s5_dense_multi_agent.yaml --seeds 0 1 2 --rl-checkpoint outputs/rl_train_s5_param_only/checkpoints/main.pt --exp-id-prefix s5_rl_gatefix_eval --deterministic-eval
```

然后立刻做归因：

```bash
python scripts/analyze_s5_rl_attribution.py --rl-run-dir outputs/s5_rl_gatefix_eval__rl_param_only --reference-run-dir outputs/s5_rl_stage1_cuda__no_rl --as-json
```

验收标准：

- `collision_count` 总和必须保持 `0`
- `boundary_violation_count` 总和必须保持 `0`
- `rl_fallback_ratio_mean` 必须严格小于 `0.8136363636363636`
- `rl_active_ratio_mean` 必须严格大于 `0.18636363636363637`
- `leader_final_x_delta_mean` 不能比当前的 `-0.0503404628473092` 更差

只有在 gatefix 后仍然无法改善这些指标，才允许讨论重新训练或改 reward。

---

## 4. Gatefix 之后的第二优先级

只有当 `RL gatefix` 完成并通过上面的 safety + attribution 验收后，才进入下一阶段：

### 4.1 白盒 canonical paper matrix

```bash
python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix
```

### 4.2 论文图表重新导出

```bash
python scripts/export_figures.py --input-dir outputs/paper_canonical
```

### 4.3 RL 是否升级为正文内容的判据

只有在新增结果同时满足以下条件时，才允许把 RL 从附录候选升级为正文内容：

- 相对 `no_rl` 有清晰正向改进
- `collision_count` 不回归
- `boundary_violation_count` 不回归
- 归因上不再呈现 `supervisor_gating` 主导

---

## 5. 技术栈红线

### 5.1 架构红线

- 必须保持三层白盒闭环：
  - `Mode Decision -> Nominal Controller -> Safety Filter`
- 白盒正文主线不能改写成端到端黑盒 RL
- 当前 RL 只能是 `param-only supervisor`

### 5.2 安全层红线

严禁修改以下文件：

- `src/apflf/safety/safety_filter.py`
- `src/apflf/safety/cbf.py`
- `src/apflf/safety/qp_solver.py`

### 5.3 接口红线

严禁破坏以下公共接口与数据结构：

- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

必须保持：

- `theta=None` 时精确退化到白盒基线
- FSM 与 RL supervisor 共用同一 `ModeDecision` 输出形状

### 5.4 RL 红线

- 不允许把 RL 扩成 `mode-only`
- 不允许把 RL 扩成 `full supervisor`
- 不允许 RL 直接输出 `mode`
- 不允许 RL 直接输出 `accel`
- 不允许 RL 直接输出 `steer`

### 5.5 工程红线

- 不允许引入 `SB3`
- PPO 后端必须继续沿用仓库内自定义 Torch 实现
- 所有实验必须保持 deterministic seeds
- 每次改动后至少执行：
  - `python -m compileall src tests scripts`
  - 相关增量 `pytest`

---

## 6. 不要做的事

- 不要再把下一步写成“先重新发起 200000 步长训”
- 不要在没有修 gate 的前提下先跑大规模 RL sweep
- 不要回头重写 export / stats / reproducibility 这条链路
- 不要把当前问题误判成 OOD
- 不要为了追求 RL 提升去动 safety red-line 文件
- 不要修改 `ModeDecision` 结构
- 不要修改 `compute_actions(observation, mode, theta=None)` 公共签名

---

## 7. 一句话交接

当前仓库已经完成了白盒论文主线收口、统计与图表导出升级、以及 S5 `rl_param_only` 的离线误差归因；真实数据表明当前 RL 的首要瓶颈是 `supervisor_gating` 而不是 OOD 或训练器缺陷。下一个工程师启动后，应该立刻去写 `Beta 方差置信度 + 两阈值滞回门控`，并在不改 safety red-line、不改 `ModeDecision`、不改 `compute_actions` 的前提下，用现有 `main.pt` 先做 gatefix benchmark，再决定是否需要重训。
