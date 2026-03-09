# 风格分治正式对比汇总

更新时间：2026-03-07

## 结论

- 当前最推荐的研发主线是 `param_shift`（动作参数化前移）版本。
- 当前最实用的按风格选型是：
  - `eco`: 用 `param_shift_long`
  - `normal`: 用 `param_shift_loose_fast`
  - `sport`: 用 `param_shift_loose_fast`
- 旧版 `constrained_fast_v2` 仍然是纯节能数值最强的 `FAST_RUN` 基线，但训练逻辑不如 `param_shift` 干净，且存在更重的环境动作投影依赖。

## 版本定义

- `baseline_constrained_fast_v2`
  - 老主线强基线，节能数字最高。
- `projection_ctx_fast`
  - 加入 `projection` 约束并做上下文加权后的版本。
- `param_shift_loose_fast`
  - 当前动作参数化前移后的 `FAST_RUN` 主线。
- `param_shift_long`
  - 当前动作参数化前移后的 `LONG_RUN` 版本。

## 对比结果

### eco

| 版本 | Saving E/Dist | Saving Net E/Dist | Saving Total | Recover Delta | Speed MAE | Dist MAE | Projection Lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_constrained_fast_v2` | 0.551% | 0.533% | 0.931% | -1.361% | 0.051 | 0.729 | N/A |
| `projection_ctx_fast` | 0.485% | 0.417% | 0.805% | -2.528% | 0.042 | 0.627 | 0.928 |
| `param_shift_loose_fast` | 0.462% | 0.460% | 0.734% | -0.793% | 0.035 | 0.537 | 0.000 |
| `param_shift_long` | 0.497% | 0.485% | 0.738% | -1.025% | 0.032 | 0.491 | 0.000 |

推荐：
- 当前推荐 `param_shift_long`
- 原因：在保持 `projection lambda = 0` 的前提下，`eco` 长跑比 `param_shift_loose_fast` 更强，且跟踪更稳。

结果文件：
- `baseline`: `artifacts/eco/constrained_fast_v2/best_motor_ppo_meta.json`
- `projection_ctx`: `artifacts/eco/projection_ctx_fast/best_motor_ppo_meta.json`
- `param_shift_fast`: `artifacts/eco/param_shift_loose_fast/best_motor_ppo_meta.json`
- `param_shift_long`: `artifacts/eco/param_shift_long/best_motor_ppo_meta.json`

### normal

| 版本 | Saving E/Dist | Saving Net E/Dist | Saving Total | Recover Delta | Speed MAE | Dist MAE | Projection Lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_constrained_fast_v2` | 0.539% | 0.533% | 0.925% | -1.058% | 0.062 | 0.878 | N/A |
| `projection_ctx_fast` | 0.227% | 0.203% | 0.378% | -1.069% | 0.023 | 0.350 | 0.598 |
| `param_shift_loose_fast` | 0.396% | 0.379% | 0.642% | -1.155% | 0.035 | 0.538 | 0.000 |
| `param_shift_long` | 0.322% | 0.341% | 0.506% | -0.051% | 0.027 | 0.401 | 0.000 |

推荐：
- 当前推荐 `param_shift_loose_fast`
- 原因：`normal` 上长跑会把策略带向偏保守区域，`FAST_RUN` 反而保留了更高节能。

结果文件：
- `baseline`: `artifacts/normal/constrained_fast_v2/best_motor_ppo_meta.json`
- `projection_ctx`: `artifacts/normal/projection_ctx_fast/best_motor_ppo_meta.json`
- `param_shift_fast`: `artifacts/normal/param_shift_loose_fast/best_motor_ppo_meta.json`
- `param_shift_long`: `artifacts/normal/param_shift_long/best_motor_ppo_meta.json`

### sport

| 版本 | Saving E/Dist | Saving Net E/Dist | Saving Total | Recover Delta | Speed MAE | Dist MAE | Projection Lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_constrained_fast_v2` | 0.397% | 0.649% | 0.642% | 8.957% | 0.054 | 0.757 | N/A |
| `projection_ctx_fast` | 0.275% | 0.401% | 0.422% | 4.432% | 0.033 | 0.442 | 0.896 |
| `param_shift_loose_fast` | 0.298% | 0.469% | 0.447% | 6.253% | 0.032 | 0.424 | 0.000 |
| `param_shift_long` | 0.165% | 0.343% | 0.252% | 6.317% | 0.016 | 0.245 | 0.000 |

推荐：
- 当前推荐 `param_shift_loose_fast`
- 原因：`sport` 长跑同样会过度保守，`FAST_RUN` 在节能和回收量上更平衡。

结果文件：
- `baseline`: `artifacts/sport/constrained_fast_v2/best_motor_ppo_meta.json`
- `projection_ctx`: `artifacts/sport/projection_ctx_fast/best_motor_ppo_meta.json`
- `param_shift_fast`: `artifacts/sport/param_shift_loose_fast/best_motor_ppo_meta.json`
- `param_shift_long`: `artifacts/sport/param_shift_long/best_motor_ppo_meta.json`

## 统一观察

- 三个风格在 `param_shift` 版本上都保持了：
  - `tracking_success_ratio = 100%`
  - `smooth_success_ratio = 100%`
  - `tracking_saving_success_ratio = 100%`
- `param_shift` 版本最重要的结构收益是：
  - `lagrange_projection_mean = 0.0`
  - 说明策略动作已经更接近可执行空间，不再依赖环境大量投影修正。

## 当前建议

- 如果目标是“当前更正规的主线研发底座”，选 `param_shift`
- 如果目标是“按风格出一组最好用的现成结果”，按下面选：
  - `eco -> param_shift_long`
  - `normal -> param_shift_loose_fast`
  - `sport -> param_shift_loose_fast`
- 如果目标是“纯追现阶段最高节能数值”，`constrained_fast_v2` 仍然更强，但不建议再作为主研发分支。
