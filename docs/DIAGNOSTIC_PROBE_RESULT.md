# 诊断探针结果契约

`configs/schemas/diagnostic_probe_result.schema.json` 定义
`schema_version: 1`、`artifact_type: verdiwm-diagnostic-probe-result`。
生产端 `wmloop.diagnose.external_video_probe`、诊断调度和检索索引统一调用
`contracts.validate_document("diagnostic_probe_result", result)`，不再分别手写字段检查。

结果必须包含状态、probe/model/runtime 身份、failure signatures、非空 metrics 和
`claim_boundary`。metrics 的每个值必须是可表示为有限双精度数值的 JSON number；
布尔值、NaN、Infinity、字符串和空指标不属于测量结果。顶层和 video 对象拒绝
未声明字段；指标名称可以扩展，但不能为空。`ready` 至少包含一个互不重复的
failure signature；`invalid` 必须包含 error，且不包含可用于路由的 signature。

生产端写文件前验证整个结果，并使用共享原子写入工具。诊断调度还核对
probe/model/runtime 与执行前的 contract 一致。检索的写入和读取都验证结果，
同时保留现有 settled archive、receipt、CAS 内容和索引投影的绑定检查。
`invalid` 是可保存的失败产物，不能作为 ready 诊断路由或检索经验。

`wmloop-diagnostic-probe-output` 仍使用原
`diagnostic_probe_output.schema.json`，其历史格式保持可读。它与本契约是不同的
artifact，不会通过改名或补默认字段获得新的准入权限。通用
`verdiwm-auto-experiment-result` 也不能替代这里的诊断结果。
缺少必需字段的旧未约束 result 不会自动升级；历史证据应保留原字节，重新执行
时生成新的结果和 receipt。

这些结果只提供探索性失败诊断和候选路由依据。schema 合法、命令成功或视频
误差下降都不授予正式模型质量 verdict，也不替代冻结目标下的配对确认。

## 带符号的配对诊断记忆

同一版本结果可以选择包含非空 `paired_observations`，最多 16 条。不含该字段的
历史结果与检索投影保持原有序列化格式。每条观察使用
`diagnostic_paired_observation.schema.json`，包含下列身份与证据：

- `query` 明确绑定 primitive、control/candidate 剂量、metric 名称和方向、冻结
  协议 digest、输入 context digest 与 seed。候选必须预先声明完整 query 才能复用。
- `control_value`、`candidate_value`、`signed_delta` 和 `outcome` 保留正向、负向、
  零响应。`signed_delta` 将改善方向统一为正；较低为好的指标取相反差值。
  绝对差不超过冻结 `null_tolerance` 时记为 `null`。
- `evidence.specification` 描述冻结协议、算法和两侧 config、selection、manifest、
  measurement、prediction、ground truth，以及原生 receipt 或 execution artifact。
  `evidence.files` 为这些相对路径逐一记录 SHA-256；所有文件还须出现在外层已结算
  receipt 的 `artifact_refs` 中。
- `authority` 固定为 `ranking_only`。单 context、单 seed 的响应不会形成正式效果、
  升级、跨模型迁移或 verifier 权限。

一个结果中不能出现两条完全相同的 canonical query，即使引用不同 proof。
共享契约报 `DIAGNOSTIC_PAIRED_QUERY_DUPLICATE`，防止同一次 trial 的冲突观察被
遍历顺序覆盖。多个 metric 或 failure signature 可以描述同一 trial，但不增加其
投票权重。

目前公共 producer 支持 `history_sparse_dilation`：只允许同时修改
`world_model.history_rollout_dilation` 和 `wm_training.history_rollout_dilation`。
两者必须为相同正整数，其余 config 语义相同；权重、数据、selection、context、
seed、split、steps、chunks、fps 与 ground-truth 视频字节保持匹配。完整 comparison
必须在生成前冻结；[两轮使用指南](TWO_ROUND_DIAGNOSTIC_MEMORY.md) 提供模板。

## 冻结和测量公共数值配方

`wmloop.evaluate.future_video_metrics` 提供无 GPU 的两个命令：

```bash
uv run --offline --frozen python -m wmloop.evaluate.future_video_metrics freeze \
  --protocol comparison.json --output-root "$RUN_ROOT/frozen"

uv run --offline --frozen python -m wmloop.evaluate.future_video_metrics measure \
  --frozen-root "$RUN_ROOT/frozen" \
  --prediction "$PREDICTION_VIDEO" --ground-truth "$GROUND_TRUTH_VIDEO" \
  --output "$RUN_ROOT/measurement.json"
```

先设置自己环境中的 `RUN_ROOT`、`PREDICTION_VIDEO` 和 `GROUND_TRUTH_VIDEO`。
输出目录和 measurement 文件必须是新的；命令在锁内检查冲突，不覆盖旧证据。
freeze 在创建产物前检查完整 comparison，包括两侧剂量、允许候选、context、seed、
固定数据/权重声明与 metric 方向。缺失字段、非有限值或正式效果声明会提前拒绝。

冻包包含 `protocol.json`、`future_video_recipe.py` 和 `freeze-record.json`。后者
使用 `frozen_future_video_recipe.schema.json`，精确绑定协议与算法字节。数值代码
单独存放于稳定的 `future_video_recipe.py`；CLI、进程清理或文档的维护不改变其
算法身份。修改数值语义必须使用新的 recipe 版本与显式兼容实现，不能执行任意
归档 Python 来继续使用旧记忆。已登记的历史双文件配方仍能复核。

当前 `wan-future-rgb-v1` 要求视频恰有 13 帧、RGB uint8、512×768；丢弃第 0 帧
anchor，仅测第 1 到 12 帧。PSNR 从所有 future 帧的 aggregate MSE 计算；SSIM
使用 RGB 全像素的总体矩均值公式，不是局部窗口 SSIM。MSE 为 0 时测量记录
`psnr_db: null`、`psnr_is_infinite: true`；这类非有限 PSNR 不能成为 PSNR 配对先验。
measurement 使用 `future_frame_measurement.schema.json`，保留 12 条逐帧值、视频
digest、freeze digest 和诊断权限边界。

measure 先复核 freeze 文件的完整清单、所有 hash 和协议，再读取视频。数值回放
在隔离 Python 子进程执行，单视频最多 64 MiB、限时 45 秒；超时与取消都会回收
该进程组及解码后代进程。回放使用已安装的受支持数值实现，不 import 模型、不
执行归档脚本。

## 生产、结算和再利用

公共 API `wmloop.diagnose.paired_observation.build_wan_history_observations`
读取相对 `evidence_root` 的 specification，返回通过契约检查的观察。对应 CLI
向新的诊断结果文件添加观察：

```bash
uv run --offline --frozen python -m wmloop.diagnose.paired_observation \
  --evidence-root "$EVIDENCE_ROOT" --specification specification.json \
  --diagnostic-result diagnostic.json --output "$RUN_ROOT/paired-result.json"
```

specification 的 `evaluator_files` 使用
`{"future_video_recipe.py": "frozen/future_video_recipe.py"}`。producer 检查每侧
实际 manifest 与 invocation；命令参数使用有限白名单逐项消费，拒绝重复、缩写、
未知参数、`--flag=value` 和隐藏 config override。它还从 receipt 绑定的视频字节
重新计算全部数值，不能只依赖 measurement 中自报的标量。

原生调用若在已有通用 auto-experiment 内执行，可以使用
`wmloop.execute.wan_arm_slot.build_native_execution_evidence` 生成
`verdiwm-wan-arm-slot-native-execution`。它共享现有 Wan receipt 的 binding、process、
validation 与 authority 定义，仅保存执行证据。外层通用实验负责唯一的 lease、
budget、verifier、settled receipt 和 CAS/archive；该 artifact 不能再次结算。
独立原生 runner 的真实 inference receipt 也可作为 producer 输入。

`index_probe_experience` 入库时核对 settled trial、receipt、CAS、result 与配对
proof；`retrieve_probe_experiences` 和 `recheck_probe_experiences` 读取时重新核对并
回放。索引只保存原有投影列，新观察从权威 CAS 重建，不复制为第二套事实来源。
篡改投影、缺失 CAS 文件、算法不兼容、方向或协议不匹配均不能授予 signed prior。

`wmloop.retrieve.paired_prior.apply_paired_observation_priors` 在返回的 batch 副本
中写 `paired_observation_prior` 及 `paired_observation_retrieval`，不改调用者数据。
也可运行以下命令生成新的 batch：

```bash
uv run --offline --frozen python -m wmloop.retrieve.paired_prior \
  --batch cold-batch.json --matches matches.json \
  --archive-db "$ARCHIVE_DB" --cas-root "$CAS_ROOT" \
  --output "$RUN_ROOT/reused-batch.json"
```

每个相容 trial 按预先固定的类别策略投票：positive 为 `+0.25`，negative 为 `-0.5`，
null 为 `0`；对 trial 等权平均，重复 signature/metric 不重复投票。独立 trial 出现
正负冲突时 abstain，并保留两侧引用和结果。缺少权威 stores、query 不匹配或 VOID
时先验为 0。此处常量只表示候选排序政策，不是效应量或统计置信度。

onboarding compiler 与上述 CLI 共用该服务；scheduler 按
`scoring.retrieval_weight` 加权此带符号先验。旧的非负 `retrieval_prior` 继续表达
文献等排序依据，执行 PASS 不再自动产生正向效果奖励。有配对证据时，bounded
evidence capsule 优先保留负向、零响应和正向代表，并按 trial 去重，避免高标量
PASS 把负结果挤出下一轮上下文。

所有本地模型、数据、runtime 和存储路径由调用者提供。query context 用声明的
内容身份替换绑定资源位置，避免把开发机路径作为可复用的模型身份；私有执行
manifest 中的路径仅用于复核原始证据，不应直接导出为社区 bundle。
