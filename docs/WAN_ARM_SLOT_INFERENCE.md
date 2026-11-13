# 现有 Wan arm-slot checkpoint 的有界推理

本入口适用于外部 SA-WM 的 Wan2.2、两机械臂 slot、每 slot 10 维动作的现有
checkpoint。它调用模型仓库自己的 rollout，不训练、不转换动作格式。项目已有的
Wan/DROID 7D action、14D proprio 训练适配器属于另一种接口，不能凭模型名称相同
就绑定到这里。训练桥的限制见 [Wan/DROID 接入说明](WAN22_DROID_ACWM_ONBOARDING.md)。

在 VerdiWM 的 Python 3.12 控制环境中运行，`--runtime-python` 则指定模型自己的
Python。模型环境可以不同版本；虚拟环境的 Python 即使是符号链接，也保留其
launcher 路径。VerdiWM 不安装模型依赖，也不在控制面 import torch。

```bash
uv run --offline --frozen python -m wmloop.execute.wan_arm_slot \
  --source-root /workspace/SA-WM \
  --checkpoint /workspace/assets/checkpoint-63020 \
  --config /workspace/experiment/inputs/model.yaml \
  --selection /workspace/experiment/inputs/selection.json \
  --data-root /workspace/assets/droid-video-latent \
  --runtime-python /workspace/model-env/bin/python \
  --output-root /workspace/experiment/baseline \
  --budget-db /workspace/experiment/shared-budget.db \
  --budget-gpu-hours 1 \
  --gpu 0 --seed 20260915 \
  --split val --inference-steps 25 --max-chunks 1 --limit 1 \
  --timeout-seconds 600
```

`source-root` 必须提供 `scripts/evaluation/rollout_full_episode_sa_wm.py`；
`checkpoint` 必须包含 `dit_model.safetensors`。原生 YAML 至少需要
`world_model.backbone=wan2.2`、`action_schema=arm_slot`、`action_dim=10`、
`max_arm_slots=2`，以及 `wm_training.video_latent_condition_view=arm_slot_eef_pose`、
`action_condition_timestep_rate=raw` 和正的 `video_latent_required_target_fps`。
支持原生完整 YAML，包括没有额外缩进的列表。
`world_model.checkpoint` 和 `world_model.vae_checkpoint` 必须指向实际的基础模型
与 VAE 文件或目录；相对路径按原生命令的工作目录 `source-root` 解释。

selection 使用原生格式，例如：

```json
{"selected": [{"episode_id": 109, "frame_now": 40, "sample_index": 0}]}
```

episode、frame 和可选的 sample index 必须是非负整数；本次选中的 sample index
不得重复。数据必须先用模型自己的 reader 检查 latent/action 的 shape、dtype、
有限值、归一化和 inactive-arm mask。路径存在不代表数据和训练协议兼容；
不得把另一训练分布的 normalization stats 当成可互换文件。

所有 baseline、probe、candidate 共用同一个 `budget-db` 和预算总额，并各自使用
新的 `output-root`。预留成本为 `timeout-seconds / 3600`，完成后按实际子进程时长
结算；失败和超时也收费。预算总额必须容纳下一次完整超时预留，不能只看预计
成功耗时。`--gpu` 可以重复提供允许选择的物理 GPU，入口申请其中一张空闲卡，
原生进程内统一使用 `cuda:0`。无卡时不运行模型，并释放未使用的预算预留。

输出目录含 `input-binding.json`、`process/`、`rollouts/`、`evidence/` 和
`receipt.json`。输入绑定记录权重、配置、selection、rollout 入口和数据文件的
SHA-256；运行结束复核输入没有漂移。receipt 将日志和输出写入 CAS，检查原生
manifest 的 checkpoint、seed、split、步数、样本覆盖和非空 GT/prediction/comparison
视频。只有子进程和这些检查都通过才返回 `state=passed`；退出码为 0 表示通过，
2 表示已留存执行失败。执行前的输入错误不会占用 GPU。

配置引用的基础模型目录及 VAE 也逐文件进行运行前后 SHA-256 校验，避免只绑定
微调 DiT 而漏掉实际解码权重。绑定不复制模型权重到 CAS，不覆盖已安装运行时
依赖；重放仍需要对应外部资产。基础模型目录必须只读，输出和预算文件不能
位于其中。旧运行的 receipt 保留原字节，不补写后来增加的资产绑定。

输出和预算数据库必须在源码、权重和数据之外，且彼此不能嵌套。已有输出拒绝
覆盖。除运行时 Python launcher 外，绑定文件及其父目录拒绝符号链接。进程使用
独立 HOME/TMP、明确 GPU 可见性和进程组超时清理；它不是操作系统沙箱。
若启动后控制进程中断、未形成持久 receipt，预算保守保留，不自动退款。

## 从一次运行到有效的优化实验

先验证无干预 baseline，再固定 checkpoint、样本、seed、采样步数和评价定义，
单独改变一个确实执行的 inference hook。history dilation 在开头帧可能因为
clamp 到第一帧而完全无效；action scale 也可能只作用于关闭的 FiLM 分支。
应保存 hook 的实际选择/张量证据，再解释视频或指标差异。

原生 rollout 会把最后一帧 GT history anchor 放在预测视频的第 0 帧。
现有 PSNR/SSIM smoke 包含这帧；它的分数只能按原定义解释。未来帧评价必须在
实验前声明裁剪规则、使用新的 evaluator ID 并冻结，不能改旧协议或旧 receipt。
comparison 视频默认 GT 在左、prediction 在右；声明视频 probe 时应使用
`*/gt_left_sa_wm_right.mp4`，避免把同一目录三个视频都匹配进去。

这些 receipt 的 `authority.diagnostic_only=true`、`promotion=false`。
一次推理或一个 screen 只能为 Characterize/候选筛选提供证据。正式改善仍需
Retrieve 的可复核先验、独立 held-out target、多个匹配 seed、预声明阈值的
置信下界和所有保护指标，以及 verify 的终态判定和 settled archive。
当前入口没有把这些阶段自动连成 Wan arm-slot 的完整 campaign。

此命令自行管理 lease 和预算，不能直接嵌套到已持有 GPU 租约的 scheduler/probe
命令中；集成时应由外层 scheduler 统一管理资源并调用原生命令。
没有 protected action-following/长时指标的 PSNR/SSIM smoke，也不能替代正式
Wan/DROID 评价。缺少任何绑定时应保留 blocked、null 或 harmful 结果，不能补造
正向证据。
