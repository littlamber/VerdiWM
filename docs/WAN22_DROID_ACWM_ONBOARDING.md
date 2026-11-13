# WAN2.2 + DROID ACWM onboarding

This note binds the user-facing request “first frame + DROID robot action
sequence -> 30 s future prediction” to the VerdiWM control-plane boundary.
It is an onboarding contract, not a claim that a checkpoint already meets the
target.

## Fixed input contract

- Backbone: external `Wan2.2-TI2V-5B` checkout, kept read-only.
- Dataset: processed DROID wrist subset at 192x320 and 5 FPS.
- Conditioning: per-frame 7D DROID action plus 14D proprioception, with one
  observed first frame and autoregressive history.
- Target horizon: 150 frames (30 seconds at 5 FPS). Episodes shorter than 150
  frames are not valid for the 30-second confirmation split.
- Outputs: generated video, paired ground truth, action-alignment report, and
  an immutable training/evaluation receipt under a fresh state root.

## What is automated

AI-generated adapters may inspect the model and data, convert metadata, bind
the WAN2.2 runtime, launch bounded training, and prepare WorldArena inputs.
The controller owns the evaluator, held-out split, GPU lease, output root,
receipt validation, and promotion decision. Generated code cannot replace
those bindings.

## Required architecture work

The existing DROID adapter reference (`embodydrive-srr`) targets Wan2.1-T2V-
1.3B. The existing WorldArena embodied example targets WAN2.2 but Robotwin.
Therefore this onboarding remains blocked until a conformance-tested WAN2.2
action/proprio/history adapter is supplied. A successful import or decreasing
training loss is insufficient evidence.

## Acceptance protocol

1. Audit train/validation manifests and verify video/latent/annotation pairing.
2. Run a CPU conformance test for action and temporal alignment.
3. Run a real WAN2.2 training pilot with a frozen dev split.
4. Evaluate 150-frame rollouts with WorldArena consistency metrics plus
   action-following and trajectory-fidelity protections.
5. Confirm on episode-disjoint validation data with at least three seeds.

The system must report `requires_interface_extension` until step 2 passes; it
must not silently fall back to a proxy or to the Wan2.1 implementation.

## 当前内置执行桥的限制

内置训练包装器当前拒绝启动训练。外部 trainer 尚未接入 `training.scale_plan`
的步数、累计批次和支持集约束，也未将 `output.root` 绑定为隔离的 checkpoint
目录。原有 checkpoint receipt 不包含可验证的训练产物路径，后续评测仍选择初始
checkpoint，因此进程退出成功不能视为训练评估闭环完成。

未启用执行时返回 `WAN22_TRAIN_EXECUTION_DISABLED`；即使设置
`VERDIWM_EXECUTE_WAN22=1`，也会在创建输出或启动子进程之前返回
`WAN22_TRAIN_BUDGET_OUTPUT_BINDING_UNSUPPORTED`。接通上述绑定并完成负向测试
与实际有界训练验证后，才能恢复此入口。

独立 PSNR/SSIM smoke 仍可用于配对 rollout 的测量。其冻结契约
`wan22-droid-psnr-ssim-smoke-v1` 只接受 `psnr`、`ssim` 两个指标。
六指标正式契约 `wan22-droid-verdi-contract-v1` 需要完整实现对应评测器；传给
此 smoke 包装器会在执行前报 `WAN22_EVAL_CONTRACT_UNSUPPORTED`，不能改写历史
冻结契约或用 smoke 测量替代正式确认。

有界 arm-slot 推理入口还会绑定源码与运行配置的内容摘要，并在进程结束后重算，
因此 `sa_wm` 依赖代码变化会使本次 receipt 失败。该检查复用 onboarding 的有界
源码扫描；扫描范围内出现未绑定的软链接或超出数量上限时会阻断启动。已有的环境、
缓存和运行产物目录排除规则继续适用；源码摘要不包含已安装 Python 环境中的
全部依赖，也不构成文件系统隔离。receipt 的 `passed` 仅表示有界进程、输入绑定
及原生 manifest/视频文件契约通过，视频非空检查不证明视频质量或正式指标提升。
