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
