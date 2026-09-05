# Community training scheduling evidence

This note records implementation facts read from public repositories on 2026-09-05.
It is an engineering reference, not a claim that their numbers transfer unchanged
to WAN2.2-TI2V-5B.

| Project and revision | Evidence from source | Design implication |
| --- | --- | --- |
| [Open-Sora](https://github.com/hpcaitech/Open-Sora), `7ad6a96a135feb81f755c84fb391818718f6beb2` | `docs/train.md` launches `torchrun --nproc_per_node 8`; multi-node uses `colossalai run`; configs express bucket-specific batch sizes; `--load` restores optimizer and dataloader state; async checkpoint I/O is supported. `scripts/diffusion/train.py` creates a distributed process group, logs only from the master, and boosts the dataloader/model for distributed training. | A launch backend must record process topology and checkpoint/resume semantics explicitly. Resolution/frame buckets can legitimately have different per-device batch sizes, so one global BS should not be inferred from a paper headline. |
| [V-JEPA 2](https://github.com/facebookresearch/vjepa2), `204698b45b3712590f06245fbfba32d3be539812` | `app/vjepa_droid/train.py` calls `init_distributed`, passes `world_size` and `rank` into data initialization, wraps encoder/predictor/target encoder in `DistributedDataParallel`, saves checkpoints on rank 0, records `batch_size` and `world_size`, and resumes scheduler/loader position. The checked-in DROID config uses `nodes: 4`, `tasks_per_node: 8`, per-device `batch_size: 8`, and `save_every_freq: 25`. | Effective batch is a computed receipt field (`per-device × accumulation × world size`), not a hidden assumption. Rank-local logs and rank-0 checkpoint writes are standard safety boundaries. |
| [DreamerV3](https://github.com/danijar/dreamerv3), `e3f02248693a79dc8b0ebd62c93683888ddaccfe` | `embodied/run/train.py` computes `batch_steps = batch_size * batch_length`, gates training until replay has that many steps, and uses clocks for logging/reporting/checkpoint cadence. `elements.Checkpoint` persists agent and replay state and loads `from_checkpoint`. | Online world-model jobs need durable replay/checkpoint state and explicit train-ratio semantics; a process-level “completed” flag alone is not a scientific resume contract. |

## Adopted control-plane rules

The VerdiWM control plane now:

1. stores an immutable command, working directory, environment digest, and
   metadata in `verdiwm-background-job-spec`;
2. detaches long jobs into a new process session and streams stdout/stderr while
   writing heartbeat and terminal lifecycle receipts;
3. supports status, cancellation with bounded escalation, and exact-spec resume;
4. exposes an explicit `DistributedLaunchConfig` for single-process, `torchrun`,
   `accelerate`, and `deepspeed` command construction;
5. records effective batch size through one tested helper;
6. leaves model support claims to each backend. WAN2.2-DROID adapter training
   remains single-process/single-GPU until its model code is proven DDP-safe.

The supervisor is a host-local process manager, not a replacement for Slurm,
Kubernetes, or a site scheduler. A cluster adapter may submit the same immutable
spec to those systems and retain the same receipt schema.

