#!/usr/bin/env python3
"""Run a paired Ctrl-World ACWM predictive fingerprint campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.evaluate.adapters.ctrl_world_predictive import evaluate_ctrl_world_prediction_receipt
from wmloop.experiments.ctrl_world_fingerprint import (
    ctrl_world_probe_context,
    evaluate_ctrl_world_fingerprint,
    load_ctrl_world_campaign,
)


class CtrlWorldPredictiveCampaignError(RuntimeError):
    """The external runtime cannot satisfy the frozen predictive campaign."""


def protocol_rows(*, split_payload: Mapping[str, Any], campaign: Mapping[str, Any], protocol: str) -> list[dict[str, Any]]:
    protocols = campaign.get("protocols", {})
    if protocol not in protocols:
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_PROTOCOL_UNKNOWN")
    split_name = str(protocols[protocol]["split"])
    rows = split_payload.get(split_name)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_PROTOCOL_SPLIT_INVALID")
    normalized = [
        {
            "task_id": str(row["task_id"]),
            "episode_id": str(row["episode_id"]),
            "seed": int(row["seed"]),
        }
        for row in rows
    ]
    required = int(protocols[protocol]["required_receipts_per_dose"])
    if len(normalized) != required:
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_PROTOCOL_SPLIT_COUNT_MISMATCH")
    if len({row["task_id"] for row in normalized}) != 1:
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_PROTOCOL_MULTI_TASK_UNSUPPORTED")
    return normalized


def prediction_receipts_from_summary(
    *,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    dose: float,
    horizon_frames: int,
    evaluator_version: str,
    receipt_root: Path,
) -> list[dict[str, object]]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(rows):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_SUMMARY_EPISODE_COUNT_MISMATCH")
    video_dir = Path(str(summary["run_dir"])) / "videos"
    video_by_index: dict[int, Path] = {}
    for path in sorted(video_dir.glob("*.mp4")):
        try:
            video_by_index[int(path.name.split("_", 1)[0])] = path.resolve()
        except (ValueError, IndexError):
            continue
    destination = Path(receipt_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for index, (identity, episode) in enumerate(zip(rows, episodes, strict=True)):
        if not isinstance(episode, Mapping):
            raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_SUMMARY_EPISODE_INVALID")
        if str(episode.get("traj_id")) != str(identity["episode_id"]):
            raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_SUMMARY_IDENTITY_MISMATCH")
        rollout_ref = video_by_index.get(index)
        if rollout_ref is None:
            raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_SUMMARY_VIDEO_MISSING")
        receipt = {
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-prediction-receipt",
            "evidence_source": "paired_ground_truth_rollout",
            "task_id": str(identity["task_id"]),
            "episode_id": str(identity["episode_id"]),
            "seed": int(identity["seed"]),
            "horizon_frames": int(horizon_frames),
            "metrics": {
                "rollout_video_psnr": float(episode["rollout_video_psnr"]),
                "rollout_video_l1": float(episode["rollout_video_l1"]),
                "segment_final_mae": float(episode["segment_final_mae_mean"]),
                "segment_view_pair_mae": float(episode["segment_view_pair_mae_mean"]),
                "segment_view_fused_mae": float(episode["segment_view_fused_mae_mean"]),
            },
            "action_conditioned": True,
            "rollout_ref": str(rollout_ref),
            "evaluator_version": evaluator_version,
        }
        receipt_path = destination / (
            f"dose_{_dose_tag(dose)}__{identity['task_id']}__{identity['episode_id']}__s{identity['seed']}.json"
        )
        _write_json(receipt_path, receipt)
        index_rows.append({"dose": float(dose), "receipt_ref": str(receipt_path)})
    return index_rows


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_OUTPUT_EXISTS")
    campaign_path = Path(args.campaign).resolve(strict=True)
    split_path = Path(args.heldout_split).resolve(strict=True)
    campaign = load_ctrl_world_campaign(campaign_path)
    split_payload = _load_json(split_path)
    rows = protocol_rows(split_payload=split_payload, campaign=campaign, protocol=args.protocol)
    configured_doses = tuple(float(value) for value in campaign["probe"]["doses"])
    selected_doses = tuple(float(value) for value in (args.doses or configured_doses))
    if not selected_doses or any(dose not in configured_doses for dose in selected_doses):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_SELECTED_DOSE_INVALID")

    task_id = str(rows[0]["task_id"])
    dataset_dir = Path(args.dataset_root).resolve(strict=True) / task_id
    assets = {
        "ctrl_world_root": Path(args.ctrl_world_root).resolve(strict=True),
        "ctrl_world_model_root": Path(args.ctrl_world_model_root or args.ctrl_world_root).resolve(strict=True),
        "dataset_dir": dataset_dir.resolve(strict=True),
        "data_stat": Path(args.data_stat).resolve(strict=True),
        "svd_model": Path(args.svd_model_path).resolve(strict=True),
        "clip_model": Path(args.clip_model_path).resolve(strict=True),
        "checkpoint": Path(args.ckpt_path).resolve(strict=True),
        "reward_checkpoint": Path(args.reward_ckpt).resolve(strict=True),
    }
    _validate_dataset_freeze(
        freeze_path=Path(args.dataset_freeze).resolve(strict=True),
        dataset_root=Path(args.dataset_root).resolve(strict=True),
        task_id=task_id,
        episode_ids=tuple(str(row["episode_id"]) for row in rows),
    )
    hash_cache_path = Path(args.asset_hash_cache).resolve() if args.asset_hash_cache else None
    hash_cache = _load_hash_cache(hash_cache_path)
    checkpoint_identity = _asset_identity(
        assets["checkpoint"], cache=hash_cache, hash_large_assets=bool(args.hash_large_assets)
    )
    reward_identity = _asset_identity(
        assets["reward_checkpoint"], cache=hash_cache, hash_large_assets=bool(args.hash_large_assets)
    )
    if hash_cache_path is not None and args.hash_large_assets:
        _write_json(hash_cache_path, {"schema_version": 1, "assets": hash_cache})
    formal_asset_identity_ready = all(
        identity.get("sha256_state") in {"computed", "cached"}
        for identity in (checkpoint_identity, reward_identity)
    )
    preflight = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-predictive-runtime-preflight",
        "state": "ready" if formal_asset_identity_ready else "pilot_ready_hash_deferred",
        "protocol": args.protocol,
        "task_id": task_id,
        "doses": list(selected_doses),
        "identities": rows,
        "interact_num": int(args.interact_num),
        "pred_step": 5,
        "horizon_frames": int(args.interact_num) * 4,
        "num_inference_steps": int(args.num_inference_steps),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assets": {name: str(path) for name, path in assets.items()},
        "formal_asset_identity_ready": formal_asset_identity_ready,
        "checkpoint_identity": checkpoint_identity,
        "reward_checkpoint_identity": reward_identity,
        "upstream_eval_sha256": _sha256(assets["ctrl_world_root"] / "scripts" / "eval_replay_rollout_long.py"),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "runtime-preflight.json", preflight)
    if args.dry_run:
        return {**preflight, "dry_run": True, "output_root": str(output_root)}

    module = _load_upstream_module_with_local_model_base(
        eval_root=assets["ctrl_world_root"],
        model_root=assets["ctrl_world_model_root"],
        local_model_base=assets["svd_model"].parent,
    )
    run_state = _install_runtime_adapters(
        module=module,
        dataset_dir=assets["dataset_dir"],
        data_stat=assets["data_stat"],
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
        probe_id=str(campaign["probe"]["probe_id"]),
        seed_by_episode={str(row["episode_id"]): int(row["seed"]) for row in rows},
    )
    episode_list = output_root / "episode-list.jsonl"
    episode_list.write_text(
        "".join(
            json.dumps(
                {
                    "split": "val",
                    "traj_id": row["episode_id"],
                    "instruction": "",
                    "start_idx": 0,
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    evaluator_version = f"ctrl-world-long:{preflight['upstream_eval_sha256'][:12]}"
    receipt_rows: list[dict[str, object]] = []
    original_argv = list(sys.argv)
    try:
        for dose in selected_doses:
            run_state["dose"] = float(dose)
            dose_root = output_root / "upstream" / f"dose_{_dose_tag(dose)}"
            before = set(dose_root.glob("*/summary.json")) if dose_root.exists() else set()
            sys.argv = [
                str(assets["ctrl_world_root"] / "scripts" / "eval_replay_rollout_long.py"),
                "--task-type",
                "replay",
                "--svd-model-path",
                str(assets["svd_model"]),
                "--clip-model-path",
                str(assets["clip_model"]),
                "--ckpt-path",
                str(assets["checkpoint"]),
                "--reward-ckpt",
                str(assets["reward_checkpoint"]),
                "--output-dir",
                str(dose_root),
                "--episode-list-jsonl",
                str(episode_list),
                "--save-video-limit",
                str(len(rows)),
                "--seed",
                str(rows[0]["seed"]),
            ]
            module.main()
            run_state["close_dose"]()
            summaries = set(dose_root.glob("*/summary.json")) - before
            if len(summaries) != 1:
                raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_UPSTREAM_SUMMARY_AMBIGUOUS")
            summary_path = summaries.pop().resolve()
            summary = _load_json(summary_path)
            summary = dict(summary)
            summary["run_dir"] = str(summary_path.parent)
            receipt_rows.extend(
                prediction_receipts_from_summary(
                    summary=summary,
                    rows=rows,
                    dose=dose,
                    horizon_frames=int(args.interact_num) * 4,
                    evaluator_version=evaluator_version,
                    receipt_root=output_root / "receipts",
                )
            )
    finally:
        sys.argv = original_argv
        run_state["close_dose"]()

    receipt_index = {
        "artifact_type": "verdiwm-ctrl-world-fingerprint-receipt-index",
        "campaign_id": campaign["campaign_id"],
        "protocol": args.protocol,
        "rows": receipt_rows,
    }
    receipt_index_path = output_root / "receipt-index.json"
    _write_json(receipt_index_path, receipt_index)
    for row in receipt_rows:
        evaluate_ctrl_world_prediction_receipt(
            receipt_path=Path(str(row["receipt_ref"])),
            heldout_split_path=split_path,
            split_name=str(campaign["protocols"][args.protocol]["split"]),
        )
    complete = set(selected_doses) == set(configured_doses)
    if complete:
        fingerprint = evaluate_ctrl_world_fingerprint(
            campaign_path=campaign_path,
            receipt_index_path=receipt_index_path,
            heldout_split_path=split_path,
            protocol=args.protocol,
            output_root=output_root / "fingerprint",
            archive_db=Path(args.archive_db).resolve() if args.archive_db else None,
            cas_root=Path(args.cas_root).resolve() if args.cas_root else None,
        )
    else:
        fingerprint = None
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-predictive-campaign-run",
        "state": "complete" if complete else "partial",
        "protocol": args.protocol,
        "doses": list(selected_doses),
        "receipt_count": len(receipt_rows),
        "receipt_index": str(receipt_index_path),
        "fingerprint_manifest": fingerprint,
        "claim_boundary": campaign["claim_scope"],
    }
    _write_json(output_root / "campaign-run.json", report)
    return report


def _load_upstream_module(*, eval_root: Path, model_root: Path) -> Any:
    root = Path(eval_root).resolve()
    model_source = Path(model_root).resolve()
    scripts = root / "scripts"
    source = scripts / "eval_replay_rollout_long.py"
    if not source.is_file():
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_UPSTREAM_EVAL_MISSING")
    ordered_paths = (str(model_source), str(scripts), str(root))
    for path in reversed(ordered_paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("verdiwm_ctrl_world_predictive_runtime", source)
    if spec is None or spec.loader is None:
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_UPSTREAM_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upstream_module_with_local_model_base(
    *, eval_root: Path, model_root: Path, local_model_base: Path
) -> Any:
    previous_model_root = os.environ.get("CTRL_WORLD_MODEL_ROOT")
    os.environ["CTRL_WORLD_MODEL_ROOT"] = str(Path(local_model_base).resolve())
    try:
        return _load_upstream_module(eval_root=eval_root, model_root=model_root)
    finally:
        if previous_model_root is None:
            os.environ.pop("CTRL_WORLD_MODEL_ROOT", None)
        else:
            os.environ["CTRL_WORLD_MODEL_ROOT"] = previous_model_root


def _install_runtime_adapters(
    *,
    module: Any,
    dataset_dir: Path,
    data_stat: Path,
    interact_num: int,
    num_inference_steps: int,
    probe_id: str,
    seed_by_episode: Mapping[str, int],
) -> dict[str, Any]:
    original_wm_args = module.wm_args
    original_agent = module.agent
    original_reward_scorer = module.RewardScorer
    state: dict[str, Any] = {"agent": None, "reward": None, "dose": 0.0, "dose_context": None}

    def patched_wm_args(*, task_type: str) -> Any:
        runtime_args = original_wm_args(task_type=task_type)
        runtime_args.val_dataset_dir = str(dataset_dir)
        runtime_args.data_stat_path = str(data_stat)
        runtime_args.interact_num = int(interact_num)
        runtime_args.num_inference_steps = int(num_inference_steps)
        return runtime_args

    def close_dose() -> None:
        context = state.get("dose_context")
        if context is not None:
            context.__exit__(None, None, None)
            state["dose_context"] = None

    def patched_agent(runtime_args: Any) -> Any:
        if state["agent"] is None:
            rollout_agent = original_agent(runtime_args)
            original_get_traj_info = rollout_agent.get_traj_info

            def seeded_get_traj_info(episode_id: object, *args: object, **kwargs: object) -> object:
                key = str(episode_id)
                if key not in seed_by_episode:
                    raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_EPISODE_SEED_MISSING")
                module.set_seed(int(seed_by_episode[key]))
                return original_get_traj_info(episode_id, *args, **kwargs)

            rollout_agent.get_traj_info = seeded_get_traj_info
            state["agent"] = rollout_agent
        close_dose()
        context = ctrl_world_probe_context(
            model=state["agent"].model,
            probe_id=probe_id,
            dose=float(state["dose"]),
        )
        context.__enter__()
        state["dose_context"] = context
        return state["agent"]

    def patched_reward_scorer(checkpoint_path: str, device: object) -> Any:
        if state["reward"] is None:
            state["reward"] = original_reward_scorer(checkpoint_path, device)
        return state["reward"]

    module.wm_args = patched_wm_args
    module.agent = patched_agent
    module.RewardScorer = patched_reward_scorer
    state["close_dose"] = close_dose
    return state


def _validate_dataset_freeze(
    *, freeze_path: Path, dataset_root: Path, task_id: str, episode_ids: Sequence[str]
) -> None:
    freeze = _load_json(freeze_path)
    selected = freeze.get("selected_files")
    if not isinstance(selected, list):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_DATASET_FREEZE_INVALID")
    wanted_prefixes = {
        f"{task_id}/annotation/val/{episode_id}.json" for episode_id in episode_ids
    } | {
        f"{task_id}/videos/val/{episode_id}/{view}.mp4"
        for episode_id in episode_ids
        for view in range(3)
    }
    by_path = {str(row["path"]): row for row in selected if isinstance(row, Mapping)}
    if not wanted_prefixes.issubset(by_path):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_DATASET_FREEZE_COVERAGE_MISSING")
    for relative in sorted(wanted_prefixes):
        path = Path(dataset_root) / relative
        row = by_path[relative]
        if not path.is_file() or path.stat().st_size != int(row["size"]) or _sha256(path) != str(row["sha256"]):
            raise CtrlWorldPredictiveCampaignError(f"CTRL_WORLD_DATASET_FREEZE_MISMATCH:{relative}")


def _dose_tag(dose: float) -> str:
    value = f"{float(dose):+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = _load_json(path)
    assets = payload.get("assets")
    if not isinstance(assets, Mapping):
        raise CtrlWorldPredictiveCampaignError("CTRL_WORLD_ASSET_HASH_CACHE_INVALID")
    return {str(key): dict(value) for key, value in assets.items() if isinstance(value, Mapping)}


def _asset_identity(
    path: Path, *, cache: dict[str, Any], hash_large_assets: bool
) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    key = str(resolved)
    cached = cache.get(key)
    base = {
        "path": key,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    if isinstance(cached, Mapping):
        if (
            int(cached.get("size", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            identity = {**base, "sha256": str(cached["sha256"]), "sha256_state": "cached"}
            cache[key] = identity
            return identity
    if hash_large_assets:
        identity = {**base, "sha256": _sha256(resolved), "sha256_state": "computed"}
        cache[key] = identity
        return identity
    identity = {**base, "sha256": None, "sha256_state": "deferred"}
    cache[key] = identity
    return identity


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CtrlWorldPredictiveCampaignError(f"CTRL_WORLD_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--ctrl-world-model-root", type=Path)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--heldout-split", type=Path, required=True)
    parser.add_argument("--dataset-freeze", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--reward-ckpt", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), default="pilot")
    parser.add_argument("--doses", type=float, nargs="+")
    parser.add_argument("--interact-num", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    parser.add_argument("--asset-hash-cache", type=Path)
    parser.add_argument("--hash-large-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run_campaign(_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
