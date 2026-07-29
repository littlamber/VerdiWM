"""CPU-only Cosmos3 forward-dynamics instantiation smoke harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.adapters.cosmos3_hooks import (
    audit_cosmos3_forward_dynamics_hooks,
    materialize_action_json,
)


class Cosmos3ForwardDynamicsSmokeError(RuntimeError):
    """The Cosmos3 instance cannot satisfy its frozen CPU smoke contract."""


def run_cosmos3_forward_dynamics_smoke(
    *,
    cosmos3_root: Path,
    runtime_python: Path,
    runner_path: Path,
    dataset_root: Path,
    dataset_freeze_path: Path,
    split_path: Path,
    split_name: str,
    checkpoint_path: Path,
    config_file: Path,
    output_root: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_SMOKE_OUTPUT_EXISTS")
    root = Path(cosmos3_root).resolve(strict=True)
    python = Path(os.path.abspath(Path(runtime_python).expanduser()))
    runner = Path(runner_path).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    config = Path(config_file).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_RUNTIME_PYTHON_INVALID")
    if not runner.is_file() or not checkpoint.is_dir() or not config.is_file():
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_RUNTIME_ASSET_MISSING")
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_CHECKPOINT_INDEX_MISSING")

    split = _load_mapping(split_path, "COSMOS3_SPLIT_INVALID")
    try:
        validate_document("cosmos3_forward_dynamics_split", split)
    except ContractValidationError as exc:
        raise Cosmos3ForwardDynamicsSmokeError(f"COSMOS3_SPLIT_INVALID:{exc}") from exc
    if split_name not in {"dev", "accept"}:
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_SPLIT_NAME_INVALID")
    identity = dict(split[split_name][0])
    dataset_audit = _verify_dataset_freeze(dataset=dataset, freeze_path=dataset_freeze_path)
    hook_audit = audit_cosmos3_forward_dynamics_hooks(root)
    if hook_audit["state"] != "ready":
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_HOOK_AUDIT_BLOCKED")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        staged = temporary / "staged"
        command = [
                str(python),
                str(runner),
                "--repo-root",
                str(root),
                "--dataset-root",
                str(dataset),
                "--checkpoint-path",
                str(checkpoint),
                "--config-file",
                str(config),
                "--output-dir",
                str(staged),
                "--num-chunks",
                "1",
                "--start-index",
                str(identity["sample_index"]),
                "--seed",
                str(identity["seed"]),
                "--skip-run",
            ]
        try:
            subprocess.run(
                command,
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no subprocess output").strip()[-2000:]
            raise Cosmos3ForwardDynamicsSmokeError(f"COSMOS3_OFFICIAL_STAGING_FAILED:{detail}") from exc
        action_path = staged / "inputs/robotics_droid_action_chunk_00.json"
        plan_path = staged / "inputs/action_forward_dynamics_robotics_custom.jsonl"
        conditioning = staged / "inputs/robotics_droid_autoregressive_input_chunk_00.png"
        actions = json.loads(action_path.read_text(encoding="utf-8"))
        if len(actions) != 16 or any(not isinstance(row, list) or len(row) != 10 for row in actions):
            raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_ACTION_CONTRACT_MISMATCH")
        records = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 1 or records[0].get("model_mode") != "forward_dynamics":
            raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_MODEL_MODE_MISMATCH")
        if int(records[0].get("action_chunk_size", -1)) != 16 or not conditioning.is_file():
            raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_STAGED_INPUT_INVALID")

        zero_receipt = materialize_action_json(
            source=action_path,
            destination=temporary / "hooks/action-dose-zero.json",
            mode="action_conditioning_scale",
            dose=0.0,
        )
        balance_receipt = materialize_action_json(
            source=action_path,
            destination=temporary / "hooks/action-dimension-balanced.json",
            mode="action_dimension_balancing",
            blend=0.5,
            max_gain=4.0,
        )
        if zero_receipt["source_sha256"] != zero_receipt["output_sha256"]:
            raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_ZERO_DOSE_NOT_IDENTITY")
        report = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-forward-dynamics-smoke",
            "state": "ready",
            "model_family": "cosmos3",
            "model_mode": "forward_dynamics",
            "split": split_name,
            "identity": identity,
            "action_shape": [16, 10],
            "conditioning_frame": str(destination / conditioning.relative_to(temporary)),
            "dataset_audit": dataset_audit,
            "hook_audit": hook_audit,
            "zero_dose_receipt": zero_receipt,
            "action_dimension_balancing_receipt": balance_receipt,
            "assets": {
                "cosmos3_root": str(root),
                "runtime_python": str(python),
                "runner": str(runner),
                "checkpoint": str(checkpoint),
                "config_file": str(config),
                "config_sha256": _sha256(config),
            },
            "side_effects": {
                "gpu_execution_started": False,
                "training_started": False,
                "prediction_generated": False,
                "source_repo_mutated": False,
            },
            "claim_boundary": "This smoke validates ACWM forward-dynamics wiring, frozen data identity, and runtime hook materialization only. It is not a model-quality or transfer result.",
        }
        (temporary / "smoke.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-forward-dynamics-smoke-manifest",
            "state": "ready",
            "model_mode": "forward_dynamics",
            "split": split_name,
            "sample_index": identity["sample_index"],
            "seed": identity["seed"],
            "action_shape": [16, 10],
            "report_path": str(destination / "smoke.json"),
            "claim_boundary": report["claim_boundary"],
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verify_dataset_freeze(*, dataset: Path, freeze_path: Path) -> dict[str, object]:
    freeze = _load_mapping(freeze_path, "COSMOS3_DATASET_FREEZE_INVALID")
    rows = freeze.get("selected_files")
    if not isinstance(rows, list) or len(rows) != int(freeze.get("selected_file_count", -1)):
        raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_DATASET_FREEZE_INVALID")
    for row in rows:
        if not isinstance(row, Mapping):
            raise Cosmos3ForwardDynamicsSmokeError("COSMOS3_DATASET_FREEZE_INVALID")
        path = dataset / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["size"]) or _sha256(path) != str(row["sha256"]):
            raise Cosmos3ForwardDynamicsSmokeError(f"COSMOS3_DATASET_FREEZE_MISMATCH:{row['path']}")
    return {"state": "verified", "freeze_id": freeze["freeze_id"], "file_count": len(rows)}


def _load_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Cosmos3ForwardDynamicsSmokeError(code) from exc
    if not isinstance(payload, Mapping):
        raise Cosmos3ForwardDynamicsSmokeError(code)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos3-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-freeze", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("dev", "accept"), default="dev")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_cosmos3_forward_dynamics_smoke(
        cosmos3_root=args.cosmos3_root,
        runtime_python=args.runtime_python,
        runner_path=args.runner_path,
        dataset_root=args.dataset_root,
        dataset_freeze_path=args.dataset_freeze,
        split_path=args.split,
        split_name=args.split_name,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
