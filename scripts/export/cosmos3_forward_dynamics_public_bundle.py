#!/usr/bin/env python3
"""Export a path-safe Cosmos3 forward-dynamics instantiation example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


class Cosmos3PublicBundleError(ValueError):
    """The Cosmos3 instantiation evidence cannot be published safely."""


def export_cosmos3_forward_dynamics_public_bundle(
    *,
    smoke_root: Path,
    instance_audit_root: Path,
    capability_root: Path,
    output_root: Path,
    gpu_runtime_root: Path | None = None,
) -> dict[str, object]:
    smoke = _load(Path(smoke_root) / "smoke.json")
    audit = _load(Path(instance_audit_root) / "backbone-instantiation.json")
    capability = _load(Path(capability_root) / "backbone-capability-matrix.json")
    gpu_runtime = (
        _load(Path(gpu_runtime_root) / "runtime-receipt.json")
        if gpu_runtime_root is not None
        else None
    )
    if smoke.get("state") != "ready" or smoke.get("model_mode") != "forward_dynamics":
        raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_SMOKE_INVALID")
    if audit.get("instance_id") != capability.get("instance_id"):
        raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_INSTANCE_MISMATCH")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_OUTPUT_EXISTS")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        smoke_summary = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-forward-dynamics-public-smoke-summary",
            "state": smoke["state"],
            "model_family": smoke["model_family"],
            "model_mode": smoke["model_mode"],
            "split": smoke["split"],
            "identity": smoke["identity"],
            "action_shape": smoke["action_shape"],
            "dataset_audit": smoke["dataset_audit"],
            "available_hooks": smoke["hook_audit"]["available_hooks"],
            "zero_dose_byte_identity": smoke["zero_dose_receipt"]["zero_dose_byte_identity"],
            "action_dimension_balancing_shape": smoke["action_dimension_balancing_receipt"]["shape"],
            "side_effects": smoke["side_effects"],
            "claim_boundary": smoke["claim_boundary"],
        }
        capability_summary = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-forward-dynamics-public-capability-summary",
            "state": capability["state"],
            "campaign_state": capability["campaign_state"],
            "available_hooks": capability["capability_summary"]["available_hooks"],
            "primitive_count": capability["capability_summary"]["primitive_count"],
            "eligible_primitive_count": capability["capability_summary"]["eligible_primitive_count"],
            "blocked_primitive_count": capability["capability_summary"]["blocked_primitive_count"],
            "eligible_primitives": [
                row["primitive"]
                for row in capability["primitive_matrix"]
                if row["status"] == "eligible_for_instance_canary"
            ],
            "closed_loop_instance_ready": audit["closed_loop_instance_ready"],
            "formal_verdict_instance_ready": audit["formal_verdict_instance_ready"],
            "instance_formal_launch_allowed": audit["instance_formal_launch_allowed"],
        }
        bundle = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-forward-dynamics-public-instance-bundle",
            "state": "pilot_draft",
            "cpu_smoke_state": smoke_summary["state"],
            "gpu_runtime_state": gpu_runtime["state"] if gpu_runtime is not None else "not_included",
            "model_mode": "forward_dynamics",
            "eligible_primitive_count": capability_summary["eligible_primitive_count"],
            "formal_launch_allowed": capability_summary["instance_formal_launch_allowed"],
            "model_quality_evidence_included": False,
            "transfer_evidence_included": False,
            "claim_boundary": (
                gpu_runtime["claim_boundary"] if gpu_runtime is not None else smoke_summary["claim_boundary"]
            ),
        }
        if gpu_runtime is not None:
            if gpu_runtime.get("model_mode") != "forward_dynamics" or gpu_runtime.get("state") != "ready":
                raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_GPU_RUNTIME_INVALID")
            public_gpu_runtime = {
                "schema_version": 1,
                "artifact_type": "verdiwm-cosmos3-forward-dynamics-public-gpu-runtime-summary",
                "state": gpu_runtime["state"],
                "model_mode": gpu_runtime["model_mode"],
                "identity": gpu_runtime["identity"],
                "action_shape": gpu_runtime["action_shape"],
                "physical_gpu": gpu_runtime["physical_gpu"],
                "video": gpu_runtime["video"],
                "runtime_seconds": gpu_runtime["runtime_seconds"],
                "artifact_sha256": gpu_runtime["sha256"],
                "claim_boundary": gpu_runtime["claim_boundary"],
            }
            _write(temporary / "gpu-runtime-summary.json", public_gpu_runtime)
        _write(temporary / "smoke-summary.json", smoke_summary)
        _write(temporary / "capability-summary.json", capability_summary)
        _write(temporary / "bundle.json", bundle)
        (temporary / "README.md").write_text(_readme(bundle, capability_summary), encoding="utf-8")
        _assert_path_safe(temporary)
        _write_manifest(temporary)
        os.replace(temporary, destination)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _readme(bundle: Mapping[str, object], capability: Mapping[str, object]) -> str:
    eligible = ", ".join(f"`{name}`" for name in capability["eligible_primitives"])
    return "\n".join(
        [
            "# Cosmos3 Forward-Dynamics Instance",
            "",
            "This path-safe bundle records Cosmos3-Nano as an ACWM forward-dynamics backbone.",
            "It validates contracts, hook materialization, and (when included) one official GPU runtime execution, not generated-video quality or transfer.",
            "",
            f"- State: `{bundle['state']}`",
            f"- Model mode: `{bundle['model_mode']}`",
            f"- GPU runtime state: `{bundle['gpu_runtime_state']}`",
            f"- Formal launch allowed: `{str(bundle['formal_launch_allowed']).lower()}`",
            f"- Explicitly bound primitives: {eligible}",
            "",
            "The smoke verifies the official 16 x 10 DROID action shape, frozen sample files, first-frame conditioning, H1-H5 anchors, and zero-dose byte identity.",
            "External checkpoints, datasets, videos, and local filesystem paths are intentionally excluded.",
            "",
        ]
    )


def _load(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_INPUT_INVALID")
    return payload


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _assert_path_safe(root: Path) -> None:
    host_prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            text = path.read_text(encoding="utf-8")
            if any(prefix in text for prefix in host_prefixes):
                raise Cosmos3PublicBundleError("COSMOS3_PUBLIC_LOCAL_PATH_LEAK")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--instance-audit-root", type=Path, required=True)
    parser.add_argument("--capability-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-runtime-root", type=Path)
    args = parser.parse_args(argv)
    bundle = export_cosmos3_forward_dynamics_public_bundle(
        smoke_root=args.smoke_root,
        instance_audit_root=args.instance_audit_root,
        capability_root=args.capability_root,
        output_root=args.output_root,
        gpu_runtime_root=args.gpu_runtime_root,
    )
    print(json.dumps(bundle, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
