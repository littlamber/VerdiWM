#!/usr/bin/env python3
"""Retain official-gate-passing ACWM visual candidates with explicit claim tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES


class ShowcaseRegistryError(RuntimeError):
    """A registry source does not satisfy the retention contract."""


def run_registry(*, output_root: Path, entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_OUTPUT_EXISTS:{destination}")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        video_root = temporary / "videos"
        video_root.mkdir(parents=True, mode=0o700)
        records: list[dict[str, object]] = []
        sources: dict[str, dict[str, object]] = {}
        for ordinal, entry in enumerate(entries, start=1):
            gate_path = Path(str(entry.get("gate_manifest") or "")).resolve()
            if gate_path.is_symlink() or not gate_path.is_file():
                raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_GATE_INVALID:{gate_path}")
            gate = _load_json(gate_path)
            quality_gate = gate.get("official_quality_gate")
            if not isinstance(quality_gate, Mapping) or quality_gate.get("pass") is not True:
                raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_GATE_NOT_PASSING:{gate_path}")
            environment = str(gate.get("environment") or "")
            primitive = str(gate.get("primitive") or "")
            if not environment or not primitive:
                raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_IDENTITY_MISSING:{gate_path}")
            sample_index = int(entry.get("sample_index") or 0)
            measurement = _measurement(gate, sample_index)
            source_video = Path(str(measurement.get("selected_video_path") or "")).resolve()
            if source_video.is_symlink() or not source_video.is_file():
                raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_VIDEO_INVALID:{source_video}")
            invalidated = primitive in INVALIDATED_QUALITY_PRIMITIVES
            tier = (
                "C_gate_pass_method_invalidated_audit_only"
                if invalidated
                else str(entry.get("tier") or "B_gate_pass_visual_pending")
            )
            note = str(entry.get("note") or "Official gate passed; visual strength remains to be judged.")
            target = video_root / (
                f"{ordinal:02d}_{environment}_{primitive}_sample_{sample_index:02d}_gt_baseline_ours.mp4"
            )
            shutil.copy2(source_video, target)
            checkpoint_sha = gate.get("candidate_checkpoint_sha256")
            runtime_sha = gate.get("candidate_runtime_sha256")
            record = {
                "id": f"candidate_{ordinal:02d}",
                "tier": tier,
                "claim_status": (
                    "official_gate_pass_method_invalidated_audit_only"
                    if invalidated
                    else "official_gate_pass_qualitative_candidate"
                ),
                "method_invalidated": invalidated,
                "method_invalidation_reason": (
                    "auxiliary_loss_uses_detached_latents_and_does_not_update_trainable_model_parameters"
                    if invalidated
                    else None
                ),
                "visual_status": "pending_human_inspection",
                "note": note,
                "environment": environment,
                "primitive": primitive,
                "seed": int(gate.get("seed") or gate.get("eval_seed") or 0),
                "sample_index": sample_index,
                "official_quality_gate": dict(quality_gate),
                "candidate_checkpoint_sha256": checkpoint_sha,
                "candidate_runtime_sha256": runtime_sha,
                "source_manifest": str(gate_path),
                "source_manifest_sha256": _sha256(gate_path),
                "source_video_path": str(source_video),
                "retained_video_path": str(destination / "videos" / target.name),
                "retained_video_sha256": _sha256(target),
                "retained_video_size_bytes": target.stat().st_size,
            }
            records.append(record)
            source_key = str(gate_path)
            sources[source_key] = {
                "manifest": source_key,
                "manifest_sha256": record["source_manifest_sha256"],
                "environment": environment,
                "primitive": primitive,
                "official_gate_pass": True,
            }

        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-validated-showcase-candidate-registry",
            "state": "ready",
            "retention_policy": {
                "retain_gate_pass": True,
                "primary_display_requires_human_visual_review": True,
                "gt_injection_allowed": False,
                "layout": "GT|baseline_prediction|ours_prediction",
                "metric_claim_boundary": "Official ACWM 50-step gate remains authoritative.",
                "invalidated_method_policy": "Retain metric-passing artifacts for audit, but exclude them from method claims and automatic routing.",
            },
            "source_artifacts": list(sources.values()),
            "records": records,
        }
        _write_json(temporary / "manifest.json", report)
        (temporary / "README.md").write_text(_markdown(records), encoding="utf-8")
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _measurement(gate: Mapping[str, object], sample_index: int) -> Mapping[str, object]:
    hard = gate.get("hard_case_visualization")
    if isinstance(hard, Mapping):
        candidates = hard.get("selected")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, Mapping) and _sample_index(candidate) == sample_index:
                    return candidate
        all_measurements = hard.get("all_measurements")
        if isinstance(all_measurements, list):
            for candidate in all_measurements:
                if isinstance(candidate, Mapping) and _sample_index(candidate) == sample_index:
                    selected = dict(candidate)
                    selected["selected_video_path"] = candidate.get("paired_video_path")
                    return selected

    paired_videos = gate.get("paired_videos")
    if isinstance(paired_videos, list):
        for candidate in paired_videos:
            if isinstance(candidate, Mapping) and _sample_index(candidate) == sample_index:
                selected = dict(candidate)
                selected["selected_video_path"] = candidate.get("paired_video_path")
                return selected
    raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_SAMPLE_MISSING:{sample_index}")


def _sample_index(candidate: Mapping[str, object]) -> int:
    value = candidate.get("sample_index")
    return int(value) if value is not None else -1


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShowcaseRegistryError(f"SHOWCASE_REGISTRY_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# ACWM Validated Showcase Candidates",
        "",
        "All listed records passed the official 50-step PSNR/SSIM/MSE/masked-MSE gate. "
        "Tier B means the record is retained for fallback use while visual strength is still pending human review.",
        "",
        "| ID | Tier | Environment | Primitive | Sample | PSNR | SSIM | MSE | Masked-MSE | Video |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        gate = record["official_quality_gate"]
        assert isinstance(gate, Mapping)
        delta = gate["delta_candidate_minus_baseline"]
        assert isinstance(delta, Mapping)
        lines.append(
            f"| {record['id']} | {record['tier']} | {record['environment']} | {record['primitive']} | "
            f"{record['sample_index']} | {float(delta['psnr']):+.4f} | {float(delta['ssim']):+.6f} | "
            f"{float(delta['mse']):+.8f} | {float(delta['masked_mse']):+.8f} | "
            f"{Path(str(record['retained_video_path'])).name} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--entry", action="append", required=True, help="gate_manifest|sample_index|tier|note")
    args = parser.parse_args(argv)
    entries = []
    for raw in args.entry:
        parts = raw.split("|", 3)
        if len(parts) != 4:
            raise SystemExit("--entry must be gate_manifest|sample_index|tier|note")
        entries.append({"gate_manifest": parts[0], "sample_index": int(parts[1]), "tier": parts[2], "note": parts[3]})
    report = run_registry(output_root=args.output_root, entries=entries)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
