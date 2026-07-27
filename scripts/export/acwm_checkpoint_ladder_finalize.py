#!/usr/bin/env python3
"""Finalize a staged ACWM checkpoint ladder from independent official gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.execute.training_monitor_policy import select_best_official_checkpoint


class AcwmCheckpointLadderFinalizeError(RuntimeError):
    """Checkpoint ladder evidence was incomplete or inconsistent."""


def finalize_checkpoint_ladder(
    *,
    checkpoint_manifest: Path,
    official_gate_specs: Sequence[str],
    supplemental_checkpoint_specs: Sequence[str] = (),
    output_root: Path,
    environment: str,
    primitive: str,
    seed: int,
) -> dict[str, object]:
    """Select and retain the best passing official-gate checkpoint."""

    if not environment or not primitive or seed < 0:
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_IDENTITY_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_OUTPUT_EXISTS")
    checkpoint_path = Path(checkpoint_manifest).resolve(strict=True)
    checkpoint_doc = _load_json(checkpoint_path, "CHECKPOINT_FINALIZE_CHECKPOINT_MANIFEST_INVALID")
    if checkpoint_doc.get("artifact_type") != "wmloop-retained-training-checkpoint":
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_CHECKPOINT_MANIFEST_INVALID")
    ladder = _retained_ladder(checkpoint_doc)
    supplemental = _parse_path_specs(
        supplemental_checkpoint_specs,
        empty_allowed=True,
        invalid_code="CHECKPOINT_FINALIZE_CHECKPOINT_SPEC_INVALID",
    )
    overlap = set(ladder) & set(supplemental)
    if overlap:
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_CHECKPOINT_STEP_DUPLICATE")
    ladder.update(supplemental)
    parsed_specs = _parse_gate_specs(official_gate_specs)
    if set(parsed_specs) != set(ladder):
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_LADDER_GATE_SET_MISMATCH")

    records: list[dict[str, object]] = []
    gate_docs: dict[int, Mapping[str, Any]] = {}
    for step in sorted(parsed_specs):
        gate_path = parsed_specs[step]
        gate = _load_json(gate_path, "CHECKPOINT_FINALIZE_GATE_INVALID")
        _validate_gate_identity(gate, environment=environment, primitive=primitive, seed=seed)
        retained = ladder[step]
        candidate_checkpoint = Path(str(gate.get("candidate_checkpoint") or "")).resolve()
        if candidate_checkpoint != retained.resolve():
            raise AcwmCheckpointLadderFinalizeError(
                f"CHECKPOINT_FINALIZE_GATE_CHECKPOINT_MISMATCH:{step}"
            )
        expected_sha = str(gate.get("candidate_checkpoint_sha256") or "")
        if len(expected_sha) != 64 or _sha256(retained) != expected_sha:
            raise AcwmCheckpointLadderFinalizeError(f"CHECKPOINT_FINALIZE_SHA_MISMATCH:{step}")
        quality_gate = gate.get("official_quality_gate")
        if not isinstance(quality_gate, Mapping):
            raise AcwmCheckpointLadderFinalizeError(f"CHECKPOINT_FINALIZE_GATE_MISSING:{step}")
        records.append(
            {
                "checkpoint_step": step,
                "checkpoint_path": str(retained),
                "checkpoint_sha256": expected_sha,
                "official_manifest_path": str(gate_path),
                "official_quality_gate": dict(quality_gate),
            }
        )
        gate_docs[step] = gate

    selection = select_best_official_checkpoint(records)
    confirmation_passed = any(
        int(record["checkpoint_step"]) >= 800
        and isinstance(record.get("official_quality_gate"), Mapping)
        and record["official_quality_gate"].get("pass") is True
        for record in records
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-checkpoint-ladder-finalization",
        "state": "ready" if selection["state"] == "ready" and confirmation_passed else "checks_failed",
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "checkpoint_manifest": str(checkpoint_path),
        "evaluated_steps": sorted(parsed_specs),
        "selection": selection,
        "confirmation_passed": confirmation_passed,
        "confirmation_rule": "At least one independently evaluated checkpoint at relative step 800 or 1000 must pass the official quality gate.",
        "best_checkpoint_path": None,
        "best_checkpoint_sha256": None,
        "best_checkpoint_step": None,
        "selected_paired_videos": [],
        "claim_boundary": (
            "The finalizer selects only among checkpoints that pass the independent official 50-step quality gate. "
            "A 512-only pass is retained as quantitative backup but does not satisfy staged confirmation. "
            "This is checkpoint selection evidence, not a formal multi-seed causal claim."
        ),
    }
    return _write_bundle(destination, report, gate_docs=gate_docs)


def _retained_ladder(manifest: Mapping[str, Any]) -> dict[int, Path]:
    raw = manifest.get("checkpoint_ladder")
    if not isinstance(raw, list) or not raw:
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_LADDER_EMPTY")
    ladder: dict[int, Path] = {}
    for row in raw:
        if not isinstance(row, Mapping) or row.get("state") != "retained":
            continue
        try:
            step = int(row["relative_step"])
            path = Path(str(row["retained_path"])).resolve(strict=True)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_LADDER_INVALID") from exc
        if path.is_symlink() or not path.is_file() or step <= 0 or step in ladder:
            raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_LADDER_INVALID")
        ladder[step] = path
    if not ladder:
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_LADDER_EMPTY")
    return ladder


def _parse_gate_specs(specs: Sequence[str]) -> dict[int, Path]:
    return _parse_path_specs(
        specs,
        empty_allowed=False,
        invalid_code="CHECKPOINT_FINALIZE_GATE_SPEC_INVALID",
    )


def _parse_path_specs(
    specs: Sequence[str], *, empty_allowed: bool, invalid_code: str
) -> dict[int, Path]:
    if not specs and not empty_allowed:
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_GATES_EMPTY")
    parsed: dict[int, Path] = {}
    for spec in specs:
        step_raw, separator, path_raw = spec.partition("=")
        if not separator or not step_raw.isdigit() or not path_raw:
            raise AcwmCheckpointLadderFinalizeError(invalid_code)
        step = int(step_raw)
        if step <= 0 or step in parsed:
            raise AcwmCheckpointLadderFinalizeError(invalid_code)
        parsed[step] = Path(path_raw).resolve(strict=True)
    return parsed


def _validate_gate_identity(
    gate: Mapping[str, Any], *, environment: str, primitive: str, seed: int
) -> None:
    expected = {"environment": environment, "primitive": primitive, "seed": seed}
    if gate.get("artifact_type") != "wmloop-acwm-formal-visualization-export" or gate.get("state") != "ready":
        raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_GATE_NOT_READY")
    for key, value in expected.items():
        if gate.get(key) != value:
            raise AcwmCheckpointLadderFinalizeError(f"CHECKPOINT_FINALIZE_GATE_IDENTITY_MISMATCH:{key}")


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmCheckpointLadderFinalizeError(code) from exc
    if not isinstance(payload, Mapping):
        raise AcwmCheckpointLadderFinalizeError(code)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_bundle(
    destination: Path,
    report: dict[str, object],
    *,
    gate_docs: Mapping[int, Mapping[str, Any]],
) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        selection = report["selection"]
        best = selection.get("best_checkpoint") if isinstance(selection, Mapping) else None
        selected_videos: list[dict[str, object]] = []
        if isinstance(best, Mapping):
            step = int(best["checkpoint_step"])
            source = Path(str(best["checkpoint_path"])).resolve(strict=True)
            retained = temporary / "best_checkpoint.pt"
            storage_mode = _link_or_copy(source, retained)
            report["best_checkpoint_path"] = str(destination / "best_checkpoint.pt")
            report["best_checkpoint_sha256"] = str(best["checkpoint_sha256"])
            report["best_checkpoint_step"] = step
            report["best_checkpoint_storage_mode"] = storage_mode
            video_root = temporary / "paired_videos"
            for index, row in enumerate(gate_docs[step].get("paired_videos", [])):
                if not isinstance(row, Mapping):
                    continue
                source_video = Path(str(row.get("paired_video_path") or ""))
                if source_video.is_symlink() or not source_video.is_file():
                    raise AcwmCheckpointLadderFinalizeError("CHECKPOINT_FINALIZE_VIDEO_MISSING")
                video_root.mkdir(mode=0o700, exist_ok=True)
                target = video_root / f"sample_{index:02d}.mp4"
                mode = _link_or_copy(source_video, target)
                selected_videos.append(
                    {
                        "sample_index": row.get("sample_index", index),
                        "path": str(destination / "paired_videos" / target.name),
                        "storage_mode": mode,
                    }
                )
        report["selected_paired_videos"] = selected_videos
        _write_json(temporary / "checkpoint-ladder-finalization.json", report)
        (temporary / "checkpoint-ladder-finalization.md").write_text(_markdown(report), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-checkpoint-ladder-finalization-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "primitive": report["primitive"],
            "seed": report["seed"],
            "best_checkpoint_step": report["best_checkpoint_step"],
            "best_checkpoint_path": report["best_checkpoint_path"],
            "best_checkpoint_sha256": report["best_checkpoint_sha256"],
            "confirmation_passed": report["confirmation_passed"],
            "report_path": str(destination / "checkpoint-ladder-finalization.json"),
            "markdown_path": str(destination / "checkpoint-ladder-finalization.md"),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return {**report, "manifest": manifest}
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def _markdown(report: Mapping[str, object]) -> str:
    selection = report["selection"]
    best = selection.get("best_checkpoint") if isinstance(selection, Mapping) else None
    lines = [
        "# ACWM Checkpoint Ladder Finalization",
        "",
        f"- Environment: `{report['environment']}`",
        f"- Primitive: `{report['primitive']}`",
        f"- State: `{report['state']}`",
        f"- Best checkpoint: `{best.get('checkpoint_step') if isinstance(best, Mapping) else 'none'}`",
        f"- Confirmation passed: `{report['confirmation_passed']}`",
        f"- Stop requested: `{selection.get('stop_requested') if isinstance(selection, Mapping) else 'unknown'}`",
        f"- Extension allowed: `{selection.get('extension_allowed') if isinstance(selection, Mapping) else 'unknown'}`",
        "",
        "## Claim Boundary",
        "",
        str(report["claim_boundary"]),
        "",
    ]
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--official-gate", action="append", required=True, dest="official_gates")
    parser.add_argument("--checkpoint", action="append", default=[], dest="supplemental_checkpoints")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize_checkpoint_ladder(
        checkpoint_manifest=args.checkpoint_manifest,
        official_gate_specs=args.official_gates,
        supplemental_checkpoint_specs=args.supplemental_checkpoints,
        output_root=args.output_root,
        environment=args.environment,
        primitive=args.primitive,
        seed=args.seed,
    )
    print(json.dumps(report["manifest"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
