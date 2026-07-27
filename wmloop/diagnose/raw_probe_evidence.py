"""Build measured M1 raw-probe evidence bundles from explicit inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.probes.action_following import no_action_delta_psnr, per_frame_inverse_dynamics_accuracy
from wmloop.diagnose.probes.appearance import low_motion_ssim
from wmloop.diagnose.probes.ood_profile import build_ood_profile


class RawProbeEvidenceError(RuntimeError):
    """A raw-probe evidence bundle could not be produced safely."""


def generate_raw_probe_evidence_report(
    *,
    measurements_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Compute probe metrics from measured inputs and write a report bundle."""

    measurements = _load_measurements(measurements_path)
    source_kind = _source_kind(measurements)
    records = [_evidence_record(item) for item in measurements["records"]]
    ready = bool(records and all(record["state"] == "ready" for record in records))
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-raw-probe-evidence-report",
        "state": "ready" if source_kind == "measured" and ready else "smoke" if source_kind == "fixture" and ready else "incomplete",
        "source_kind": source_kind,
        "measurements_path": str(Path(measurements_path).resolve()),
        "record_count": len(records),
        "ready_count": sum(1 for record in records if record["state"] == "ready"),
        "records": records,
        "limitations": [
            "This report only validates and aggregates supplied raw measurements; it does not run model inference itself.",
        ],
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _load_measurements(path: Path) -> Mapping[str, Any]:
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_INPUT_INVALID")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_INPUT_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-raw-probe-measurement-input"
        or not isinstance(payload.get("records"), list)
    ):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_INPUT_INVALID")
    return payload


def _source_kind(payload: Mapping[str, Any]) -> str:
    value = payload.get("source_kind", "measured")
    if value not in {"measured", "fixture"}:
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_SOURCE_KIND_INVALID")
    return str(value)


def _evidence_record(record: Mapping[str, Any]) -> dict[str, object]:
    environment = _string_field(record, "environment")
    split = _string_field(record, "split")
    appearance = _appearance(record.get("appearance_drift"))
    action = _action_following(record.get("action_following"))
    ood = _ood_profile(record.get("ood_profile"))
    state = "ready" if appearance["state"] == action["state"] == ood["state"] == "measured" else "incomplete"
    return {
        "environment": environment,
        "split": split,
        "state": state,
        "appearance_drift": appearance,
        "action_following": action,
        "ood_profile": ood,
    }


def _appearance(raw: object) -> dict[str, object]:
    if raw is None:
        return {"state": "missing", "reason": "appearance_drift measurement missing"}
    if not isinstance(raw, Mapping):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_APPEARANCE_INVALID")
    motion = _number_list(raw.get("motion_magnitudes"), "RAW_PROBE_EVIDENCE_APPEARANCE_INVALID")
    ssim = _number_list(raw.get("ssim_scores"), "RAW_PROBE_EVIDENCE_APPEARANCE_INVALID")
    fraction = raw.get("low_motion_fraction", 0.25)
    if not _finite_number(fraction):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_APPEARANCE_INVALID")
    value = low_motion_ssim(motion_magnitudes=motion, ssim_scores=ssim, fraction=float(fraction))
    return {
        "state": "measured",
        "low_motion_ssim_64": value,
        "sample_count": len(ssim),
        "low_motion_fraction": float(fraction),
        "evidence_refs": _cas_refs(raw.get("evidence_refs")),
    }


def _action_following(raw: object) -> dict[str, object]:
    if raw is None:
        return {"state": "missing", "reason": "action_following measurement missing"}
    if not isinstance(raw, Mapping):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_ACTION_INVALID")
    predicted = _matrix(raw.get("predicted_actions"), "RAW_PROBE_EVIDENCE_ACTION_INVALID")
    target = _matrix(raw.get("target_actions"), "RAW_PROBE_EVIDENCE_ACTION_INVALID")
    tolerance = raw.get("tolerance", 1e-3)
    conditioned = raw.get("action_conditioned_psnr")
    no_action = raw.get("no_action_psnr")
    inverse_r2 = raw.get("inverse_dynamics_r2")
    low_confidence = raw.get("low_confidence")
    if not _finite_number(tolerance) or not _finite_number(conditioned) or not _finite_number(no_action):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_ACTION_INVALID")
    if inverse_r2 is not None and not _finite_number(inverse_r2):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_ACTION_INVALID")
    if not isinstance(low_confidence, bool):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_ACTION_INVALID")
    accuracy = per_frame_inverse_dynamics_accuracy(predicted_actions=predicted, target_actions=target, tolerance=float(tolerance))
    delta = no_action_delta_psnr(action_conditioned_psnr=float(conditioned), no_action_psnr=float(no_action))
    return {
        "state": "measured",
        "inv_dyn_acc_perframe": accuracy,
        "no_action_delta_psnr": delta,
        "inverse_dynamics_r2": float(inverse_r2) if inverse_r2 is not None else None,
        "low_confidence": low_confidence,
        "frame_count": len(target),
        "tolerance": float(tolerance),
        "evidence_refs": _cas_refs(raw.get("evidence_refs")),
    }


def _ood_profile(raw: object) -> dict[str, object]:
    if raw is None:
        return {"state": "missing", "reason": "ood_profile measurement missing"}
    if not isinstance(raw, Mapping):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_OOD_INVALID")
    ind_auc = raw.get("ind_auc")
    by_condition = raw.get("ood_auc_by_condition")
    if not _finite_number(ind_auc) or not isinstance(by_condition, Mapping):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_OOD_INVALID")
    parsed_conditions: dict[str, float] = {}
    for key, value in by_condition.items():
        if not isinstance(key, str) or not key or not _finite_number(value):
            raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_OOD_INVALID")
        parsed_conditions[key] = float(value)
    profile = build_ood_profile(ind_auc=float(ind_auc), ood_auc_by_condition=parsed_conditions)
    return {
        "state": "measured",
        **profile,
        "condition_count": len(by_condition),
        "evidence_refs": _cas_refs(raw.get("evidence_refs")),
    }


def _string_field(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_RECORD_INVALID")
    return value


def _number_list(value: object, code: str) -> list[float]:
    if not isinstance(value, list) or not value or any(not _finite_number(item) for item in value):
        raise RawProbeEvidenceError(code)
    return [float(item) for item in value]


def _matrix(value: object, code: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise RawProbeEvidenceError(code)
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or not row or any(not _finite_number(item) for item in row):
            raise RawProbeEvidenceError(code)
        rows.append([float(item) for item in row])
    return rows


def _cas_refs(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.startswith("cas://sha256/") for item in value):
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_REFS_INVALID")
    return list(value)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Raw-Probe Evidence Report",
        "",
        f"State: `{report['state']}`",
        f"Ready records: `{report['ready_count']}/{report['record_count']}`",
        "",
        "| Environment | Split | State | Low-Motion SSIM | No-Action Delta | OOD Gap |",
        "|:--|:--|:--|--:|--:|--:|",
    ]
    for record in report["records"]:
        appearance = record["appearance_drift"]
        action = record["action_following"]
        ood = record["ood_profile"]
        lines.append(
            "| {env} | {split} | {state} | {appearance} | {action} | {ood} |".format(
                env=record["environment"],
                split=record["split"],
                state=record["state"],
                appearance=_display(appearance.get("low_motion_ssim_64")),
                action=_display(action.get("no_action_delta_psnr")),
                ood=_display(ood.get("gap")),
            )
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _display(value: object) -> str:
    return f"{float(value):.4f}" if _finite_number(value) else "n/a"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    markdown: str,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "raw-probe-evidence.json", report_bytes)
        _write_bytes_atomic(temporary / "raw-probe-evidence.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("raw_probe_evidence_json", report_bytes, "application/json"),
                ("raw_probe_evidence_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-raw-probe-evidence-manifest",
            "state": report["state"],
            "report_path": str(destination / "raw-probe-evidence.json"),
            "markdown_path": str(destination / "raw-probe-evidence.md"),
            "cas_refs": cas_refs,
            "record_count": report["record_count"],
            "ready_count": report["ready_count"],
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RawProbeEvidenceError("RAW_PROBE_EVIDENCE_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate raw-probe evidence report")
    generate.add_argument("--measurements", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_raw_probe_evidence_report(
            measurements_path=args.measurements,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
