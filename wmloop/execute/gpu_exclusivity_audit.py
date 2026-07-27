"""Audit GPU exclusivity before launching GPU-backed work."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class GpuExclusivityAuditError(RuntimeError):
    """GPU exclusivity could not be proven safely."""


Runner = Callable[..., subprocess.CompletedProcess[Any]]
ProcessCommandProvider = Callable[[int], str | None]


def verify_gpu_exclusivity_ready(
    manifest_path: Path | None,
    *,
    gpu_index: int | None = None,
    requested_gpus: Sequence[int] | None = None,
    max_age_seconds: float | None = None,
) -> dict[str, object]:
    """Verify a ready GPU exclusivity audit for a launch about to use GPUs."""

    if manifest_path is None:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_REQUIRED")
    required_gpus = _required_gpu_set(gpu_index=gpu_index, requested_gpus=requested_gpus)
    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest_bytes = manifest_source.read_bytes()
    manifest = _json_mapping_from_bytes(manifest_bytes, "GPU_EXCLUSIVITY_AUDIT_MANIFEST_INVALID")
    if manifest.get("artifact_type") != "wmloop-gpu-exclusivity-audit-manifest" or manifest.get("schema_version") != 1:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_MANIFEST_INVALID")
    if manifest.get("state") != "ready" or manifest.get("gpu_exclusivity_ready") is not True:
        raise GpuExclusivityAuditError(f"GPU_EXCLUSIVITY_NOT_READY:{manifest.get('state')}:{manifest.get('gpu_exclusivity_ready')}")
    if manifest.get("blocked_requested_gpu_count") not in (0, 0.0):
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_HAS_BLOCKED_GPUS")
    audited_gpus = _manifest_requested_gpus(manifest)
    missing = [gpu for gpu in required_gpus if gpu not in audited_gpus]
    if missing:
        raise GpuExclusivityAuditError(f"GPU_EXCLUSIVITY_AUDIT_GPU_MISSING:{','.join(str(value) for value in missing)}")
    side_effects = manifest.get("side_effects")
    if not isinstance(side_effects, Mapping) or any(
        side_effects.get(key) is not False
        for key in ("gpu_execution_started", "process_kill_attempted", "active_training_mutated")
    ):
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_SIDE_EFFECTS_INVALID")

    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_REPORT_PATH_INVALID")
    report_source = Path(report_path).resolve(strict=True)
    report_bytes = report_source.read_bytes()
    report = _json_mapping_from_bytes(report_bytes, "GPU_EXCLUSIVITY_AUDIT_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-gpu-exclusivity-audit" or report.get("state") != "ready":
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_REPORT_INVALID")
    _verify_report_gpu_readiness(report, required_gpus)
    age_seconds = _verify_freshness(report, max_age_seconds)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-gpu-exclusivity-authorization",
        "state": "ready",
        "manifest_path": str(manifest_source),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "report_path": str(report_source),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "audited_requested_gpus": audited_gpus,
        "required_gpus": list(required_gpus),
        "max_age_seconds": max_age_seconds,
        "age_seconds": age_seconds,
        "cas_refs": manifest.get("cas_refs", {}),
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
    }


def run_gpu_exclusivity_audit(
    *,
    output_root: Path,
    requested_gpus: Sequence[int] | None = None,
    allowed_pids: Sequence[int] = (),
    memory_used_threshold_mib: int = 1024,
    utilization_threshold_percent: int = 10,
    nvidia_smi_timeout_seconds: float = 10.0,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    runner: Runner | None = None,
    process_command_provider: ProcessCommandProvider | None = None,
) -> dict[str, object]:
    """Write a CAS-backed, read-only audit for requested GPU launch slots."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_OUTPUT_EXISTS")
    if memory_used_threshold_mib < 0:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_MEMORY_THRESHOLD_INVALID")
    if utilization_threshold_percent < 0:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_UTILIZATION_THRESHOLD_INVALID")
    requested = _normalize_gpu_indices(requested_gpus)
    allowlist = frozenset(_normalize_pid_list(allowed_pids))
    command_runner = runner or subprocess.run
    snapshot = _collect_snapshot(
        runner=command_runner,
        timeout_seconds=nvidia_smi_timeout_seconds,
        process_command_provider=process_command_provider or _read_process_command,
    )
    policy = {
        "requested_gpus": list(requested),
        "requested_scope": "all_observed_gpus" if requested_gpus is None else "explicit",
        "allowed_pids": sorted(allowlist),
        "memory_used_threshold_mib": memory_used_threshold_mib,
        "utilization_threshold_percent": utilization_threshold_percent,
        "nvidia_smi_timeout_seconds": nvidia_smi_timeout_seconds,
    }
    report = _build_report(
        snapshot=snapshot,
        policy=policy,
        requested=requested,
        allowed_pids=allowlist,
        memory_used_threshold_mib=memory_used_threshold_mib,
        utilization_threshold_percent=utilization_threshold_percent,
    )
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _normalize_gpu_indices(indices: Sequence[int] | None) -> tuple[int, ...]:
    if indices is None:
        return ()
    normalized: list[int] = []
    for index in indices:
        value = int(index)
        if value < 0:
            raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_GPU_INDEX_INVALID")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_pid_list(pids: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for pid in pids:
        value = int(pid)
        if value <= 0:
            raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_PID_INVALID")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _required_gpu_set(*, gpu_index: int | None, requested_gpus: Sequence[int] | None) -> tuple[int, ...]:
    values: list[int] = []
    if gpu_index is not None:
        values.append(gpu_index)
    if requested_gpus is not None:
        values.extend(int(value) for value in requested_gpus)
    required = _normalize_gpu_indices(tuple(values))
    if not required:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_REQUIRED_GPUS_EMPTY")
    return required


def _json_mapping_from_bytes(payload: bytes, code: str) -> Mapping[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GpuExclusivityAuditError(code) from exc
    if not isinstance(document, Mapping):
        raise GpuExclusivityAuditError(code)
    return document


def _manifest_requested_gpus(manifest: Mapping[str, object]) -> list[int]:
    requested = manifest.get("requested_gpus")
    if not isinstance(requested, list):
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_REQUESTED_GPUS_INVALID")
    normalized = _normalize_gpu_indices(tuple(int(value) for value in requested))
    return list(normalized)


def _verify_report_gpu_readiness(report: Mapping[str, object], required_gpus: tuple[int, ...]) -> None:
    gpus = _as_mapping_sequence(report.get("gpus"))
    by_index = {int(gpu["index"]): gpu for gpu in gpus if isinstance(gpu.get("index"), int)}
    for gpu_index in required_gpus:
        gpu = by_index.get(gpu_index)
        if gpu is None:
            raise GpuExclusivityAuditError(f"GPU_EXCLUSIVITY_AUDIT_REPORT_GPU_MISSING:{gpu_index}")
        if gpu.get("requested") is not True or gpu.get("exclusive_ready") is not True:
            raise GpuExclusivityAuditError(f"GPU_EXCLUSIVITY_AUDIT_REPORT_GPU_NOT_READY:{gpu_index}")


def _verify_freshness(report: Mapping[str, object], max_age_seconds: float | None) -> float | None:
    if max_age_seconds is None:
        return None
    if max_age_seconds < 0:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_MAX_AGE_INVALID")
    timestamp = report.get("timestamp_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_TIMESTAMP_MISSING")
    try:
        if timestamp.endswith("Z"):
            observed = _dt.datetime.fromisoformat(timestamp[:-1] + "+00:00")
        else:
            observed = _dt.datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_TIMESTAMP_INVALID") from exc
    if observed.tzinfo is None:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_TIMESTAMP_INVALID")
    age = (_dt.datetime.now(tz=_dt.timezone.utc) - observed.astimezone(_dt.timezone.utc)).total_seconds()
    if age < -1.0:
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_TIMESTAMP_IN_FUTURE")
    if age > max_age_seconds:
        raise GpuExclusivityAuditError(f"GPU_EXCLUSIVITY_AUDIT_STALE:{age:.3f}:{max_age_seconds:.3f}")
    return age


def _collect_snapshot(
    *,
    runner: Runner,
    timeout_seconds: float,
    process_command_provider: ProcessCommandProvider,
) -> dict[str, object]:
    gpu_command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    apps_command = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    gpu_result = _run_command(gpu_command, runner=runner, timeout_seconds=timeout_seconds)
    commands = {"gpu_inventory": gpu_result["receipt"]}
    blockers: list[dict[str, object]] = []
    if gpu_result["returncode"] != 0:
        blockers.append(
            {
                "surface": "nvidia_smi",
                "reason": "NVIDIA_SMI_GPU_QUERY_FAILED",
                "returncode": gpu_result["returncode"],
                "stderr_tail": gpu_result["stderr_tail"],
            }
        )
        return {
            "state": "blocked",
            "timestamp_utc": _utc_now(),
            "commands": commands,
            "gpus": [],
            "blockers": blockers,
        }
    gpus, gpu_parse_blockers = _parse_gpu_inventory(_as_text(gpu_result["stdout"]))
    blockers.extend(gpu_parse_blockers)
    if not gpus:
        blockers.append({"surface": "nvidia_smi", "reason": "NVIDIA_SMI_NO_GPUS_OBSERVED"})
    apps_result = _run_command(apps_command, runner=runner, timeout_seconds=timeout_seconds)
    commands["compute_apps"] = apps_result["receipt"]
    if apps_result["returncode"] != 0:
        blockers.append(
            {
                "surface": "nvidia_smi",
                "reason": "NVIDIA_SMI_COMPUTE_APPS_QUERY_FAILED",
                "returncode": apps_result["returncode"],
                "stderr_tail": apps_result["stderr_tail"],
            }
        )
    else:
        apps, app_parse_blockers = _parse_compute_apps(
            _as_text(apps_result["stdout"]),
            process_command_provider=process_command_provider,
        )
        blockers.extend(app_parse_blockers)
        _attach_apps(gpus, apps, blockers)
    return {
        "state": "ready" if not blockers else "blocked",
        "timestamp_utc": _utc_now(),
        "commands": commands,
        "gpus": gpus,
        "blockers": blockers,
    }


def _run_command(
    command: Sequence[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> dict[str, object]:
    try:
        completed = runner(
            tuple(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _as_text(exc.stderr)
        return {
            "returncode": -1,
            "stdout": _as_text(exc.stdout),
            "stderr": stderr,
            "stderr_tail": _tail(stderr),
            "receipt": {
                "argv": list(command),
                "returncode": -1,
                "timed_out": True,
                "stdout_size": len(_as_text(exc.stdout).encode("utf-8")),
                "stderr_size": len(stderr.encode("utf-8")),
                "stderr_tail": _tail(stderr),
            },
        }
    except OSError as exc:
        stderr = f"{exc.__class__.__name__}: {exc}"
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": stderr,
            "stderr_tail": _tail(stderr),
            "receipt": {
                "argv": list(command),
                "returncode": -1,
                "timed_out": False,
                "stdout_size": 0,
                "stderr_size": len(stderr.encode("utf-8")),
                "stderr_tail": _tail(stderr),
            },
        }
    stdout = _as_text(completed.stdout)
    stderr = _as_text(completed.stderr)
    return {
        "returncode": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "stderr_tail": _tail(stderr),
        "receipt": {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "timed_out": False,
            "stdout_size": len(stdout.encode("utf-8")),
            "stderr_size": len(stderr.encode("utf-8")),
            "stderr_tail": _tail(stderr),
        },
    }


def _parse_gpu_inventory(text: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gpus: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for row_number, row in enumerate(csv.reader(text.splitlines()), start=1):
        fields = [field.strip() for field in row]
        if not fields or fields == [""]:
            continue
        if len(fields) != 6:
            blockers.append(
                {
                    "surface": "nvidia_smi",
                    "reason": "GPU_INVENTORY_ROW_MALFORMED",
                    "row_number": row_number,
                    "field_count": len(fields),
                }
            )
            continue
        index = _parse_int(fields[0])
        memory_used = _parse_int(fields[3])
        memory_total = _parse_int(fields[4])
        utilization = _parse_int(fields[5])
        if index is None:
            blockers.append({"surface": "nvidia_smi", "reason": "GPU_INDEX_UNPARSEABLE", "row_number": row_number})
            continue
        gpus.append(
            {
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu_percent": utilization,
                "compute_apps": [],
            }
        )
    return sorted(gpus, key=lambda item: int(item["index"])), blockers


def _parse_compute_apps(
    text: str,
    *,
    process_command_provider: ProcessCommandProvider,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    apps: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for row_number, row in enumerate(csv.reader(text.splitlines()), start=1):
        fields = [field.strip() for field in row]
        if not fields or fields == [""]:
            continue
        if len(fields) != 4:
            blockers.append(
                {
                    "surface": "nvidia_smi",
                    "reason": "COMPUTE_APP_ROW_MALFORMED",
                    "row_number": row_number,
                    "field_count": len(fields),
                }
            )
            continue
        pid = _parse_int(fields[1])
        used_memory = _parse_int(fields[3])
        if pid is None:
            blockers.append({"surface": "nvidia_smi", "reason": "COMPUTE_APP_PID_UNPARSEABLE", "row_number": row_number})
            continue
        apps.append(
            {
                "gpu_uuid": fields[0],
                "pid": pid,
                "process_name": fields[2],
                "used_memory_mib": used_memory,
                "command": process_command_provider(pid),
            }
        )
    return apps, blockers


def _attach_apps(
    gpus: list[dict[str, object]],
    apps: Sequence[Mapping[str, object]],
    blockers: list[dict[str, object]],
) -> None:
    by_uuid = {str(gpu["uuid"]): gpu for gpu in gpus}
    for app in apps:
        gpu = by_uuid.get(str(app["gpu_uuid"]))
        if gpu is None:
            blockers.append(
                {
                    "surface": "nvidia_smi",
                    "reason": "COMPUTE_APP_GPU_UUID_UNKNOWN",
                    "gpu_uuid": app["gpu_uuid"],
                    "pid": app["pid"],
                }
            )
            continue
        compute_apps = gpu.setdefault("compute_apps", [])
        if isinstance(compute_apps, list):
            compute_apps.append(dict(app))


def _build_report(
    *,
    snapshot: Mapping[str, object],
    policy: Mapping[str, object],
    requested: tuple[int, ...],
    allowed_pids: frozenset[int],
    memory_used_threshold_mib: int,
    utilization_threshold_percent: int,
) -> dict[str, object]:
    gpus = [dict(gpu) for gpu in _as_mapping_sequence(snapshot.get("gpus"))]
    observed_indices = tuple(int(gpu["index"]) for gpu in gpus if isinstance(gpu.get("index"), int))
    requested_indices = requested or observed_indices
    requested_set = set(requested_indices)
    blockers: list[dict[str, object]] = [dict(item) for item in _as_mapping_sequence(snapshot.get("blockers"))]
    for index in requested_indices:
        if index not in observed_indices:
            blockers.append({"surface": "gpu_selection", "reason": "REQUESTED_GPU_NOT_FOUND", "gpu_index": index})
    annotated_gpus = []
    requested_ready_count = 0
    for gpu in gpus:
        gpu_index = int(gpu["index"])
        in_requested_scope = gpu_index in requested_set
        gpu_blockers = []
        if in_requested_scope:
            gpu_blockers = _gpu_blockers(
                gpu,
                allowed_pids=allowed_pids,
                memory_used_threshold_mib=memory_used_threshold_mib,
                utilization_threshold_percent=utilization_threshold_percent,
            )
            blockers.extend(gpu_blockers)
            if not gpu_blockers:
                requested_ready_count += 1
        annotated_gpus.append(
            {
                **gpu,
                "requested": in_requested_scope,
                "exclusive_ready": bool(in_requested_scope and not gpu_blockers),
                "blocker_count": len(gpu_blockers),
                "blockers": gpu_blockers,
            }
        )
    requested_count = len(requested_indices)
    state = "ready" if not blockers and requested_count > 0 and requested_ready_count == requested_count else "blocked"
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-gpu-exclusivity-audit",
        "state": state,
        "timestamp_utc": snapshot.get("timestamp_utc"),
        "gpu_exclusivity_ready": state == "ready",
        "requested_gpu_count": requested_count,
        "ready_requested_gpu_count": requested_ready_count,
        "blocked_requested_gpu_count": requested_count - requested_ready_count,
        "observed_gpu_count": len(gpus),
        "requested_gpus": list(requested_indices),
        "policy": policy,
        "commands": snapshot.get("commands", {}),
        "gpus": annotated_gpus,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "side_effects": {
            "gpu_execution_started": False,
            "process_kill_attempted": False,
            "active_training_mutated": False,
        },
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "limitations": [
            "This audit is a read-only launch preflight and does not reserve GPUs.",
            "A ready result only proves the sampled moment had no blocking occupancy under the configured policy.",
            "This audit never authorizes M4 training; formal training still requires a ready strict phase gate.",
        ],
    }


def _gpu_blockers(
    gpu: Mapping[str, object],
    *,
    allowed_pids: frozenset[int],
    memory_used_threshold_mib: int,
    utilization_threshold_percent: int,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    gpu_index = int(gpu["index"])
    apps = _as_mapping_sequence(gpu.get("compute_apps"))
    non_allowed_apps = [app for app in apps if int(app.get("pid", -1)) not in allowed_pids]
    for app in non_allowed_apps:
        blockers.append(
            {
                "surface": "gpu_process",
                "reason": "COMPUTE_PROCESS_PRESENT",
                "gpu_index": gpu_index,
                "pid": app.get("pid"),
                "process_name": app.get("process_name"),
                "used_memory_mib": app.get("used_memory_mib"),
                "command": app.get("command"),
            }
        )
    memory_used = gpu.get("memory_used_mib")
    if isinstance(memory_used, int) and memory_used > memory_used_threshold_mib and non_allowed_apps:
        blockers.append(
            {
                "surface": "gpu_memory",
                "reason": "MEMORY_USED_ABOVE_THRESHOLD",
                "gpu_index": gpu_index,
                "observed_mib": memory_used,
                "threshold_mib": memory_used_threshold_mib,
            }
        )
    elif memory_used is None:
        blockers.append({"surface": "gpu_memory", "reason": "MEMORY_USED_UNAVAILABLE", "gpu_index": gpu_index})
    utilization = gpu.get("utilization_gpu_percent")
    if isinstance(utilization, int) and utilization > utilization_threshold_percent and non_allowed_apps:
        blockers.append(
            {
                "surface": "gpu_utilization",
                "reason": "UTILIZATION_ABOVE_THRESHOLD",
                "gpu_index": gpu_index,
                "observed_percent": utilization,
                "threshold_percent": utilization_threshold_percent,
            }
        )
    elif utilization is None:
        blockers.append({"surface": "gpu_utilization", "reason": "UTILIZATION_UNAVAILABLE", "gpu_index": gpu_index})
    return blockers


def _as_mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _read_process_command(pid: int) -> str | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    command = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return command or None


def _parse_int(value: str) -> int | None:
    text = value.strip()
    if not text or text.upper() == "N/A":
        return None
    token = text.split()[0]
    try:
        return int(float(token))
    except ValueError:
        return None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tail(text: str, limit: int = 500) -> str:
    return text[-limit:]


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_OUTPUT_EXISTS")
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else destination.parent
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "gpu-exclusivity-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "gpu-exclusivity-audit.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-gpu-exclusivity-audit-manifest",
            "state": report["state"],
            "gpu_exclusivity_ready": report["gpu_exclusivity_ready"],
            "requested_gpus": report["requested_gpus"],
            "requested_gpu_count": report["requested_gpu_count"],
            "ready_requested_gpu_count": report["ready_requested_gpu_count"],
            "blocked_requested_gpu_count": report["blocked_requested_gpu_count"],
            "observed_gpu_count": report["observed_gpu_count"],
            "blocker_count": report["blocker_count"],
            "blockers": report["blockers"],
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "side_effects": report["side_effects"],
            "report_path": str(destination / "gpu-exclusivity-audit.json"),
            "markdown_path": str(destination / "gpu-exclusivity-audit.md"),
            "cas_refs": {
                "gpu_exclusivity_audit_json": report_ref,
                "gpu_exclusivity_audit_markdown": markdown_ref,
            },
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# GPU Exclusivity Audit",
        "",
        f"- state: `{report['state']}`",
        f"- gpu_exclusivity_ready: `{report['gpu_exclusivity_ready']}`",
        f"- requested_gpus: `{report['requested_gpus']}`",
        f"- ready_requested_gpu_count: `{report['ready_requested_gpu_count']}`",
        f"- blocked_requested_gpu_count: `{report['blocked_requested_gpu_count']}`",
        f"- blocker_count: `{report['blocker_count']}`",
        "",
        "| GPU | Requested | Memory MiB | Util % | Compute Apps | Ready |",
        "|:--|:--|--:|--:|--:|:--|",
    ]
    for gpu in _as_mapping_sequence(report.get("gpus")):
        apps = _as_mapping_sequence(gpu.get("compute_apps"))
        lines.append(
            "| "
            f"`{gpu.get('index')}` | "
            f"`{gpu.get('requested')}` | "
            f"`{gpu.get('memory_used_mib')}` | "
            f"`{gpu.get('utilization_gpu_percent')}` | "
            f"`{len(apps)}` | "
            f"`{gpu.get('exclusive_ready')}` |"
        )
    blockers = _as_mapping_sequence(report.get("blockers"))
    if blockers:
        lines.extend(["", "## Blockers"])
        for blocker in blockers:
            surface = blocker.get("surface")
            reason = blocker.get("reason")
            detail = ""
            if "gpu_index" in blocker:
                detail += f" gpu={blocker['gpu_index']}"
            if "pid" in blocker:
                detail += f" pid={blocker['pid']}"
            lines.append(f"- `{surface}`: `{reason}`{detail}")
    lines.extend(["", "This audit is read-only and does not authorize M4 training."])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_OUTPUT_EXISTS")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit requested GPUs for exclusive launch readiness")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--gpus", type=int, nargs="*", default=None)
    run.add_argument("--allow-pid", type=int, action="append", default=[])
    run.add_argument("--memory-used-threshold-mib", type=int, default=1024)
    run.add_argument("--utilization-threshold-percent", type=int, default=10)
    run.add_argument("--nvidia-smi-timeout-seconds", type=float, default=10.0)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_gpu_exclusivity_audit(
            output_root=args.output_root,
            requested_gpus=args.gpus,
            allowed_pids=args.allow_pid,
            memory_used_threshold_mib=args.memory_used_threshold_mib,
            utilization_threshold_percent=args.utilization_threshold_percent,
            nvidia_smi_timeout_seconds=args.nvidia_smi_timeout_seconds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise GpuExclusivityAuditError("GPU_EXCLUSIVITY_AUDIT_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
