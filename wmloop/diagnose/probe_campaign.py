"""Execute a declared diagnostic probe and settle its failure signature.

The probe is a bounded ``screen`` transaction through the same GPU lease,
sampling, CAS, and archive boundary as an experiment.  It is intentionally
diagnostic-only: its output can rank candidates, but it cannot satisfy a model
quality gate or create a confirmed intervention effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding_admission import admission_from_manifest
from wmloop.execute.auto_experiment import run_auto_experiment
from wmloop.retrieve.index import index_probe_experience, probe_asset_fingerprint
from wmloop.storage import atomic_write, checked_path


class DiagnosticProbeError(RuntimeError):
    """A diagnostic probe contract, receipt, or result is invalid."""


_SOURCE_PLACEHOLDER = re.compile(
    r"\{(?:python|verdiwm_python|verdiwm_root|runtime_path|repo_root|asset:--[A-Za-z0-9_-]+)\}"
)
_RUNTIME_PLACEHOLDERS = {"{scratch_dir}", "{workspace_root}", "{output_root}", "{gpu_index}", "{gpu_uuid}"}


def run_diagnostic_probe(
    *,
    contract_path: Path,
    sidecar_root: Path,
    conformance_root: Path,
    output_root: Path,
    workspace_root: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path,
    budget_db: Path | None = None,
    budget_total_gpu_hours: float | None = None,
    budget_max_trial_gpu_hours: float = 120.0,
    budget_high_trial_limit: int = 2,
    budget_require_high_cost_approval: bool = True,
    retrieval_db: Path | None = None,
    campaign_id: str | None = None,
    recover_only: bool = False,
) -> dict[str, object]:
    """Run or resume one probe, then optionally publish it to retrieval."""

    contract = _load_contract(Path(contract_path))
    sidecar = Path(sidecar_root).resolve(strict=True)
    conformance = Path(conformance_root).resolve(strict=True)
    workspace = Path(workspace_root).resolve(strict=True)
    destination = checked_path(Path(output_root), code="DIAGNOSTIC_PROBE_OUTPUT_INVALID", error=DiagnosticProbeError)
    admission = admission_from_manifest(conformance)
    report_path = sidecar / "onboarding-report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_ONBOARDING_REPORT_INVALID")
    try:
        report_bytes = report_path.read_bytes()
    except OSError as exc:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_ONBOARDING_REPORT_INVALID") from exc
    if _sha256(report_bytes) != admission.get("onboarding_report_sha256"):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_ONBOARDING_REPORT_HASH_MISMATCH")
    report = _json_object(report_bytes, "DIAGNOSTIC_PROBE_ONBOARDING_REPORT_INVALID")
    input_hash = _sha256(
        _canonical_json(
            {"contract": contract, "admission": admission, "workspace": str(workspace)}
        )
    )
    manifest_path = destination / "manifest.json"
    existing_manifest = False
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_OUTPUT_INVALID")
        if manifest_path.is_file() and not manifest_path.is_symlink():
            existing = _load_json(
                manifest_path, "DIAGNOSTIC_PROBE_MANIFEST_INVALID"
            )
            if existing.get("input_hash") != input_hash:
                raise DiagnosticProbeError("DIAGNOSTIC_PROBE_INPUT_MISMATCH")
            existing_manifest = True
        elif {path.name for path in destination.iterdir()} - {"probe-plan.json", "run"}:
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_OUTPUT_UNBOUND")

    plan = _build_plan(
        contract,
        report=report,
        admission=admission,
        campaign_id=campaign_id,
    )
    run_root = destination / "run"
    if existing_manifest or recover_only:
        # A manifest is only a derived view. Never rerun a previously finished
        # probe merely because its archive or CAS has disappeared, and never
        # trust the manifest's routing fields on recovery.
        _require_archived_probe_receipt(
            run_root=run_root, trial_id=str(plan["trial_id"]),
            archive_db=Path(archive_db), cas_root=Path(cas_root),
        )
    execution = run_auto_experiment(
        plan_path=_write_plan(destination, plan),
        output_root=run_root,
        workspace_root=workspace,
        archive_db=Path(archive_db),
        cas_root=Path(cas_root),
        lock_root=Path(lock_root),
        budget_db=Path(budget_db) if budget_db is not None else None,
        budget_total_gpu_hours=budget_total_gpu_hours,
        budget_max_trial_gpu_hours=budget_max_trial_gpu_hours,
        budget_high_trial_limit=budget_high_trial_limit,
        budget_require_high_cost_approval=budget_require_high_cost_approval,
        recover_only=recover_only or existing_manifest,
    )
    receipt_path = Path(str(execution["receipt_path"])).resolve(strict=True)
    receipt = _load_json(receipt_path, "DIAGNOSTIC_PROBE_RECEIPT_INVALID")
    if execution.get("verdict") != "PASS":
        manifest = _manifest(
            contract=contract,
            input_hash=input_hash,
            destination=destination,
            state="blocked",
            verdict="BLOCKED",
            receipt=receipt,
            blockers=["DIAGNOSTIC_PROBE_EXECUTION_NOT_PASS"],
        )
        _write_json_atomic(manifest_path, manifest)
        return manifest

    result_ref = _result_ref(receipt, contract=contract)
    cas = ContentAddressedStore(Path(cas_root))
    result = _json_object(cas.read_bytes(result_ref), "DIAGNOSTIC_PROBE_RESULT_INVALID")
    _validate_probe_result(result, contract)
    asset_fingerprint = probe_asset_fingerprint(receipt, cas=cas)
    if retrieval_db is not None:
        index_probe_experience(
            database_path=Path(retrieval_db),
            result=result,
            receipt=receipt,
            archive_db=Path(archive_db),
            cas_root=Path(cas_root),
            asset_fingerprint=asset_fingerprint,
            result_artifact_path=str(
                contract.get("diagnostic_result_path") or "result.json"
            ),
        )
    manifest = _manifest(
        contract=contract,
        input_hash=input_hash,
        destination=destination,
        state="settled",
        verdict="PASS",
        receipt=receipt,
        result=result,
        result_ref=result_ref,
        asset_fingerprint=asset_fingerprint,
        retrieval_db=retrieval_db,
        blockers=[],
    )
    _write_json_atomic(destination / "probe-result.json", result)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _require_archived_probe_receipt(*, run_root: Path, trial_id: str, archive_db: Path, cas_root: Path) -> None:
    """Require an existing archive binding before recovering a finished probe."""

    receipt_path = run_root / "receipts" / f"{trial_id}.json"
    checked_path(receipt_path, code="DIAGNOSTIC_PROBE_RECOVERY_EVIDENCE_MISSING", error=DiagnosticProbeError)
    if (receipt_path.is_symlink() or not receipt_path.is_file()
            or archive_db.is_symlink() or not archive_db.is_file()
            or cas_root.is_symlink() or not (cas_root / "cas").is_dir()):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RECOVERY_EVIDENCE_MISSING")
    receipt = _load_json(receipt_path, "DIAGNOSTIC_PROBE_RECOVERY_RECEIPT_INVALID")
    ref = receipt.get("receipt_ref")
    digest = receipt.get("receipt_hash")
    archive_trial_id = receipt.get("archive_trial_id")
    if not all(isinstance(value, str) and value for value in (ref, digest, archive_trial_id)):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RECOVERY_RECEIPT_INVALID")
    try:
        payload = ContentAddressedStore(cas_root).read_bytes(ref)
        stored = _json_object(payload, "DIAGNOSTIC_PROBE_RECOVERY_RECEIPT_INVALID")
        body = {key: value for key, value in receipt.items() if key not in {"receipt_ref", "receipt_hash", "cleanup"}}
        if (_sha256(payload) != digest or stored != body
                or not ArchiveStore(archive_db).has_settled_receipt(archive_trial_id, ref, digest)):
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RECOVERY_RECEIPT_MISMATCH")
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RECOVERY_RECEIPT_INVALID") from exc


def _build_plan(
    contract: Mapping[str, object],
    *,
    report: Mapping[str, object],
    admission: Mapping[str, object],
    campaign_id: str | None = None,
) -> dict[str, object]:
    values = _materialization_values(report)
    command = [_materialize_token(str(token), values) for token in contract["command"]]
    environment = {
        str(key): _materialize_token(str(value), values)
        for key, value in dict(contract["environment"]).items()
    }
    environment.update(
        {
            "VERDIWM_PROBE_ID": str(contract["probe_id"]),
            "VERDIWM_PROBE_RESULT_PATH": "{scratch_dir}/" + str(contract["result_path"]),
        }
    )
    diagnostic_campaign_id = f"diagnostic-{contract['probe_id']}"
    if campaign_id:
        # Archive identities are global, while a probe's local output is
        # campaign-scoped. Isolate pipeline runs so a prior campaign cannot
        # masquerade as a resumable settlement for the current one.
        suffix = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()[:12]
        diagnostic_campaign_id = f"{diagnostic_campaign_id}-{suffix}"
    plan = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-plan",
        "campaign_id": diagnostic_campaign_id,
        "trial_id": diagnostic_campaign_id,
        "objective": contract["objective"],
        "hypothesis": contract["hypothesis"],
        "selection_reason": contract["selection_reason"],
        "falsification_criterion": contract["falsification_criterion"],
        "stage": "screen",
        "command": command,
        "working_directory": contract["working_directory"],
        "allowed_gpu_indices": contract["allowed_gpu_indices"],
        "estimated_gpu_hours": contract["estimated_gpu_hours"],
        "total_budget_gpu_hours": contract["total_budget_gpu_hours"],
        "timeout_seconds": contract["timeout_seconds"],
        "gpu_wait_seconds": contract["gpu_wait_seconds"],
        "sample_interval_seconds": contract["sample_interval_seconds"],
        "result_path": contract["result_path"],
        "artifacts": contract["artifacts"],
        "metric_gates": contract["metric_gates"],
        "environment": environment,
        "cleanup_policy": contract["cleanup_policy"],
        "onboarding_admission": admission,
    }
    return plan


def _materialization_values(report: Mapping[str, object]) -> dict[str, str]:
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("selected_python"), str):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RUNTIME_INVALID")
    runtime_bin = Path(str(runtime["selected_python"])).parent
    runtime_path_entries = [str(runtime_bin)]
    for ffmpeg_dir in sorted(
        runtime_bin.parent.glob("lib/python*/site-packages/imageio_ffmpeg/binaries")
    ):
        if ffmpeg_dir.is_dir():
            runtime_path_entries.append(str(ffmpeg_dir))
    runtime_path_entries.append(os.environ.get("PATH", ""))
    values = {
        "{python}": str(runtime["selected_python"]),
        # Preserve a virtual-environment entrypoint symlink. Resolving it to
        # the base interpreter drops the environment's site-packages.
        "{verdiwm_python}": str(Path(__import__("sys").executable).absolute()),
        "{verdiwm_root}": str(Path(__file__).resolve().parents[2]),
        "{runtime_path}": os.pathsep.join(
            runtime_path_entries
        ),
        "{repo_root}": str(Path(str(report["repo_root"])).resolve()),
    }
    connector = report.get("connector")
    bindings = connector.get("asset_bindings") if isinstance(connector, Mapping) else None
    if not isinstance(bindings, list):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_ASSET_BINDINGS_INVALID")
    for row in bindings:
        if not isinstance(row, Mapping) or row.get("state") != "discovered":
            continue
        parameter = row.get("parameter")
        path = row.get("resolved_path")
        if isinstance(parameter, str) and isinstance(path, str):
            values[f"{{asset:{parameter}}}"] = path
    return values


def _materialize_token(value: str, values: Mapping[str, str]) -> str:
    for placeholder in _SOURCE_PLACEHOLDER.findall(value):
        if placeholder not in values:
            raise DiagnosticProbeError(f"DIAGNOSTIC_PROBE_PLACEHOLDER_UNBOUND:{placeholder}")
        value = value.replace(placeholder, values[placeholder])
    remaining = set(re.findall(r"\{[^{}]+\}", value))
    if remaining - _RUNTIME_PLACEHOLDERS:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_PLACEHOLDER_INVALID")
    return value


def _validate_probe_result(result: Mapping[str, object], contract: Mapping[str, object]) -> None:
    try:
        validate_document("diagnostic_probe_result", result)
    except ContractValidationError as exc:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_CONTRACT_INVALID") from exc
    if result["state"] != "ready" or any(
        result[field] != contract[field]
        for field in ("probe_id", "model_family", "runtime_capability")
    ):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_ID_OR_STATE_INVALID")
    for field in contract["failure_signature_fields"]:
        if field not in result:
            raise DiagnosticProbeError(f"DIAGNOSTIC_PROBE_SIGNATURE_FIELD_MISSING:{field}")


def _load_contract(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_CONTRACT_INVALID")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError
        validate_document("diagnostic_probe_contract", payload)
    except (OSError, json.JSONDecodeError, TypeError, ContractValidationError) as exc:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_CONTRACT_INVALID") from exc
    if float(payload["estimated_gpu_hours"]) > float(payload["total_budget_gpu_hours"]):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_ESTIMATE_EXCEEDS_BUDGET")
    if len(set(payload["allowed_gpu_indices"])) != len(payload["allowed_gpu_indices"]):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_GPU_ALLOWLIST_DUPLICATE")
    diagnostic_result_path = payload.get("diagnostic_result_path")
    if diagnostic_result_path is not None:
        if not isinstance(diagnostic_result_path, str) or not diagnostic_result_path:
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_PATH_INVALID")
        if diagnostic_result_path not in payload["artifacts"]:
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_ARTIFACT_NOT_DECLARED")
    return payload


def _result_ref(
    receipt: Mapping[str, object], *, contract: Mapping[str, object] | None = None
) -> str:
    refs = receipt.get("artifact_refs")
    result_name = str((contract or {}).get("diagnostic_result_path") or "result.json")
    if not isinstance(refs, Mapping) or not isinstance(refs.get(result_name), str):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_REF_MISSING")
    return str(refs[result_name])


def _manifest(
    *,
    contract: Mapping[str, object],
    input_hash: str,
    destination: Path,
    state: str,
    verdict: str,
    receipt: Mapping[str, object],
    result: Mapping[str, object] | None = None,
    result_ref: str | None = None,
    asset_fingerprint: str | None = None,
    retrieval_db: Path | None = None,
    blockers: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-manifest",
        "state": state,
        "verdict": verdict,
        "probe_id": contract["probe_id"],
        "model_family": result.get("model_family") if result else contract["model_family"],
        "runtime_capability": result.get("runtime_capability") if result else contract["runtime_capability"],
        "failure_signatures": list(result.get("failure_signatures", [])) if result else [],
        "archive_trial_id": receipt.get("archive_trial_id"),
        "asset_fingerprint": asset_fingerprint,
        "input_hash": input_hash,
        "run_manifest_path": str(destination / "run" / "manifest.json"),
        "receipt_path": str(destination / "run" / "receipts" / f"{receipt.get('trial_id', contract['probe_id'])}.json"),
        "receipt_ref": receipt.get("receipt_ref"),
        "result_ref": result_ref,
        "retrieval_db": str(Path(retrieval_db).resolve()) if retrieval_db is not None else None,
        "blockers": blockers,
        "claim_boundary": "A diagnostic probe identifies a failure signature and routing context. It cannot establish a model-quality gain or formal verdict.",
    }


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticProbeError(code) from exc
    if not isinstance(value, dict):
        raise DiagnosticProbeError(code)
    return value


def _json_object(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticProbeError(code) from exc
    if not isinstance(value, dict):
        raise DiagnosticProbeError(code)
    return value


def _write_plan(destination: Path, plan: Mapping[str, object]) -> Path:
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = destination / "probe-plan.json"
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_PLAN_INVALID")
        existing = _load_json(path, "DIAGNOSTIC_PROBE_PLAN_INVALID")
        if existing != dict(plan):
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_PLAN_MISMATCH")
        return path
    _write_json_atomic(path, plan)
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    checked_path(path, code="DIAGNOSTIC_PROBE_OUTPUT_INVALID", error=DiagnosticProbeError)
    atomic_write(path, _canonical_json(payload))


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
