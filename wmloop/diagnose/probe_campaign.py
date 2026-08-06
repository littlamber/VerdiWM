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
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding_admission import admission_from_manifest
from wmloop.execute.auto_experiment import run_auto_experiment
from wmloop.retrieve.index import index_probe_experience


class DiagnosticProbeError(RuntimeError):
    """A diagnostic probe contract, receipt, or result is invalid."""


_SOURCE_PLACEHOLDER = re.compile(r"\{(?:python|verdiwm_python|repo_root|asset:--[A-Za-z0-9_-]+)\}")
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
    retrieval_db: Path | None = None,
) -> dict[str, object]:
    """Run or resume one probe, then optionally publish it to retrieval."""

    contract = _load_contract(Path(contract_path))
    sidecar = Path(sidecar_root).resolve(strict=True)
    conformance = Path(conformance_root).resolve(strict=True)
    workspace = Path(workspace_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    report = _load_json(
        sidecar / "onboarding-report.json",
        "DIAGNOSTIC_PROBE_ONBOARDING_REPORT_INVALID",
    )
    admission = admission_from_manifest(conformance)
    input_hash = _sha256(
        _canonical_json(
            {"contract": contract, "admission": admission, "workspace": str(workspace)}
        )
    )
    manifest_path = destination / "manifest.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_OUTPUT_INVALID")
        if manifest_path.is_file() and not manifest_path.is_symlink():
            existing = _load_json(
                manifest_path, "DIAGNOSTIC_PROBE_MANIFEST_INVALID"
            )
            if existing.get("input_hash") != input_hash:
                raise DiagnosticProbeError("DIAGNOSTIC_PROBE_INPUT_MISMATCH")
            return existing
        if {path.name for path in destination.iterdir()} - {"probe-plan.json", "run"}:
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_OUTPUT_UNBOUND")

    plan = _build_plan(contract, report=report, admission=admission)
    run_root = destination / "run"
    execution = run_auto_experiment(
        plan_path=_write_plan(destination, plan),
        output_root=run_root,
        workspace_root=workspace,
        archive_db=Path(archive_db),
        cas_root=Path(cas_root),
        lock_root=Path(lock_root),
        budget_db=Path(budget_db) if budget_db is not None else None,
        budget_total_gpu_hours=budget_total_gpu_hours,
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
    asset_fingerprint = _asset_fingerprint_from_report(report)
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


def _build_plan(
    contract: Mapping[str, object],
    *,
    report: Mapping[str, object],
    admission: Mapping[str, object],
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
    plan = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-plan",
        "campaign_id": f"diagnostic-{contract['probe_id']}",
        "trial_id": f"diagnostic-{contract['probe_id']}",
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
    values = {
        "{python}": str(runtime["selected_python"]),
        # Preserve a virtual-environment entrypoint symlink. Resolving it to
        # the base interpreter drops the environment's site-packages.
        "{verdiwm_python}": str(Path(__import__("sys").executable).absolute()),
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
    if result.get("schema_version") != 1 or result.get("artifact_type") not in {
        "verdiwm-auto-experiment-result",
        "verdiwm-diagnostic-probe-result",
    }:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_CONTRACT_INVALID")
    if result.get("state") != "ready" or result.get("probe_id") != contract["probe_id"]:
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_RESULT_ID_OR_STATE_INVALID")
    for field in ("model_family", "runtime_capability"):
        if not isinstance(result.get(field), str) or not str(result[field]):
            raise DiagnosticProbeError(f"DIAGNOSTIC_PROBE_RESULT_FIELD_INVALID:{field}")
    signatures = result.get("failure_signatures")
    if not isinstance(signatures, list) or not signatures or any(not isinstance(item, str) or not item for item in signatures):
        raise DiagnosticProbeError("DIAGNOSTIC_PROBE_FAILURE_SIGNATURES_INVALID")
    for field in contract["failure_signature_fields"]:
        if field not in result:
            raise DiagnosticProbeError(f"DIAGNOSTIC_PROBE_SIGNATURE_FIELD_MISSING:{field}")


def _asset_fingerprint_from_report(report: Mapping[str, object]) -> str | None:
    connector = report.get("connector")
    bindings = connector.get("asset_bindings") if isinstance(connector, Mapping) else None
    if not isinstance(bindings, list):
        return None
    fingerprints = sorted(
        str(row["fingerprint"])
        for row in bindings
        if isinstance(row, Mapping) and isinstance(row.get("fingerprint"), str)
    )
    return _sha256(_canonical_json(fingerprints)) if fingerprints else None


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
        existing = _load_json(path, "DIAGNOSTIC_PROBE_PLAN_INVALID")
        if existing != dict(plan):
            raise DiagnosticProbeError("DIAGNOSTIC_PROBE_PLAN_MISMATCH")
        return path
    _write_json_atomic(path, plan)
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
