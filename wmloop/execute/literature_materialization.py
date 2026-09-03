"""Automatic, receipt-bound materialization for literature method work orders.

Literature records are hypotheses, not executable code.  This module closes
the control-plane gap by translating each record into a versioned research
idea and sending it through the existing isolated materialization engine.  The
default implementation is a small, explicit adapter surrogate: it is useful
for testing the target runtime contract, but its receipt never claims that the
paper was faithfully reproduced.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.execute.automatic_materialization import (
    AutomaticMaterializationError,
    materialization_plan_digest,
    run_automatic_materialization,
)


class LiteratureMaterializationError(RuntimeError):
    """The literature materialization bundle could not be built safely."""


def run_literature_method_materialization(
    *,
    method_staging_manifest: Path,
    output_root: Path,
    source_root: Path,
    project_root: Path,
    evaluator_contract: Path,
    runtime_python: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    max_candidates: int = 8,
) -> dict[str, object]:
    """Materialize staged unknown methods independently and merge ready catalogs."""

    source = _load(method_staging_manifest)
    if source.get("artifact_type") != "wmloop-literature-method-staging-manifest":
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_SOURCE_INVALID")
    orders = source.get("work_order_paths")
    if not isinstance(orders, Mapping):
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_WORK_ORDERS_INVALID")
    if max_candidates < 1:
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_LIMIT_INVALID")
    prototype = _scheduler_prototype(evaluator_contract)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_OUTPUT_INVALID")
        return _load(manifest_path)
    destination.mkdir(mode=0o700, parents=True)
    records: list[dict[str, object]] = []
    catalogs: list[dict[str, object]] = []
    for candidate_id, raw_order in sorted(orders.items())[:max_candidates]:
        if not isinstance(candidate_id, str) or not isinstance(raw_order, str):
            continue
        try:
            order = _load(Path(raw_order))
            method = order.get("literature_method")
            if not isinstance(method, Mapping):
                raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_METHOD_INVALID")
            idea, converted_order, plan = _build_transaction(
                candidate_id=candidate_id,
                order=order,
                method=method,
                prototype=prototype,
                project_root=Path(project_root).resolve(strict=True),
                runtime_python=runtime_python,
                output_root=destination,
            )
            child = destination / "candidates" / _safe_id(candidate_id)
            result = run_automatic_materialization(
                plan_path=plan,
                work_order_path=converted_order,
                idea_path=idea,
                source_root=Path(source_root).resolve(strict=True),
                output_root=child,
                project_root=Path(project_root).resolve(strict=True),
            )
            catalog_path = result.get("candidate_catalog_path")
            if result.get("state") == "ready_for_candidate_compilation" and isinstance(catalog_path, str):
                catalog = _load(Path(catalog_path))
                catalogs.append(catalog)
            records.append({"candidate_id": candidate_id, "state": result.get("state"), "manifest_path": str(child / "manifest.json"), "blocker_count": result.get("blocker_count", 0)})
        except Exception as exc:
            records.append({"candidate_id": candidate_id, "state": "blocked", "error": str(exc)[:500]})
    merged = _merge_catalogs(
        catalogs,
        source=source,
        model_family=str(prototype.get("model_family") or "world_model"),
        blocked=[row for row in records if row.get("state") == "blocked"],
    )
    catalog_path = destination / "candidate-catalog.json"
    _write(catalog_path, merged)
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-literature-method-materialization-manifest",
        "state": "ready",
        "source_staging_manifest": str(Path(method_staging_manifest).resolve()),
        "record_count": len(records),
        "ready_count": sum(row.get("state") == "ready_for_candidate_compilation" for row in records),
        "blocked_count": sum(row.get("state") == "blocked" for row in records),
        "records": records,
        "candidate_catalog_path": str(catalog_path),
        "candidate_catalog_sha256": _sha256(catalog_path.read_bytes()),
        "claim_boundary": "Ready receipts admit bounded candidate compilation of explicit adapter surrogates; they do not certify faithful reproduction of the cited paper or promotion.",
    }
    _write(destination / "manifest.json", manifest)
    return manifest


def _build_transaction(*, candidate_id: str, order: Mapping[str, Any], method: Mapping[str, Any], prototype: Mapping[str, Any], project_root: Path, runtime_python: Path | None, output_root: Path) -> tuple[Path, Path, Path]:
    safe = _module_id(candidate_id)
    transaction = output_root / "transactions" / safe
    transaction.mkdir(mode=0o700, parents=True, exist_ok=True)
    idea_id = f"literature-idea-{safe}"
    source = dict(method.get("source") or {})
    hypothesis = f"A bounded adapter surrogate for {source.get('title') or candidate_id} reduces the observed long-horizon failure without changing the frozen evaluator."
    contract = {"preserved_components": ["model_runtime", "candidate_adapter"], "mode": "surrogate_probe", "faithful_paper_reproduction": False}
    idea_payload = {"schema_version": 1, "artifact_type": "verdiwm-acwm-research-idea", "idea_id": idea_id, "hypothesis": hypothesis, "track": "predictive_quality", "failure_context": list(method.get("target_failure_signatures") or ["unclassified_failure"]), "source": source, "materialization_contract": contract, "claim_boundary": "This is an adapter surrogate for runtime validation, not a claim of faithful paper implementation."}
    idea_path = transaction / "idea.json"
    _write(idea_path, idea_payload)
    converted = {"schema_version": 1, "artifact_type": "verdiwm-acwm-research-materialization-work-order", "idea_id": idea_id, "idea_sha256": _sha256(_canonical(idea_payload)), "execution_authority": "none_until_materialized_and_compiled", "source_grounding": source, "materialization_contract": contract}
    order_path = transaction / "work-order.json"
    _write(order_path, converted)
    module = f"wmloop/primitives/definitions/{safe}/apply.py"
    package_files = [
        "wmloop/__init__.py",
        "wmloop/primitives/__init__.py",
        "wmloop/primitives/definitions/__init__.py",
        f"wmloop/primitives/definitions/{safe}/__init__.py",
    ]
    test = f"tests/test_{safe}.py"
    descriptor = f"wmloop/primitives/definitions/{safe}/descriptor.json"
    runtime = str(runtime_python or sys.executable)
    agent = project_root / "wmloop" / "execute" / "literature_materializer_agent.py"
    template = copy.deepcopy(dict(prototype))
    template["candidate_id"] = candidate_id
    template["hypothesis"] = hypothesis
    template["selection_reason"] = "Automatically materialized literature adapter surrogate with receipt-bound checks."
    template["falsification_criterion"] = "Reject when the surrogate fails static, offline, or frozen runtime checks."
    template.setdefault("literature_arxiv_ids", [str(source.get("arxiv_id") or "")])
    template["retrieval_keys"] = {"failure_signatures": list(method.get("target_failure_signatures") or ["unclassified_failure"]), "stage": "literature_materialized"}
    plan_payload = {"schema_version": 1, "artifact_type": "verdiwm-automatic-materialization-plan", "plan_id": f"literature-materialize-{safe}", "plan_digest": "", "idea_id": idea_id, "candidate_id": candidate_id, "model_family": str(prototype.get("model_family") or "world_model"), "required_hooks": [str(method.get("required_hook") or "H3")], "estimated_gpu_hours": float(method.get("estimated_gpu_hours") or 1.0), "descriptor_path": descriptor, "allowed_changed_paths": [*package_files, module, test, descriptor], "forbidden_changed_paths": ["eval.py", "scripts/eval_all.sh", "results/**", "configs/goal/**", "runs/m0/protocol/**"], "source_overlay": {"include_untracked_globs": ["wmloop/**", "tests/**"], "max_file_bytes": 16777216, "max_total_bytes": 67108864}, "agent_command": [runtime, str(agent), "--workspace", "{workspace}", "--descriptor-path", "{descriptor_path}", "--candidate-id", "{candidate_id}", "--idea-id", "{idea_id}"], "agent_timeout_seconds": 60, "fixed_checks": [{"label": "static", "argv": [runtime, "-m", "py_compile", *package_files, module, test], "timeout_seconds": 120}, {"label": "offline", "argv": [runtime, "-m", "pytest", "-q", test], "timeout_seconds": 180}], "inherit_environment_keys": [], "preserve_codex_auth": False, "candidate_template": template, "implementation_contract": {"module": module, "required_symbol": "apply", "surrogate": True}}
    plan_payload["plan_digest"] = materialization_plan_digest(plan_payload)
    plan_path = transaction / "plan.json"
    _write(plan_path, plan_payload)
    return idea_path, order_path, plan_path


def _scheduler_prototype(evaluator: Path) -> dict[str, Any]:
    payload = _load(evaluator)
    raw = payload.get("scheduler_template") or payload.get("scheduler_template_path")
    if not isinstance(raw, str) or not raw:
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_SCHEDULER_TEMPLATE_MISSING")
    path = Path(raw)
    if not path.is_absolute():
        path = evaluator.parent / path
    template = _load(path)
    rows = template.get("candidates")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_SCHEDULER_TEMPLATE_INVALID")
    return dict(rows[0])


def _merge_catalogs(catalogs: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any], model_family: str, blocked: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    candidates = [dict(row) for catalog in catalogs for row in catalog.get("candidates", []) if isinstance(row, Mapping)]
    gaps = [
        {
            "candidate_id": str(row.get("candidate_id") or "unknown"),
            "source": "literature_materialization",
            "mechanism_hypothesis": "Materialization transaction failed before candidate compilation.",
            "required_hooks": ["H3"],
            "failure_signatures": ["materialization_failure"],
            "estimated_gpu_hours": 0.0,
            "reason": str(row.get("error") or "MATERIALIZATION_BLOCKED"),
        }
        for row in blocked
    ]
    return {"schema_version": 1, "artifact_type": "verdiwm-method-candidate-catalog", "catalog_id": "literature-materialized-" + _sha256(_canonical(source))[:16], "model_family": model_family, "candidates": candidates, "capability_gaps": gaps, "claim_boundary": "Candidates are admitted only through their bound automatic-materialization receipts; each is an explicit adapter surrogate unless its descriptor says otherwise."}


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:90] or "unknown"


def _module_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")[:90] or "unknown"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiteratureMaterializationError("LITERATURE_MATERIALIZATION_JSON_INVALID")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
