"""Compile untrusted paper records into typed, non-executable method hypotheses.

The literature retriever deliberately stops at data-only paper candidates.  This
module is the next control-plane boundary: a synthesis client may suggest a
method, but the result must satisfy a strict contract, resolve to a registered
primitive or an explicit materialization work order, and carry bounded dose and
falsification information.  No field in this artifact is a shell command.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveValidationError
from wmloop.propose.primitive_materialization_prompt import (
    run_primitive_materialization_prompt,
)


class LiteratureMethodStagingError(RuntimeError):
    """A paper-to-method conversion failed closed."""


class MethodSynthesisClient(Protocol):
    """Injectable offline or remote method synthesis boundary."""

    def synthesize(
        self,
        paper: Mapping[str, Any],
        *,
        model_family: str | None,
        failure_signatures: Sequence[str],
        registry: PrimitiveRegistry,
    ) -> Mapping[str, Any]:
        """Return one JSON-compatible method hypothesis without commands."""


class HeuristicMethodSynthesisClient:
    """Deterministic cold-start synthesizer used when no external model is set.

    A registered primitive is selected only when its name, family, or declared
    literature label is explicitly present in the paper text.  Otherwise the
    paper remains a materialization proposal and cannot influence execution.
    """

    def synthesize(
        self,
        paper: Mapping[str, Any],
        *,
        model_family: str | None,
        failure_signatures: Sequence[str],
        registry: PrimitiveRegistry,
    ) -> Mapping[str, Any]:
        text = _normalise(
            " ".join(
                str(paper.get(key) or "")
                for key in ("title", "mechanism_summary")
            )
        )
        primitive = _explicit_primitive(text, registry)
        failures = tuple(dict.fromkeys(str(value) for value in failure_signatures if str(value)))
        if primitive is not None:
            manifest = registry.manifest(primitive)
            targets = failures or manifest.targets_failures
            dose_hours = min(48.0, max(0.1, float(manifest.estimated_gpu_hours)))
            mode = "training" if manifest.layer in {"L3", "L5"} else "inference"
            return {
                "target_failure_signatures": list(targets),
                "primitive_reference": primitive,
                "proposed_primitive": None,
                "required_hook": manifest.hooks[0],
                "dose": {"mode": mode, "steps": 1, "gpu_hours": dose_hours},
                "applicability_conditions": [
                    (
                        f"model_family={model_family}"
                        if model_family
                        else "compatible world-model runtime"
                    ),
                    "diagnostic signature is present before intervention",
                ],
                "invariants": ["frozen evaluator and protocol remain unchanged"],
                "falsifiable_prediction": (
                    f"The {primitive} intervention improves the declared failure "
                    "metric within the bounded dose "
                    "without violating evaluator invariants."
                ),
                "estimated_implementation_hours": 1.0,
                "estimated_gpu_hours": dose_hours,
                "execution_authority": "ranking_only",
                "state": "validated",
            }
        safe_name = "literature_" + _safe_id(str(paper.get("arxiv_id") or "unknown"))
        return {
            "target_failure_signatures": list(failures or ("unclassified_failure",)),
            "primitive_reference": None,
            "proposed_primitive": {
                "name": safe_name,
                "family": "literature_method",
                "rationale": (
                    "A paper-derived mechanism has no explicit match in the "
                    "frozen primitive registry."
                ),
            },
            "required_hook": "H3",
            "dose": {"mode": "training", "steps": 1, "gpu_hours": 1.0},
            "applicability_conditions": [
                (
                    f"model_family={model_family}"
                    if model_family
                    else "compatible world-model runtime"
                ),
                "diagnostic signature is reproduced in a local probe",
            ],
            "invariants": ["frozen evaluator and protocol remain unchanged"],
            "falsifiable_prediction": (
                "The proposed mechanism reduces the target failure signature "
                "under a bounded canary; otherwise "
                "the work order is rejected."
            ),
            "estimated_implementation_hours": 8.0,
            "estimated_gpu_hours": 1.0,
            "execution_authority": "materialization_required",
            "state": "validated",
        }


def run_literature_method_staging(
    *,
    literature_manifest: Path,
    output_root: Path,
    repo_root: Path,
    failure_signatures: Sequence[str] = (),
    model_family: str | None = None,
    synthesis_client: MethodSynthesisClient | None = None,
    max_candidates: int = 8,
) -> dict[str, object]:
    """Create an immutable method-staging bundle from one retrieval manifest."""

    if max_candidates < 1 or max_candidates > 50:
        raise LiteratureMethodStagingError("LITERATURE_METHOD_MAX_CANDIDATES_INVALID")
    Path(repo_root).resolve(strict=True)
    control_root = Path(__file__).resolve().parents[2]
    source = Path(literature_manifest).resolve(strict=True)
    destination = Path(output_root).resolve()
    retrieval = _load_object(source, "LITERATURE_METHOD_RETRIEVAL_INVALID")
    if retrieval.get("artifact_type") != "verdiwm-literature-retrieval-manifest":
        raise LiteratureMethodStagingError("LITERATURE_METHOD_RETRIEVAL_INVALID")
    rows = retrieval.get("rows")
    if not isinstance(rows, list):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_RETRIEVAL_ROWS_INVALID")
    # The external model checkout is an execution target; the frozen registry
    # and its schema live in the VerdiWM control plane.
    registry = PrimitiveRegistry.from_root(control_root)
    client = synthesis_client or HeuristicMethodSynthesisClient()
    input_hash = _sha256(
        source.read_bytes()
        + b"\0"
        + _canonical_json(
            {
                "failure_signatures": list(failure_signatures),
                "model_family": model_family,
                "max_candidates": max_candidates,
                "registry_digest": registry.digest(),
                "synthesis_client": f"{type(client).__module__}.{type(client).__qualname__}",
            }
        )
    )
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise LiteratureMethodStagingError("LITERATURE_METHOD_OUTPUT_INVALID")
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise LiteratureMethodStagingError("LITERATURE_METHOD_OUTPUT_UNBOUND")
        return _resume(destination, input_hash=input_hash)
    records: list[dict[str, object]] = []
    for row in rows[:max_candidates]:
        if not isinstance(row, Mapping) or row.get("state") != "staged":
            continue
        candidate_path = _paper_path(row, source=source)
        paper = _load_object(candidate_path, "LITERATURE_METHOD_PAPER_INVALID")
        _validate_paper(paper, row=row)
        try:
            method = dict(
                client.synthesize(
                    paper,
                    model_family=model_family,
                    failure_signatures=failure_signatures,
                    registry=registry,
                )
            )
        except Exception as exc:
            raise LiteratureMethodStagingError("LITERATURE_METHOD_SYNTHESIS_FAILED") from exc
        method = _complete_and_validate(
            method,
            paper=paper,
            paper_path=candidate_path,
            registry=registry,
        )
        records.append(method)

    ranking = [item for item in records if item["primitive_reference"] is not None]
    work_orders = [item for item in records if item["primitive_reference"] is None]
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-literature-method-staging",
        "state": "ready",
        "input_hash": input_hash,
        "source_literature_manifest": str(source),
        "source_literature_manifest_sha256": _sha256(source.read_bytes()),
        "source_revision": _git_revision(control_root),
        "registry_digest": registry.digest(),
        "record_count": len(records),
        "ranking_candidate_count": len(ranking),
        "materialization_work_order_count": len(work_orders),
        "records": records,
        "ranking_candidates": ranking,
        "work_orders": work_orders,
        "claim_boundary": (
            "Validated method hypotheses are control-plane data. Registered "
            "references may affect bounded ranking only; unknown methods require "
            "a next-version materialization and all admission gates before scheduling."
        ),
    }
    return _write_bundle(report, destination)


def run_literature_method_prompt_batch(
    *,
    method_staging_manifest: Path,
    output_root: Path,
    repo_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Render guarded implementation prompts for staged unknown methods."""

    source = Path(method_staging_manifest).resolve(strict=True)
    destination = Path(output_root).resolve()
    root = Path(repo_root).resolve(strict=True)
    source_bytes = source.read_bytes()
    input_hash = _sha256(source_bytes)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise LiteratureMethodStagingError(
                "LITERATURE_METHOD_PROMPT_OUTPUT_INVALID"
            )
        manifest = _load_object(
            destination / "manifest.json",
            "LITERATURE_METHOD_PROMPT_MANIFEST_INVALID",
        )
        if manifest.get("input_hash") != input_hash:
            raise LiteratureMethodStagingError(
                "LITERATURE_METHOD_PROMPT_INPUT_MISMATCH"
            )
        if (
            manifest.get("artifact_type")
            != "wmloop-literature-method-prompt-batch-manifest"
            or manifest.get("state") != "ready"
        ):
            raise LiteratureMethodStagingError(
                "LITERATURE_METHOD_PROMPT_MANIFEST_INVALID"
            )
        return manifest
    staging = _load_object(source, "LITERATURE_METHOD_PROMPT_SOURCE_INVALID")
    if staging.get("artifact_type") != "wmloop-literature-method-staging-manifest":
        raise LiteratureMethodStagingError(
            "LITERATURE_METHOD_PROMPT_SOURCE_INVALID"
        )
    work_orders = staging.get("work_order_paths")
    if not isinstance(work_orders, Mapping):
        raise LiteratureMethodStagingError(
            "LITERATURE_METHOD_PROMPT_WORK_ORDERS_INVALID"
        )
    records: list[dict[str, object]] = []
    try:
        destination.mkdir(mode=0o700, parents=True)
        prompt_root = destination / "prompts"
        prompt_root.mkdir(mode=0o700)
        for candidate_id, raw_path in sorted(work_orders.items()):
            if not isinstance(candidate_id, str) or not isinstance(raw_path, str):
                raise LiteratureMethodStagingError(
                    "LITERATURE_METHOD_PROMPT_WORK_ORDERS_INVALID"
                )
            work_order = Path(raw_path).resolve(strict=True)
            child_root = prompt_root / _safe_id(candidate_id)
            prompt = run_primitive_materialization_prompt(
                repo_root=root,
                work_order=work_order,
                output_root=child_root,
                archive_db=archive_db,
                cas_root=cas_root,
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "work_order_path": str(work_order),
                    "prompt_manifest_path": str(child_root / "manifest.json"),
                    "prompt_path": prompt["prompt_path"],
                    "state": prompt["state"],
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-literature-method-prompt-batch-manifest",
            "state": "ready",
            "input_hash": input_hash,
            "source_method_staging_manifest": str(source),
            "prompt_count": len(records),
            "records": records,
            "side_effects": {
                "source_code_mutated": False,
                "gpu_execution_started": False,
                "primitive_promoted": False,
            },
            "claim_boundary": (
                "Prompt packets are isolated implementation instructions. They "
                "grant neither source mutation nor experiment authority."
            ),
        }
        _write_json(destination / "manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _complete_and_validate(
    method: dict[str, Any],
    *,
    paper: Mapping[str, Any],
    paper_path: Path,
    registry: PrimitiveRegistry,
) -> dict[str, object]:
    arxiv_id = str(paper.get("arxiv_id") or "")
    title = str(paper.get("title") or "")
    source_url = str(paper.get("proposed_manifest", {}).get("source_url") or "")
    method["schema_version"] = 1
    method["artifact_type"] = "wmloop-literature-method-candidate"
    method["candidate_id"] = "method-" + _safe_id(arxiv_id)
    method["source"] = {
        "arxiv_id": arxiv_id,
        "title": title,
        "source_url": source_url,
        "paper_candidate_path": str(paper_path),
        "paper_candidate_sha256": _sha256(paper_path.read_bytes()),
    }
    try:
        validate_document(
            "literature_method_candidate",
            method,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise LiteratureMethodStagingError(f"LITERATURE_METHOD_SCHEMA_INVALID:{exc}") from exc
    primitive = method.get("primitive_reference")
    if primitive is not None:
        try:
            manifest = registry.manifest(str(primitive))
        except PrimitiveValidationError as exc:
            raise LiteratureMethodStagingError("LITERATURE_METHOD_PRIMITIVE_UNKNOWN") from exc
        if (
            method.get("proposed_primitive") is not None
            or method.get("execution_authority") != "ranking_only"
        ):
            raise LiteratureMethodStagingError(
                "LITERATURE_METHOD_KNOWN_PRIMITIVE_AUTHORITY_INVALID"
            )
        if method.get("required_hook") not in manifest.hooks:
            raise LiteratureMethodStagingError("LITERATURE_METHOD_HOOK_MISMATCH")
    else:
        if not isinstance(method.get("proposed_primitive"), Mapping):
            raise LiteratureMethodStagingError("LITERATURE_METHOD_PROPOSAL_REQUIRED")
        if method.get("execution_authority") != "materialization_required":
            raise LiteratureMethodStagingError(
                "LITERATURE_METHOD_UNKNOWN_PRIMITIVE_AUTHORITY_INVALID"
            )
    dose = method.get("dose")
    if not isinstance(dose, Mapping) or float(dose.get("gpu_hours", 0)) != float(
        method["estimated_gpu_hours"]
    ):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_DOSE_MISMATCH")
    serialized = json.dumps(method, sort_keys=True, ensure_ascii=False).lower()
    if any(marker in serialized for marker in _instruction_markers()):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_INSTRUCTION_CONTENT")
    return method


def _paper_path(row: Mapping[str, Any], *, source: Path) -> Path:
    raw = row.get("path")
    if not isinstance(raw, str) or not raw:
        raise LiteratureMethodStagingError("LITERATURE_METHOD_PAPER_PATH_INVALID")
    raw_path = Path(raw)
    if raw_path.is_symlink():
        raise LiteratureMethodStagingError("LITERATURE_METHOD_PAPER_PATH_INVALID")
    path = raw_path.resolve()
    staging_root = source.parent / "candidates"
    if not path.is_file() or staging_root not in path.parents:
        raise LiteratureMethodStagingError("LITERATURE_METHOD_PAPER_PATH_INVALID")
    return path


def _validate_paper(
    paper: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    try:
        validate_document(
            "literature_candidate",
            paper,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise LiteratureMethodStagingError(
            "LITERATURE_METHOD_PAPER_INVALID"
        ) from exc
    proposed = paper.get("proposed_manifest")
    if (
        not isinstance(proposed, Mapping)
        or proposed.get("state") != "staged"
        or proposed.get("source") != "arxiv"
        or proposed.get("execution_authority") != "shadow_only"
        or paper.get("arxiv_id") != row.get("arxiv_id")
        or paper.get("candidate_id") != row.get("candidate_id")
    ):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_PAPER_INVALID")
    if _contains_execution_field(paper):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_PAPER_COMMAND_FIELD")
    serialized = json.dumps(paper, sort_keys=True, ensure_ascii=False).lower()
    if any(marker in serialized for marker in _instruction_markers()):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_INSTRUCTION_CONTENT")


def _instruction_markers() -> tuple[str, ...]:
    return (
        "ignore previous",
        "system prompt",
        "developer message",
        "<script",
        "```",
    )


def _contains_execution_field(value: object) -> bool:
    forbidden = {
        "command",
        "argv",
        "shell",
        "working_directory",
        "allowed_gpu_indices",
        "environment",
        "gpu_index",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_execution_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_execution_field(child) for child in value)
    return False


def _explicit_primitive(text: str, registry: PrimitiveRegistry) -> str | None:
    for name in registry.names():
        manifest = registry.manifest(name)
        labels = (name, manifest.family, *manifest.literature)
        if any(_normalise(label) in text for label in labels):
            return name
    return None


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:90] or "unknown"


def _load_object(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiteratureMethodStagingError(code) from exc
    if not isinstance(value, dict):
        raise LiteratureMethodStagingError(code)
    return value


def _resume(destination: Path, *, input_hash: str) -> dict[str, object]:
    manifest = _load_object(
        destination / "manifest.json",
        "LITERATURE_METHOD_MANIFEST_INVALID",
    )
    if manifest.get("input_hash") != input_hash:
        raise LiteratureMethodStagingError("LITERATURE_METHOD_INPUT_MISMATCH")
    if (
        manifest.get("artifact_type")
        != "wmloop-literature-method-staging-manifest"
        or manifest.get("state") != "ready"
    ):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_MANIFEST_INVALID")
    report_path = _bound_file(
        destination,
        manifest.get("report_path"),
        "LITERATURE_METHOD_REPORT_INVALID",
    )
    report_bytes = report_path.read_bytes()
    if _sha256(report_bytes) != manifest.get("report_sha256"):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_REPORT_HASH_MISMATCH")
    report = _load_object(report_path, "LITERATURE_METHOD_REPORT_INVALID")
    if (
        report.get("artifact_type") != "wmloop-literature-method-staging"
        or report.get("input_hash") != input_hash
        or report.get("record_count") != manifest.get("record_count")
    ):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_REPORT_INVALID")
    paths = manifest.get("work_order_paths")
    if not isinstance(paths, Mapping):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_WORK_ORDERS_INVALID")
    for raw_path in paths.values():
        work_order_path = _bound_file(
            destination,
            raw_path,
            "LITERATURE_METHOD_WORK_ORDER_INVALID",
        )
        work_order = _load_object(
            work_order_path,
            "LITERATURE_METHOD_WORK_ORDER_INVALID",
        )
        if work_order.get("artifact_type") != "wmloop-primitive-materialization-work-order":
            raise LiteratureMethodStagingError("LITERATURE_METHOD_WORK_ORDER_INVALID")
    return manifest


def _bound_file(root: Path, value: object, code: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LiteratureMethodStagingError(code)
    raw = Path(value)
    if raw.is_symlink():
        raise LiteratureMethodStagingError(code)
    path = raw.resolve()
    if not path.is_file() or root not in path.parents:
        raise LiteratureMethodStagingError(code)
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle(report: Mapping[str, object], destination: Path) -> dict[str, object]:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    try:
        payload = _canonical_json(report)
        _write_json(temporary / "literature-method-staging.json", report)
        work_order_dir = temporary / "work-orders"
        work_order_dir.mkdir(mode=0o700)
        work_order_paths: dict[str, str] = {}
        for method in report["work_orders"]:  # type: ignore[index]
            if not isinstance(method, Mapping):
                raise LiteratureMethodStagingError("LITERATURE_METHOD_WORK_ORDER_INVALID")
            primitive = method["proposed_primitive"]
            assert isinstance(primitive, Mapping)
            filename = f"{primitive['name']}.json"
            _write_json(
                work_order_dir / filename,
                _work_order(
                    method,
                    source_revision=str(report["source_revision"]),
                    registry_digest=str(report["registry_digest"]),
                ),
            )
            work_order_paths[str(method["candidate_id"])] = str(
                destination / "work-orders" / filename
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-literature-method-staging-manifest",
            "state": report["state"],
            "input_hash": report["input_hash"],
            "record_count": report["record_count"],
            "ranking_candidate_count": report["ranking_candidate_count"],
            "materialization_work_order_count": report["materialization_work_order_count"],
            "report_path": str(destination / "literature-method-staging.json"),
            "records": report["records"],
            "ranking_candidates": report["ranking_candidates"],
            "work_order_paths": work_order_paths,
            "claim_boundary": report["claim_boundary"],
            "report_sha256": _sha256(payload),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _work_order(
    method: Mapping[str, Any],
    *,
    source_revision: str,
    registry_digest: str,
) -> dict[str, object]:
    primitive = method["proposed_primitive"]
    assert isinstance(primitive, Mapping)
    primitive_name = str(primitive["name"])
    return {
        "schema_version": 1,
        # Use the existing guarded prompt boundary.  It can render a packet,
        # but it still cannot mutate source or grant execution authority.
        "artifact_type": "wmloop-primitive-materialization-work-order",
        "primitive": primitive_name,
        "source_revision": source_revision,
        "registry_digest": registry_digest,
        "layer": "L3",
        "hooks": [method["required_hook"]],
        "hook_order": {str(method["required_hook"]): 0},
        "targets_failures": method["target_failure_signatures"],
        "params_schema": {"type": "object", "additionalProperties": False},
        "apply_module": f"literature_staging.{primitive_name}",
        "current_admission_state": "materialization_missing",
        "target_admission_state": "closed_loop_runtime_ready",
        "required_gates": [
            "schema_valid",
            "clean_diff_no_frozen_evaluator",
            "primitive_apply_audit_passed",
            "runtime_hook_unit_passed",
            "gpu_training_smoke_passed",
            "human_approved_version_boundary",
        ],
        "allowed_mutation_paths": [
            f"wmloop/primitives/definitions/{primitive_name}/apply.py",
            f"wmloop/primitives/definitions/{primitive_name}/templates/",
            f"tests/test_{primitive_name}.py",
            "vendor/ACWM-Phys/acwm/wmloop_hooks/",
            "vendor/ACWM-Phys/acwm/trainer/train_dynamics.py",
            "vendor/ACWM-Phys/acwm/dynamics/",
        ],
        "forbidden_paths": [
            "eval.py",
            "scripts/eval_all.sh",
            "results/",
            "configs/goal/",
            "runs/m0/protocol/",
        ],
        "literature_method": method,
        "literature_execution_authority": "shadow_only",
        "agent_staging_contract": {
            "session_class": "wmloop.execute.agent_staging.AgentRepairSession",
            "candidate_id": method["candidate_id"],
            "source_revision": source_revision,
            "registry_digest": registry_digest,
            "required_check_labels": [
                "static",
                "offline",
                "canary",
                "shadow_replay",
            ],
            "current_campaign_promotion_allowed": False,
        },
        "promotion_rule": (
            "Only a next-version registry and all static, offline, canary, and "
            "shadow replay gates can admit this method; this packet never does."
        ),
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LiteratureMethodStagingError("LITERATURE_METHOD_SOURCE_REVISION_UNAVAILABLE") from exc
    revision = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise LiteratureMethodStagingError("LITERATURE_METHOD_SOURCE_REVISION_INVALID")
    return revision


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the bounded literature-to-method staging transaction."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("literature_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--failure-signature", action="append", default=[])
    parser.add_argument("--model-family")
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        manifest = run_literature_method_staging(
            literature_manifest=args.literature_manifest,
            output_root=args.output_root,
            repo_root=args.repo_root,
            failure_signatures=tuple(args.failure_signature),
            model_family=args.model_family,
            max_candidates=args.max_candidates,
        )
    except LiteratureMethodStagingError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
