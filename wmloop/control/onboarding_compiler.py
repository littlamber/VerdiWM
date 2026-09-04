"""Compile a passing onboarding admission into a scheduler-ready queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding_admission import (
    OnboardingAdmissionError,
    admission_from_manifest,
    verify_onboarding_admission,
)
from wmloop.control.method_candidate_compiler import compile_method_candidates
from wmloop.execute.experiment_scheduler import (
    ExperimentSchedulerError,
    _validate_batch_semantics,
    plan_candidate_batch,
)


_PLACEHOLDER = re.compile(
    r"\{(?:python|verdiwm_python|verdiwm_root|runtime_path|repo_root|control_root|model_parent|asset:--[A-Za-z0-9_-]+)\}"
)
_RUNTIME_PLACEHOLDERS = {
    "{scratch_dir}",
    "{workspace_root}",
    "{output_root}",
    "{gpu_index}",
    "{gpu_uuid}",
}


class OnboardingCompilerError(RuntimeError):
    """A passing admission could not be compiled safely."""


def compile_and_plan(
    *,
    sidecar_root: Path,
    conformance_root: Path,
    output_root: Path,
    diagnostic_probe_manifest: Path | None = None,
    retrieval_context: Mapping[str, object] | None = None,
    literature_manifest: Path | None = None,
    literature_method_manifest: Path | None = None,
    candidate_catalog: Path | None = None,
    settlement_manifest: Path | None = None,
) -> dict[str, object]:
    """Materialize a frozen template and produce its deterministic queue."""

    sidecar = Path(sidecar_root).expanduser().resolve()
    conformance = Path(conformance_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    sidecar_manifest = _load_json(
        sidecar / "manifest.json", "ONBOARDING_COMPILER_SIDECAR_MANIFEST_INVALID"
    )
    report_path = sidecar / "onboarding-report.json"
    report_bytes = _read_bound_file(
        report_path,
        str(sidecar_manifest.get("report_sha256", "")),
        "ONBOARDING_COMPILER_REPORT_HASH_MISMATCH",
    )
    report = _decode_object(report_bytes, "ONBOARDING_COMPILER_REPORT_INVALID")
    repo = Path(str(report.get("repo_root", ""))).resolve()
    _validate_output(destination, repo=repo, sidecar=sidecar, conformance=conformance)

    admission = admission_from_manifest(conformance)
    try:
        receipt = verify_onboarding_admission(admission, expected_repo_root=repo)
    except OnboardingAdmissionError as exc:
        raise OnboardingCompilerError(
            f"ONBOARDING_COMPILER_ADMISSION_INVALID:{exc}"
        ) from exc
    evaluator = report.get("evaluator_contract")
    if not isinstance(evaluator, Mapping) or evaluator.get("state") != "ready":
        raise OnboardingCompilerError("ONBOARDING_COMPILER_EVALUATOR_NOT_READY")
    template_path = Path(str(evaluator.get("scheduler_template_path", ""))).resolve()
    template_sha256 = str(evaluator.get("scheduler_template_sha256", ""))
    template_bytes = _read_bound_file(
        template_path,
        template_sha256,
        "ONBOARDING_COMPILER_TEMPLATE_HASH_MISMATCH",
    )
    template = _decode_object(template_bytes, "ONBOARDING_COMPILER_TEMPLATE_INVALID")
    probe_bytes = b""
    probe = None
    if diagnostic_probe_manifest is not None:
        probe_path = Path(diagnostic_probe_manifest).resolve(strict=True)
        probe_bytes = probe_path.read_bytes()
        probe = _decode_object(
            probe_bytes, "ONBOARDING_COMPILER_PROBE_MANIFEST_INVALID"
        )
        if probe.get("state") != "settled" or probe.get("verdict") != "PASS":
            raise OnboardingCompilerError("ONBOARDING_COMPILER_PROBE_NOT_SETTLED")
    literature_bytes = b""
    literature = None
    if literature_manifest is not None:
        literature_path = Path(literature_manifest).resolve(strict=True)
        literature_bytes = literature_path.read_bytes()
        literature = _decode_object(
            literature_bytes, "ONBOARDING_COMPILER_LITERATURE_MANIFEST_INVALID"
        )
    literature_method_bytes = b""
    literature_methods = None
    if literature_method_manifest is not None:
        literature_method_path = Path(literature_method_manifest).resolve(strict=True)
        literature_method_bytes = literature_method_path.read_bytes()
        literature_methods = _decode_object(
            literature_method_bytes,
            "ONBOARDING_COMPILER_LITERATURE_METHOD_MANIFEST_INVALID",
        )
        if (
            literature_methods.get("artifact_type")
            != "wmloop-literature-method-staging-manifest"
        ):
            raise OnboardingCompilerError(
                "ONBOARDING_COMPILER_LITERATURE_METHOD_MANIFEST_INVALID"
            )
    candidate_catalog_bytes = b""
    candidate_catalog_path = None
    if candidate_catalog is not None:
        candidate_catalog_path = Path(candidate_catalog).resolve(strict=True)
        candidate_catalog_bytes = candidate_catalog_path.read_bytes()
    settlement_bytes = b""
    settlement_manifest_path = None
    if settlement_manifest is not None:
        settlement_manifest_path = Path(settlement_manifest).resolve(strict=True)
        settlement_bytes = _settlement_bundle_bytes(settlement_manifest_path)
    retrieval = dict(retrieval_context or {"state": "cold_start", "matches": []})
    input_hash = _sha256(
        report_bytes
        + b"\0"
        + _canonical_json(receipt)
        + b"\0"
        + template_bytes
        + b"\0"
        + probe_bytes
        + b"\0"
        + literature_bytes
        + b"\0"
        + literature_method_bytes
        + b"\0"
        + candidate_catalog_bytes
        + b"\0"
        + settlement_bytes
        + b"\0"
        + _canonical_json(retrieval)
    )

    if destination.exists() or destination.is_symlink():
        manifest = _resume(destination, input_hash=input_hash)
        queue = plan_candidate_batch(
            batch_path=destination / "candidate-batch.json",
            output_root=destination / "queue",
            workspace_root=repo,
        )
        return _settle_queue_admission(destination, manifest=manifest, queue=queue)

    values = _materialization_values(report, repo=repo)
    batch = _materialize(template, values)
    if not isinstance(batch, dict):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_TEMPLATE_OBJECT_REQUIRED")
    batch["onboarding_admission"] = admission
    method_compilation = None
    if candidate_catalog_path is not None:
        method_compilation = compile_method_candidates(
            batch=batch,
            catalog_path=candidate_catalog_path,
            diagnostic_probe=probe,
            settlement_manifest=settlement_manifest_path,
            literature_methods=literature_methods,
            materialization_values=values,
        )
        batch["method_candidate_compilation"] = {
            "manifest_path": str(destination / "method-candidates" / "manifest.json"),
            "manifest_sha256": _sha256(_canonical_json(method_compilation)),
            "compiled_candidate_ids": [
                str(row["candidate_id"])
                for row in method_compilation["compiled_candidates"]
                if isinstance(row, Mapping)
            ],
            "capability_gap_count": method_compilation["capability_gap_count"],
            "historical_constraint_count": method_compilation[
                "historical_constraint_count"
            ],
        }
    _apply_retrieval_context(
        batch,
        probe=probe,
        retrieval=retrieval,
        literature=literature,
        literature_methods=literature_methods,
        probe_manifest_path=diagnostic_probe_manifest,
    )
    if method_compilation is not None:
        _prefer_compiled_method_candidates(
            batch,
            compilation=method_compilation,
        )
    try:
        validate_document("auto_experiment_candidate_batch", batch)
        _validate_batch_semantics(batch, workspace_root=repo)
    except (ContractValidationError, ExperimentSchedulerError) as exc:
        raise OnboardingCompilerError(
            f"ONBOARDING_COMPILER_BATCH_INVALID:{exc}"
        ) from exc

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        _write_json(temporary / "candidate-batch.json", batch)
        _write_json(temporary / "onboarding-admission.json", admission)
        if method_compilation is not None:
            _write_json(
                temporary / "method-candidates" / "manifest.json",
                method_compilation,
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-onboarded-candidate-compilation",
            "state": "ready",
            "input_hash": input_hash,
            "repo_root": str(repo),
            "candidate_batch_path": str(destination / "candidate-batch.json"),
            "candidate_batch_sha256": _sha256(_canonical_json(batch)),
            "admission_receipt_path": admission["receipt_path"],
            "admission_receipt_sha256": admission["receipt_sha256"],
            "queue_path": str(destination / "queue" / "queue.json"),
            "optimization_launch_allowed": True,
            "literature_method_manifest_path": (
                str(Path(literature_method_manifest).resolve())
                if literature_method_manifest is not None
                else None
            ),
            "method_candidate_compilation_path": (
                str(destination / "method-candidates" / "manifest.json")
                if method_compilation is not None
                else None
            ),
            "method_candidate_compilation_sha256": (
                _sha256(_canonical_json(method_compilation))
                if method_compilation is not None
                else None
            ),
            "compiled_method_candidate_count": (
                method_compilation["compiled_candidate_count"]
                if method_compilation is not None
                else 0
            ),
            "method_capability_gap_count": (
                method_compilation["capability_gap_count"]
                if method_compilation is not None
                else 0
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    queue = plan_candidate_batch(
        batch_path=destination / "candidate-batch.json",
        output_root=destination / "queue",
        workspace_root=repo,
    )
    return _settle_queue_admission(destination, manifest=manifest, queue=queue)


def _settle_queue_admission(
    destination: Path,
    *,
    manifest: Mapping[str, object],
    queue: Mapping[str, object],
) -> dict[str, object]:
    selected = queue.get("selected")
    routing_blocked = queue.get("routing_blocked")
    selected_count = len(selected) if isinstance(selected, list) else 0
    blocked_count = len(routing_blocked) if isinstance(routing_blocked, list) else 0
    settled = dict(manifest)
    settled["eligible_candidate_count"] = selected_count
    settled["routing_blocked_candidate_count"] = blocked_count
    settled["optimization_launch_allowed"] = selected_count > 0
    settled["state"] = "ready" if selected_count > 0 else "blocked"
    settled["blockers"] = (
        []
        if selected_count > 0
        else [
            (
                "NO_CANDIDATE_ROUTING_ELIGIBLE"
                if blocked_count > 0
                else "NO_CANDIDATE_SELECTED"
            )
        ]
    )
    _write_json(destination / "manifest.json", settled)
    return settled


def _apply_retrieval_context(
    batch: dict[str, object],
    *,
    probe: Mapping[str, object] | None,
    retrieval: Mapping[str, object],
    literature: Mapping[str, object] | None,
    literature_methods: Mapping[str, object] | None,
    probe_manifest_path: Path | None,
) -> None:
    """Attach provenance and a bounded ordering prior to a candidate batch."""

    matches = [row for row in retrieval.get("matches", []) if isinstance(row, Mapping)]
    literature_rows = (
        [row for row in literature.get("rows", []) if isinstance(row, Mapping)]
        if literature is not None
        else []
    )
    if probe is not None and probe_manifest_path is not None:
        batch["diagnostic_probe"] = {
            "manifest_path": str(probe_manifest_path.resolve()),
            "manifest_sha256": _sha256(probe_manifest_path.read_bytes()),
            "failure_signatures": list(probe.get("failure_signatures", [])),
            "retrieval_state": str(retrieval.get("state") or "cold_start"),
        }
    if literature is not None:
        batch["literature_sources"] = literature_rows
    method_rows = (
        [
            row
            for row in literature_methods.get("records", [])
            if isinstance(row, Mapping)
        ]
        if literature_methods is not None
        else []
    )
    irg_guided = retrieval.get("irg_guided")
    if isinstance(irg_guided, Mapping):
        # Keep the complete IRG plan in the immutable candidate batch so every
        # staged method can be traced back to the model-conditioned bottleneck
        # that motivated retrieval.  The plan remains ranking/shadow-only.
        batch["irg_guided_retrieval"] = dict(irg_guided)
    _apply_diagnostic_routing(batch, probe=probe, retrieval=retrieval)
    if method_rows:
        batch["literature_method_sources"] = method_rows
    ranking_method_rows = [
        row
        for row in method_rows
        if isinstance(row.get("primitive_reference"), str)
        and row.get("execution_authority") == "ranking_only"
    ]
    if not matches and not literature_rows and not method_rows:
        return
    scoring = batch.get("scoring")
    if isinstance(scoring, dict):
        scoring.setdefault("retrieval_weight", 0.25)
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        keys = candidate.get("retrieval_keys")
        key_signatures = (
            {
                str(value)
                for value in keys.get("failure_signatures", [])
                if isinstance(value, str)
            }
            if isinstance(keys, Mapping)
            else set()
        )
        key_primitive = keys.get("primitive") if isinstance(keys, Mapping) else None
        candidate_matches = []
        for row in matches:
            signature = str(row.get("failure_signature") or "")
            primitive = row.get("primitive")
            signature_match = not key_signatures or signature in key_signatures
            primitive_match = key_primitive is None or primitive == key_primitive
            if signature_match and primitive_match:
                candidate_matches.append(dict(row))
        paper_ids = {
            str(value)
            for value in candidate.get("literature_arxiv_ids", [])
            if isinstance(value, str)
        }
        candidate_papers = [
            dict(row)
            for row in literature_rows
            if not paper_ids or str(row.get("arxiv_id") or "") in paper_ids
        ]
        candidate_methods = [
            dict(row)
            for row in ranking_method_rows
            if (
                (
                    key_primitive is not None
                    and row.get("primitive_reference") == key_primitive
                )
                or (
                    bool(paper_ids)
                    and isinstance(row.get("source"), Mapping)
                    and str(row["source"].get("arxiv_id") or "") in paper_ids
                )
            )
        ]
        if not candidate_matches and not candidate_papers and not candidate_methods:
            continue
        positive = sum(str(row.get("verdict")) == "PASS" for row in candidate_matches)
        method_bonus = min(0.3, 0.1 * len(candidate_methods))
        prior = min(
            1.0, ((0.2 + 0.1 * positive) if candidate_matches else 0.1) + method_bonus
        )
        candidate["retrieval_prior"] = prior
        candidate["retrieval_matches"] = [
            *candidate_matches,
            *candidate_papers,
            *candidate_methods,
        ]


def _apply_diagnostic_routing(
    batch: dict[str, object],
    *,
    probe: Mapping[str, object] | None,
    retrieval: Mapping[str, object] | None = None,
) -> None:
    """Require every post-probe candidate to name the failure it addresses.

    A candidate without an explicit signature route is retained in the
    immutable compiled batch for auditability, but marked blocked so it cannot
    consume GPU budget or create an unconnected experiment receipt.
    """

    observed: set[str] = set()
    if probe is not None:
        observed.update(
            str(value)
            for value in probe.get("failure_signatures", [])
            if isinstance(value, str) and value
        )
    irg_guided = (retrieval or {}).get("irg_guided")
    if isinstance(irg_guided, Mapping):
        request = irg_guided.get("request")
        if isinstance(request, Mapping):
            observed.update(
                str(value)
                for value in request.get("failure_signatures", [])
                if isinstance(value, str) and value
            )
    if not observed:
        return
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        keys = candidate.get("retrieval_keys")
        declared = (
            {
                str(value)
                for value in keys.get("failure_signatures", [])
                if isinstance(value, str) and value
            }
            if isinstance(keys, Mapping)
            else set()
        )
        matched = sorted(observed & declared)
        if matched:
            candidate["routing_admission"] = {
                "state": "eligible",
                "reason": "candidate_declares_observed_failure_signature",
                "matched_failure_signatures": matched,
            }
        else:
            reason = (
                "candidate_missing_failure_signature_route"
                if not declared
                else "candidate_failure_signature_mismatch"
            )
            candidate["routing_admission"] = {
                "state": "blocked",
                "reason": reason,
                "matched_failure_signatures": [],
            }


def _prefer_compiled_method_candidates(
    batch: dict[str, object], *, compilation: Mapping[str, object]
) -> None:
    compiled_rows = [
        row
        for row in compilation.get("compiled_candidates", [])
        if isinstance(row, Mapping)
    ]
    compiled_ids = {
        str(row.get("candidate_id"))
        for row in compiled_rows
        if isinstance(row.get("candidate_id"), str)
    }
    compiled_signatures = {
        str(signature)
        for row in compiled_rows
        for signature in row.get("failure_signatures", [])
        if isinstance(signature, str)
    }
    if not compiled_ids or not compiled_signatures:
        return
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in compiled_ids:
            continue
        routing = candidate.get("routing_admission")
        if not isinstance(routing, Mapping) or routing.get("state") != "eligible":
            continue
        matched = {
            str(value)
            for value in routing.get("matched_failure_signatures", [])
            if isinstance(value, str)
        }
        overlap = sorted(matched & compiled_signatures)
        if overlap:
            candidate["routing_admission"] = {
                "state": "blocked",
                "reason": "superseded_by_compiled_method_candidate",
                "matched_failure_signatures": overlap,
            }


def _materialization_values(
    report: Mapping[str, object], *, repo: Path
) -> dict[str, str]:
    runtime = report.get("runtime")
    connector = report.get("connector")
    if not isinstance(runtime, Mapping) or not isinstance(connector, Mapping):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_CONNECTOR_INVALID")
    selected_python = runtime.get("selected_python")
    if not isinstance(selected_python, str) or not Path(selected_python).is_file():
        raise OnboardingCompilerError("ONBOARDING_COMPILER_RUNTIME_INVALID")
    runtime_bin = Path(selected_python).parent
    runtime_path = [str(runtime_bin)]
    runtime_path.extend(
        str(path)
        for path in sorted(
            runtime_bin.parent.glob("lib/python*/site-packages/imageio_ffmpeg/binaries")
        )
        if path.is_dir()
    )
    runtime_path.append(os.environ.get("PATH", ""))
    values = {
        "{python}": selected_python,
        "{verdiwm_python}": str(Path(__import__("sys").executable).absolute()),
        "{verdiwm_root}": str(Path(__file__).resolve().parents[2]),
        "{runtime_path}": os.pathsep.join(runtime_path),
        "{repo_root}": str(repo),
        "{control_root}": str(Path(__file__).resolve().parents[2]),
        "{model_parent}": str(repo.parent),
    }
    asset_rows = connector.get("asset_bindings")
    if not isinstance(asset_rows, list):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_ASSETS_INVALID")
    for row in asset_rows:
        if not isinstance(row, Mapping) or row.get("state") != "discovered":
            continue
        parameter = str(row.get("parameter", ""))
        resolved_path = row.get("resolved_path")
        if parameter and isinstance(resolved_path, str):
            values[f"{{asset:{parameter}}}"] = resolved_path
    return values


def _materialize(value: object, values: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {str(key): _materialize(child, values) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize(child, values) for child in value]
    if not isinstance(value, str):
        return value
    materialized = value
    for placeholder in _PLACEHOLDER.findall(value):
        replacement = values.get(placeholder)
        if replacement is None:
            raise OnboardingCompilerError(
                f"ONBOARDING_COMPILER_PLACEHOLDER_UNBOUND:{placeholder}"
            )
        materialized = materialized.replace(placeholder, replacement)
    remaining = set(re.findall(r"\{[^{}]+\}", materialized))
    if remaining - _RUNTIME_PLACEHOLDERS:
        raise OnboardingCompilerError("ONBOARDING_COMPILER_PLACEHOLDER_INVALID")
    return materialized


def _resume(destination: Path, *, input_hash: str) -> dict[str, object]:
    if destination.is_symlink() or not destination.is_dir():
        raise OnboardingCompilerError("ONBOARDING_COMPILER_OUTPUT_INVALID")
    manifest = _load_json(
        destination / "manifest.json", "ONBOARDING_COMPILER_MANIFEST_INVALID"
    )
    if manifest.get("input_hash") != input_hash:
        raise OnboardingCompilerError("ONBOARDING_COMPILER_INPUT_MISMATCH")
    batch_path = destination / "candidate-batch.json"
    if _sha256(batch_path.read_bytes()) != manifest.get("candidate_batch_sha256"):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_BATCH_HASH_MISMATCH")
    method_path = manifest.get("method_candidate_compilation_path")
    method_hash = manifest.get("method_candidate_compilation_sha256")
    if method_path is not None:
        path = Path(str(method_path))
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path.read_bytes()) != method_hash
        ):
            raise OnboardingCompilerError(
                "ONBOARDING_COMPILER_METHOD_COMPILATION_HASH_MISMATCH"
            )
    return manifest


def _settlement_bundle_bytes(manifest_path: Path) -> bytes:
    records_root = manifest_path.parent / "records"
    payload = bytearray(manifest_path.read_bytes())
    if not records_root.is_dir() or records_root.is_symlink():
        raise OnboardingCompilerError("ONBOARDING_COMPILER_SETTLEMENT_RECORDS_INVALID")
    for path in sorted(records_root.glob("*.json")):
        if not path.is_file() or path.is_symlink():
            raise OnboardingCompilerError(
                "ONBOARDING_COMPILER_SETTLEMENT_RECORDS_INVALID"
            )
        payload.extend(b"\0")
        payload.extend(path.name.encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(path.read_bytes())
    return bytes(payload)


def _validate_output(
    destination: Path, *, repo: Path, sidecar: Path, conformance: Path
) -> None:
    for root, code in (
        (repo, "ONBOARDING_COMPILER_OUTPUT_INSIDE_SOURCE"),
        (sidecar, "ONBOARDING_COMPILER_OUTPUT_INSIDE_SIDECAR"),
        (conformance, "ONBOARDING_COMPILER_OUTPUT_INSIDE_CONFORMANCE"),
    ):
        if destination == root or root in destination.parents:
            raise OnboardingCompilerError(code)


def _read_bound_file(path: Path, expected_sha256: str, code: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise OnboardingCompilerError(code)
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise OnboardingCompilerError(code)
    return payload


def _decode_object(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnboardingCompilerError(code) from exc
    if not isinstance(value, dict):
        raise OnboardingCompilerError(code)
    return value


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        return _decode_object(path.read_bytes(), code)
    except OSError as exc:
        raise OnboardingCompilerError(code) from exc


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--conformance-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = compile_and_plan(
            sidecar_root=args.sidecar_root,
            conformance_root=args.conformance_root,
            output_root=args.output_root,
        )
    except (OnboardingCompilerError, OnboardingAdmissionError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
