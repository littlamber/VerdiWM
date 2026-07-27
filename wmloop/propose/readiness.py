"""M3 proposal-generation readiness report over real failure reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.generator import (
    CURRENT_LIBRARY_VERSION,
    LLMClient,
    ProposalContext,
    ProposalGenerationError,
    ProposalGenerator,
)


class ProposalReadinessError(RuntimeError):
    """Proposal readiness evidence could not be generated."""


def run_proposal_readiness(
    *,
    repo_root: Path,
    failure_reports: Sequence[Path],
    goal_config: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    strict_required_reports: int = 3,
) -> dict[str, object]:
    """Generate legal proposals for the supplied real reports and write evidence."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProposalReadinessError("PROPOSAL_READINESS_OUTPUT_EXISTS")
    if not failure_reports:
        raise ProposalReadinessError("PROPOSAL_READINESS_FAILURE_REPORTS_EMPTY")
    if strict_required_reports < 1:
        raise ProposalReadinessError("PROPOSAL_READINESS_REQUIRED_COUNT_INVALID")
    registry = PrimitiveRegistry.from_root(root)
    goal = load_yaml_document(goal_config)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(cas_storage_root)
    generator = ProposalGenerator(_DeterministicReadinessClient(), max_attempts=3)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        proposals_dir = temporary / "proposals"
        prompts_dir = temporary / "prompts"
        responses_dir = temporary / "responses"
        for directory in (proposals_dir, prompts_dir, responses_dir):
            directory.mkdir(mode=0o700)
        rows = []
        legal_count = 0
        for index, report_path in enumerate(failure_reports, start=1):
            row = _generate_one(
                index=index,
                report_path=Path(report_path),
                root=root,
                goal=goal,
                registry=registry,
                archive=archive,
                cas=cas,
                generator=generator,
                proposals_dir=proposals_dir,
                prompts_dir=prompts_dir,
                responses_dir=responses_dir,
            )
            rows.append(row)
            if row["state"] == "legal":
                legal_count += 1
        retry_demo = _run_retry_demo(
            first_report=Path(failure_reports[0]),
            goal=goal,
            registry=registry,
            archive_statistics=archive.archive_statistics() if archive is not None else {},
        )
        strict_pass = legal_count >= strict_required_reports and bool(retry_demo["passed"])
        public_rows = _public_rows(rows, temporary=temporary, destination=destination)
        blockers = []
        if legal_count < strict_required_reports:
            blockers.append(
                {
                    "reason": "raw_failure_report_count_below_T3_1_required_count",
                    "legal_proposal_count": legal_count,
                    "strict_required_reports": strict_required_reports,
                }
            )
        if not retry_demo["passed"]:
            blockers.append({"reason": "invalid_retry_demo_failed"})
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-proposal-readiness-report",
            "state": "ready" if strict_pass else "partial",
            "strict_t3_1_pass": strict_pass,
            "strict_required_reports": strict_required_reports,
            "failure_report_count": len(failure_reports),
            "legal_proposal_count": legal_count,
            "registry_digest": registry.digest(),
            "goal_id": goal.get("goal_id"),
            "goal_config_path": str(Path(goal_config).resolve()),
            "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "rows": public_rows,
            "invalid_retry_demo": retry_demo,
            "blockers": blockers,
            "limitations": [
                "This report validates constrained proposal generation over supplied real failure reports; it does not claim model-quality improvement.",
                "Strict T3.1 requires at least three real failure reports. A partial state records the exact coverage shortfall instead of fabricating reports.",
                "The readiness client is deterministic and exercises the LLMClient contract boundary; it is not an external API model call.",
            ],
        }
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        _write_bytes_atomic(temporary / "proposal-readiness.json", report_bytes)
        _write_bytes_atomic(temporary / "proposal-readiness.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-proposal-readiness-manifest",
            "state": report["state"],
            "strict_t3_1_pass": strict_pass,
            "failure_report_count": len(failure_reports),
            "legal_proposal_count": legal_count,
            "strict_required_reports": strict_required_reports,
            "report_path": str(destination / "proposal-readiness.json"),
            "markdown_path": str(destination / "proposal-readiness.md"),
            "proposal_paths": [str(row["proposal_path"]) for row in public_rows if row["state"] == "legal"],
            "cas_refs": {
                "proposal_readiness_json": report_ref,
                "proposal_readiness_markdown": markdown_ref,
            },
            "blockers": blockers,
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


def _generate_one(
    *,
    index: int,
    report_path: Path,
    root: Path,
    goal: Mapping[str, Any],
    registry: PrimitiveRegistry,
    archive: ArchiveStore | None,
    cas: ContentAddressedStore,
    generator: ProposalGenerator,
    proposals_dir: Path,
    prompts_dir: Path,
    responses_dir: Path,
) -> dict[str, object]:
    failure = _load_failure_report(report_path)
    try:
        generated = generator.generate(
            ProposalContext(
                failure_report=failure,
                goal_spec=goal,
                archive_statistics=archive.archive_statistics() if archive is not None else {},
                registry=registry,
            )
        )
    except ProposalGenerationError as exc:
        return {
            "state": "blocked",
            "failure_report_path": str(report_path.resolve()),
            "environment": failure.get("env"),
            "dominant_failure": failure.get("dominant_failure"),
            "reason": str(exc),
        }
    stem = f"{index:02d}-{_safe_name(str(failure['env']))}"
    proposal_path = proposals_dir / f"{stem}-proposal.json"
    prompt_path = prompts_dir / f"{stem}-prompt.json"
    responses_path = responses_dir / f"{stem}-responses.json"
    proposal_bytes = _canonical_json_bytes(generated.proposal)
    prompt_bytes = generated.prompt.encode("utf-8") + b"\n"
    responses_bytes = _canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact_type": "wmloop-proposal-readiness-responses",
            "raw_responses": list(generated.raw_responses),
        }
    )
    _write_bytes_atomic(proposal_path, proposal_bytes)
    _write_bytes_atomic(prompt_path, prompt_bytes)
    _write_bytes_atomic(responses_path, responses_bytes)
    refs = {
        "failure_report": cas.put_bytes(report_path.read_bytes(), media_type="application/json").uri,
        "proposal": cas.put_bytes(proposal_bytes, media_type="application/json").uri,
        "prompt": cas.put_bytes(prompt_bytes, media_type="application/json").uri,
        "raw_responses": cas.put_bytes(responses_bytes, media_type="application/json").uri,
    }
    if archive is not None:
        for ref in refs.values():
            archive.record_artifact_reference(ref)
    return {
        "state": "legal",
        "failure_report_path": str(report_path.resolve()),
        "environment": failure["env"],
        "dominant_failure": failure["dominant_failure"],
        "proposal_id": generated.proposal["proposal_id"],
        "primitive_names": [row["primitive"] for row in generated.proposal["interventions"]],
        "attempts": generated.attempts,
        "proposal_path": str(proposal_path),
        "prompt_path": str(prompt_path),
        "responses_path": str(responses_path),
        "cas_refs": refs,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
    }


def _public_rows(rows: Sequence[Mapping[str, object]], *, temporary: Path, destination: Path) -> list[dict[str, object]]:
    public = []
    for row in rows:
        item = dict(row)
        for key in ("proposal_path", "prompt_path", "responses_path"):
            value = item.get(key)
            if isinstance(value, str):
                path = Path(value)
                try:
                    item[key] = str(destination / path.relative_to(temporary))
                except ValueError:
                    item[key] = value
        public.append(item)
    return public


class _DeterministicReadinessClient:
    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = packet["allowed_primitives"]
        if not isinstance(allowed, list) or not allowed:
            return json.dumps({"error": "no_allowed_primitives"}, sort_keys=True)
        primitive = allowed[0]
        params = _example_params(primitive["params_schema"], failure=failure)
        goal = packet["goal_spec"]
        horizons = goal.get("horizons")
        horizon = max(horizons) if isinstance(horizons, list) and horizons else 64
        proposal = {
            "proposal_id": f"m3-proposal-readiness-{_safe_name(str(failure['env']))}-{primitive['name']}",
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [
                {
                    "layer": primitive["layer"],
                    "primitive": primitive["name"],
                    "params": params,
                }
            ],
            "falsifiable_prediction": {
                "metric": str(goal.get("primary_objective") or "auc_psnr_16_64"),
                "horizon": int(horizon),
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(primitive["estimated_gpu_hours"]),
            "rationale_ref": f"m3_proposal_readiness#{failure['dominant_failure']}->{primitive['name']}",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class _RetryDemoClient:
    def __init__(self) -> None:
        self._delegate = _DeterministicReadinessClient()
        self._calls = 0

    def complete(self, prompt: str) -> str:
        self._calls += 1
        if self._calls == 1:
            return "not json"
        return self._delegate.complete(prompt)


def _run_retry_demo(
    *,
    first_report: Path,
    goal: Mapping[str, Any],
    registry: PrimitiveRegistry,
    archive_statistics: Mapping[str, Any],
) -> dict[str, object]:
    failure = _load_failure_report(first_report)
    generator = ProposalGenerator(_RetryDemoClient(), max_attempts=2)
    try:
        generated = generator.generate(
            ProposalContext(
                failure_report=failure,
                goal_spec=goal,
                archive_statistics=archive_statistics,
                registry=registry,
            )
        )
    except ProposalGenerationError as exc:
        return {"passed": False, "reason": str(exc), "attempts": 2}
    return {
        "passed": generated.attempts == 2,
        "attempts": generated.attempts,
        "proposal_id": generated.proposal["proposal_id"],
        "first_response_invalid_json": True,
    }


def _example_params(schema: Mapping[str, Any], *, failure: Mapping[str, Any]) -> dict[str, object]:
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise ProposalReadinessError("PROPOSAL_READINESS_PARAMS_SCHEMA_INVALID")
    return {str(name): _example_value(str(name), properties[name], failure=failure) for name in required}


def _example_value(name: str, schema: object, *, failure: Mapping[str, Any]) -> object:
    if not isinstance(schema, Mapping):
        raise ProposalReadinessError("PROPOSAL_READINESS_PARAM_SCHEMA_INVALID")
    schema_type = schema.get("type")
    if schema_type == "string":
        if name == "condition":
            ood = failure.get("ood_profile")
            if isinstance(ood, Mapping) and isinstance(ood.get("worst_ood_condition"), str):
                return ood["worst_ood_condition"]
        return "readiness"
    if schema_type == "integer":
        minimum = schema.get("minimum", 1)
        exclusive = schema.get("exclusiveMinimum")
        value = int(exclusive) + 1 if isinstance(exclusive, int) else int(minimum)
        maximum = schema.get("maximum")
        if isinstance(maximum, int):
            value = min(value, maximum)
        return max(value, 1)
    if schema_type == "number":
        if isinstance(schema.get("exclusiveMinimum"), (int, float)):
            value = float(schema["exclusiveMinimum"]) + 0.1
        elif isinstance(schema.get("minimum"), (int, float)):
            value = float(schema["minimum"]) + 0.1
        else:
            value = 0.1
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)):
            value = min(value, float(maximum))
        if not math.isfinite(value):
            raise ProposalReadinessError("PROPOSAL_READINESS_PARAM_VALUE_INVALID")
        return value
    raise ProposalReadinessError("PROPOSAL_READINESS_PARAM_TYPE_UNSUPPORTED")


def _load_failure_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalReadinessError("PROPOSAL_READINESS_FAILURE_REPORT_INVALID") from exc
    if not isinstance(payload, dict):
        raise ProposalReadinessError("PROPOSAL_READINESS_FAILURE_REPORT_INVALID")
    try:
        validate_document("failure_report", payload)
    except ContractValidationError as exc:
        raise ProposalReadinessError("PROPOSAL_READINESS_FAILURE_REPORT_SCHEMA_INVALID") from exc
    return payload


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# M3 Proposal Readiness",
        "",
        f"State: `{report['state']}`",
        f"Strict T3.1 pass: `{report['strict_t3_1_pass']}`",
        f"Legal proposals: `{report['legal_proposal_count']}/{report['strict_required_reports']}`",
        f"Invalid retry demo: `{report['invalid_retry_demo']}`",
        "",
        "| Environment | Dominant failure | State | Proposal | Primitives | Attempts |",
        "|:--|:--|:--|:--|:--|--:|",
    ]
    for row in report["rows"]:  # type: ignore[index]
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {env} | {failure} | {state} | {proposal} | {primitives} | {attempts} |".format(
                env=row.get("environment"),
                failure=row.get("dominant_failure"),
                state=row.get("state"),
                proposal=row.get("proposal_id", ""),
                primitives=",".join(str(item) for item in row.get("primitive_names", [])),
                attempts=row.get("attempts", ""),
            )
        )
    blockers = report.get("blockers")
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:  # type: ignore[assignment]
            lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    return safe.strip("_") or "item"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProposalReadinessError("PROPOSAL_READINESS_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="generate M3 proposal-readiness evidence")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--failure-report", type=Path, nargs="+", required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--strict-required-reports", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_proposal_readiness(
            repo_root=args.repo_root,
            failure_reports=tuple(args.failure_report),
            goal_config=args.goal_config,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            strict_required_reports=args.strict_required_reports,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
