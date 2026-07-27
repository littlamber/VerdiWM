"""Read-only M4 completion gate for the v2.1 P0 criteria."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class M4CompletionGateError(RuntimeError):
    """The M4 completion gate failed closed."""


def run_m4_completion_gate(
    *,
    archive_db: Path,
    phase_gate_manifest: Path,
    output_root: Path,
    prior_convergence_manifest: Path | None = None,
    verifier_meta_validation_manifest: Path | None = None,
    settled_trial_target: int = 150,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a non-launching M4 completion receipt."""

    if settled_trial_target < 1:
        raise M4CompletionGateError("M4_COMPLETION_GATE_TARGET_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M4CompletionGateError("M4_COMPLETION_GATE_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db)
    cas_storage_root = Path(cas_root) if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(cas_storage_root)
    sources = {
        "phase_gate": _load_source(phase_gate_manifest, cas=cas, archive=archive),
    }
    if prior_convergence_manifest is not None:
        sources["t4_4_prior_convergence"] = _load_source(prior_convergence_manifest, cas=cas, archive=archive)
    if verifier_meta_validation_manifest is not None:
        sources["t4_5_verifier_meta_validation"] = _load_source(
            verifier_meta_validation_manifest,
            cas=cas,
            archive=archive,
        )
    archive_statistics = archive.archive_statistics()
    requirements = _requirements(
        archive_statistics=archive_statistics,
        sources=sources,
        settled_trial_target=settled_trial_target,
    )
    ready = all(item["passed"] is True for item in requirements.values())
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m4-completion-gate",
        "state": "ready" if ready else "blocked",
        "m4_completion_allowed": ready,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "settled_trial_target": settled_trial_target,
        "archive_statistics": archive_statistics,
        "requirements": requirements,
        "blockers": _blockers(requirements),
        "sources": {name: source["summary"] for name, source in sources.items()},
        "next_actions": _next_actions(requirements),
        "limitations": [
            "This gate judges M4 completion, not M4 launch authorization.",
            "The v2.1 P0 criteria require settled-trial volume plus T4.4 and T4.5 artifacts before M4 can be reported as passed.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _requirements(
    *,
    archive_statistics: Mapping[str, int],
    sources: Mapping[str, Mapping[str, object]],
    settled_trial_target: int,
) -> dict[str, dict[str, object]]:
    phase_gate = _payload(sources, "phase_gate")
    settled_trials = int(archive_statistics.get("settled_trials", 0))
    t44 = _payload_or_none(sources, "t4_4_prior_convergence")
    t45 = _payload_or_none(sources, "t4_5_verifier_meta_validation")
    return {
        "strict_phase_gate_ready": {
            "passed": phase_gate.get("artifact_type") == "wmloop-strict-phase-gate-manifest"
            and phase_gate.get("state") == "ready"
            and phase_gate.get("m4_launch_allowed") is True,
            "expected": "The strict M4 launch gate has already passed under the active protocol.",
            "observed": {
                "artifact_type": phase_gate.get("artifact_type"),
                "state": phase_gate.get("state"),
                "m4_launch_allowed": phase_gate.get("m4_launch_allowed"),
            },
        },
        "m4_settled_trial_volume": {
            "passed": settled_trials >= settled_trial_target,
            "expected": f"At least {settled_trial_target} trial records are settled and visible in the archive.",
            "observed": {"settled_trials": settled_trials, "target": settled_trial_target},
        },
        "T4_4_prior_vs_cold_start_convergence": {
            "passed": _t44_passes(t44),
            "expected": "T4.4 three-arm prior/cold-start/shuffled-prior hypothesis test is settled with a positive or negative outcome.",
            "observed": _t44_observed(t44),
        },
        "T4_5_verifier_meta_validation": {
            "passed": _t45_passes(t45, settled_trials=settled_trials),
            "expected": "T4.5 verifier meta-validation covers all settled trials with confusion matrix and real INCONCLUSIVE/VOID cases.",
            "observed": _t45_observed(t45, settled_trials=settled_trials),
        },
    }


def _t44_passes(manifest: Mapping[str, Any] | None) -> bool:
    result_settled = _t44_result_settled(manifest)
    return (
        manifest is not None
        and manifest.get("artifact_type") == "wmloop-t4-4-prior-convergence-manifest"
        and manifest.get("state") == "ready"
        and {"prior", "cold_start", "shuffled_prior"}.issubset(_arm_names(manifest.get("arms")))
        and manifest.get("convergence_curves_ready") is True
        and manifest.get("plateau_consistency_ready") is True
        and result_settled is True
    )


def _t44_observed(manifest: Mapping[str, Any] | None) -> dict[str, object]:
    if manifest is None:
        return {"state": "not_provided"}
    return {
        "artifact_type": manifest.get("artifact_type"),
        "state": manifest.get("state"),
        "arms": sorted(_arm_names(manifest.get("arms"))),
        "convergence_curves_ready": manifest.get("convergence_curves_ready"),
        "budget_ratio_ready": manifest.get("budget_ratio_ready"),
        "plateau_consistency_ready": manifest.get("plateau_consistency_ready"),
        "t4_4_result_settled": manifest.get("t4_4_result_settled"),
        "t4_4_outcome": manifest.get("t4_4_outcome"),
        "positive_result_ready": manifest.get("positive_result_ready"),
        "negative_result_ready": manifest.get("negative_result_ready"),
        "outcome_reasons": manifest.get("outcome_reasons"),
    }


def _t44_result_settled(manifest: Mapping[str, Any] | None) -> bool:
    if manifest is None:
        return False
    if manifest.get("t4_4_result_settled") is True and manifest.get("t4_4_outcome") in {"positive", "negative"}:
        return manifest.get("positive_result_ready") is True or manifest.get("negative_result_ready") is True
    return manifest.get("budget_ratio_ready") is True


def _t45_passes(manifest: Mapping[str, Any] | None, *, settled_trials: int) -> bool:
    expected_trial_count = _t45_expected_trial_count(manifest, settled_trials=settled_trials)
    return (
        manifest is not None
        and manifest.get("artifact_type") == "wmloop-t4-5-verifier-meta-validation-manifest"
        and manifest.get("state") == "ready"
        and int(manifest.get("settled_trial_count") or -1) == expected_trial_count
        and manifest.get("verifier_meta_validation_ready") is True
        and manifest.get("confusion_matrix_ready") is True
        and {"INCONCLUSIVE", "VOID"}.issubset(_case_verdicts(manifest.get("case_examples")))
    )


def _t45_observed(manifest: Mapping[str, Any] | None, *, settled_trials: int) -> dict[str, object]:
    if manifest is None:
        return {"state": "not_provided", "archive_settled_trials": settled_trials}
    return {
        "artifact_type": manifest.get("artifact_type"),
        "state": manifest.get("state"),
        "settled_trial_count": manifest.get("settled_trial_count"),
        "archive_settled_trials": settled_trials,
        "archive_verifier_eligible_settled_trials": _t45_expected_trial_count(manifest, settled_trials=settled_trials),
        "excluded_archive_trial_count": manifest.get("excluded_archive_trial_count"),
        "verifier_meta_validation_ready": manifest.get("verifier_meta_validation_ready"),
        "confusion_matrix_ready": manifest.get("confusion_matrix_ready"),
        "case_verdicts": sorted(_case_verdicts(manifest.get("case_examples"))),
    }


def _t45_expected_trial_count(manifest: Mapping[str, Any] | None, *, settled_trials: int) -> int:
    if manifest is not None:
        value = manifest.get("archive_verifier_eligible_settled_trial_count")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return settled_trials


def _arm_names(value: object) -> set[str]:
    names: set[str] = set()
    if not isinstance(value, list):
        return names
    for item in value:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, Mapping):
            for key in ("name", "arm", "arm_id"):
                raw = item.get(key)
                if isinstance(raw, str):
                    names.add(raw)
    return names


def _case_verdicts(value: object) -> set[str]:
    verdicts: set[str] = set()
    if not isinstance(value, list):
        return verdicts
    for item in value:
        if isinstance(item, str):
            verdicts.add(item)
        elif isinstance(item, Mapping):
            raw = item.get("verdict") or item.get("case_type")
            if isinstance(raw, str):
                verdicts.add(raw)
    return verdicts


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise M4CompletionGateError(f"M4_COMPLETION_GATE_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise M4CompletionGateError(f"M4_COMPLETION_GATE_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "cas_ref": ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    try:
        payload = sources[name].get("payload")
    except KeyError as exc:
        raise M4CompletionGateError(f"M4_COMPLETION_GATE_SOURCE_MISSING:{name}") from exc
    if not isinstance(payload, Mapping):
        raise M4CompletionGateError(f"M4_COMPLETION_GATE_SOURCE_INVALID:{name}")
    return payload


def _payload_or_none(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _payload(sources, name)


def _blockers(requirements: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {"requirement": name, "expected": item["expected"], "observed": item["observed"]}
        for name, item in requirements.items()
        if item.get("passed") is not True
    ]


def _next_actions(requirements: Mapping[str, Mapping[str, object]]) -> list[str]:
    actions = []
    if requirements["strict_phase_gate_ready"]["passed"] is not True:
        actions.append("Do not report M4 as complete until the strict M4 launch gate is ready under the active protocol.")
    if requirements["m4_settled_trial_volume"]["passed"] is not True:
        actions.append("Continue settled closed-loop trials until the archive reaches the M4 target count.")
    if requirements["T4_4_prior_vs_cold_start_convergence"]["passed"] is not True:
        actions.append("Run or settle T4.4 prior/cold-start/shuffled-prior evidence and archive the ready manifest.")
    if requirements["T4_5_verifier_meta_validation"]["passed"] is not True:
        actions.append("Run T4.5 verifier meta-validation over all settled trials and archive the ready manifest.")
    return actions or ["M4 completion gate is ready; export M4 evidence for consolidation."]


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "m4-completion-gate.json", report_bytes)
        _write_bytes_atomic(temporary / "m4-completion-gate.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-completion-gate-manifest",
            "state": report["state"],
            "m4_completion_allowed": report["m4_completion_allowed"],
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "settled_trial_target": report["settled_trial_target"],
            "blockers": report["blockers"],
            "report_path": str(destination / "m4-completion-gate.json"),
            "markdown_path": str(destination / "m4-completion-gate.md"),
            "cas_refs": {"m4_completion_gate_json": report_ref, "m4_completion_gate_markdown": markdown_ref},
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M4 Completion Gate",
        "",
        f"State: `{report['state']}`",
        f"M4 completion allowed: `{report['m4_completion_allowed']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "| Requirement | Passed | Observed |",
        "|:--|:--|:--|",
    ]
    for name, requirement in report["requirements"].items():
        observed = json.dumps(requirement["observed"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        lines.append(f"| {name} | {requirement['passed']} | `{observed}` |")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M4CompletionGateError("M4_COMPLETION_GATE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit whether M4 completion criteria are met")
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--prior-convergence-manifest", type=Path)
    run.add_argument("--verifier-meta-validation-manifest", type=Path)
    run.add_argument("--settled-trial-target", type=int, default=150)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_m4_completion_gate(
            archive_db=args.archive_db,
            phase_gate_manifest=args.phase_gate_manifest,
            prior_convergence_manifest=args.prior_convergence_manifest,
            verifier_meta_validation_manifest=args.verifier_meta_validation_manifest,
            settled_trial_target=args.settled_trial_target,
            output_root=args.output_root,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise M4CompletionGateError("M4_COMPLETION_GATE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
