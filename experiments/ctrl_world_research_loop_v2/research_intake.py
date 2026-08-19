"""Strict source-to-mechanism research intake for Ctrl-World ACWM.

The v1 intake used broad keyword tracks.  This version admits a materialization
work order only when every declared mechanism component is anchored to source
text and every required target capability is present.  Diagnostic methods,
research substrates, incompatible methods, and capability gaps remain useful
knowledge records but never receive GPU authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import canonical_json_bytes, sha256_bytes, sha256_file
from wmloop.control.acwm_dual_evaluation import validate_acwm_dual_evaluation_contract


class ACWMResearchIntakeV2Error(RuntimeError):
    """A source, transfer assessment, or durable intake boundary failed closed."""


_SAFE_ID = re.compile(r"[^a-z0-9_-]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MATERIALIZABLE_ROLE = "transferable_optimizer"


def run_research_intake_v2(
    *,
    config_path: Path,
    contract_path: Path,
    output_root: Path,
    project_root: Path,
    failure_context: Sequence[str] = (),
    source_bundle_path: Path | None = None,
    fetch_bytes: Any | None = None,
) -> dict[str, object]:
    """Retrieve or replay sources and emit only source-grounded work orders."""

    root = Path(project_root).expanduser().resolve()
    config_file = _require_file(config_path, "ACWM_RESEARCH_V2_CONFIG_INVALID")
    contract_file = _require_file(contract_path, "ACWM_RESEARCH_V2_CONTRACT_INVALID")
    config = _load_mapping(config_file, "ACWM_RESEARCH_V2_CONFIG_INVALID")
    contract = _load_mapping(contract_file, "ACWM_RESEARCH_V2_CONTRACT_INVALID")
    _validate_config(config, contract, root=root)
    legacy = _legacy_intake_module()
    bundle_file = (
        _require_file(source_bundle_path, "ACWM_RESEARCH_V2_SOURCE_BUNDLE_INVALID")
        if source_bundle_path is not None
        else None
    )
    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_OUTPUT_INSIDE_SOURCE")

    normalized_context = tuple(
        sorted({value.strip() for value in failure_context if value.strip()})
    )
    input_lock = {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-research-intake-lock",
        "policy_id": config["policy_id"],
        "policy_digest": config["policy_digest"],
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "contract_path": str(contract_file),
        "contract_sha256": sha256_file(contract_file),
        "intake_implementation_sha256": sha256_file(Path(__file__)),
        "retrieval_dependency_sha256": sha256_file(Path(legacy.__file__)),
        "retrieval_mode": "replay" if bundle_file is not None else "network",
        "source_bundle_path": str(bundle_file) if bundle_file is not None else None,
        "source_bundle_sha256": sha256_file(bundle_file) if bundle_file is not None else None,
        "failure_context": list(normalized_context),
    }
    input_lock["input_sha256"] = sha256_bytes(canonical_json_bytes(input_lock))
    resumed = _resume_if_bound(destination, input_lock)
    if resumed is not None:
        return resumed

    if bundle_file is not None:
        raw_sources, inherited_rejected, retrieval = _replay_sources(
            bundle_file, legacy=legacy
        )
    else:
        fetch = fetch_bytes or legacy._fetch_url
        collected, retrieval = legacy._collect_sources(config, fetch=fetch)
        raw_sources, inherited_rejected = legacy._safe_sources(collected)
    safe_sources, newly_rejected = legacy._safe_sources(raw_sources)
    rejected = [*inherited_rejected, *newly_rejected]
    assessments = [
        _assess_source(config, source, failure_context=normalized_context)
        for source in safe_sources
    ]
    ideas = [
        _idea_from_assessment(config, assessment)
        for assessment in assessments
        if assessment["execution_state"] == "materialization_required"
    ]
    maximum = int(_mapping(config, "idea_policy")["max_ideas"])
    ideas = ideas[:maximum]
    work_orders = [_work_order(config, contract, idea) for idea in ideas]
    for assessment in assessments:
        _validate("acwm_source_transfer_assessment_v2", assessment, root=root)
    for idea in ideas:
        _validate("acwm_research_idea_v2", idea, root=root)
    for work_order in work_orders:
        _validate("acwm_research_work_order_v2", work_order, root=root)

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if destination.exists() or destination.is_symlink() or temporary.exists():
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_OUTPUT_EXISTS")
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json(temporary / "input-lock.json", input_lock)
        _write_json(
            temporary / "sources.json",
            {
                "artifact_type": "verdiwm-acwm-research-sources",
                "accepted_sources": [source.to_document() for source in safe_sources],
                "rejected_sources": rejected,
                "retrieval": retrieval,
            },
        )
        for source in safe_sources:
            _write_json(
                temporary / "sources" / f"{source.source_id}.json",
                source.to_document(),
            )
        for assessment in assessments:
            _write_json(
                temporary / "assessments" / f"{assessment['source_id']}.json",
                assessment,
            )
        for idea, work_order in zip(ideas, work_orders):
            _write_json(temporary / "ideas" / f"{idea['idea_id']}.json", idea)
            _write_json(
                temporary / "work-orders" / f"{idea['idea_id']}.json",
                work_order,
            )
        role_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        for assessment in assessments:
            role = str(assessment["source_role"])
            state = str(assessment["execution_state"])
            role_counts[role] = role_counts.get(role, 0) + 1
            state_counts[state] = state_counts.get(state, 0) + 1
        manifest = {
            "schema_version": 2,
            "artifact_type": "verdiwm-acwm-research-intake-manifest",
            "state": "ready_for_materialization" if ideas else "no_admissible_external_idea",
            "policy_id": config["policy_id"],
            "policy_digest": config["policy_digest"],
            "input_sha256": input_lock["input_sha256"],
            "retrieval_mode": input_lock["retrieval_mode"],
            "source_count": len(safe_sources),
            "rejected_source_count": len(rejected),
            "assessment_count": len(assessments),
            "idea_count": len(ideas),
            "role_counts": dict(sorted(role_counts.items())),
            "execution_state_counts": dict(sorted(state_counts.items())),
            "assessment_paths": [
                str(destination / "assessments" / f"{row['source_id']}.json")
                for row in assessments
            ],
            "idea_paths": [
                str(destination / "ideas" / f"{idea['idea_id']}.json")
                for idea in ideas
            ],
            "work_order_paths": [
                str(destination / "work-orders" / f"{idea['idea_id']}.json")
                for idea in ideas
            ],
            "retrieval": retrieval,
            "side_effects": {
                "network_metadata_read": bundle_file is None,
                "source_bundle_replayed": bundle_file is not None,
                "source_code_mutated": False,
                "gpu_execution_started": False,
                "candidate_batch_created": False,
                "model_promoted": False,
            },
            "claim_boundary": (
                "Only source-grounded mechanism transfers with complete target capabilities "
                "receive non-executable materialization work orders. Diagnostics, substrates, "
                "incompatible methods, and capability gaps remain knowledge-only records."
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            _remove_tree(temporary)
        raise


def _validate_config(
    config: Mapping[str, object], contract: Mapping[str, object], *, root: Path
) -> None:
    _validate("acwm_research_intake_v2", config, root=root)
    validate_acwm_dual_evaluation_contract(contract, root=root)
    if config.get("policy_digest") != _digest_without(config, "policy_digest"):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_POLICY_DIGEST_MISMATCH")
    if config.get("contract_id") != contract.get("contract_id"):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_CONTRACT_ID_MISMATCH")
    if config.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_CONTRACT_DIGEST_MISMATCH")
    profiles = config.get("mechanism_profiles")
    assert isinstance(profiles, list)
    priorities = [int(row["priority"]) for row in profiles if isinstance(row, Mapping)]
    profile_ids = [str(row["profile_id"]) for row in profiles if isinstance(row, Mapping)]
    if len(priorities) != len(profiles) or len(priorities) != len(set(priorities)):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_PROFILE_PRIORITY_INVALID")
    if len(profile_ids) != len(profiles) or len(profile_ids) != len(set(profile_ids)):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_PROFILE_ID_INVALID")


def _replay_sources(path: Path, *, legacy: ModuleType) -> tuple[list[Any], list[dict[str, str]], list[dict[str, object]]]:
    payload = _load_mapping(path, "ACWM_RESEARCH_V2_SOURCE_BUNDLE_INVALID")
    rows = payload.get("accepted_sources")
    rejected = payload.get("rejected_sources", [])
    retrieval = payload.get("retrieval", [])
    if not isinstance(rows, list) or not isinstance(rejected, list) or not isinstance(retrieval, list):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_SOURCE_BUNDLE_INVALID")
    sources = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_SOURCE_BUNDLE_INVALID")
        fields = {
            key: str(row.get(key) or "")
            for key in (
                "source_id",
                "source_type",
                "title",
                "summary",
                "source_url",
                "query",
                "source_digest",
            )
        }
        expected = _source_digest(fields)
        if not fields["source_id"] or fields["source_digest"] != expected:
            raise ACWMResearchIntakeV2Error(
                f"ACWM_RESEARCH_V2_SOURCE_DIGEST_MISMATCH:{fields['source_id']}"
            )
        sources.append(legacy.ResearchSource(**fields))
    return sources, [dict(row) for row in rejected if isinstance(row, Mapping)], [dict(row) for row in retrieval if isinstance(row, Mapping)]


def _assess_source(
    config: Mapping[str, object], source: Any, *, failure_context: Sequence[str]
) -> dict[str, object]:
    content = f"{source.title} {source.summary}".lower()
    profiles = config["mechanism_profiles"]
    assert isinstance(profiles, list)
    matches = [
        row
        for row in profiles
        if isinstance(row, Mapping) and _profile_matches(row, content)
    ]
    if not matches:
        return _ungrounded_assessment(source, failure_context=failure_context)
    profile = max(matches, key=lambda row: int(row["priority"]))
    target_capabilities = {
        str(value) for value in config["target_capabilities"] if isinstance(value, str)
    }
    required = [str(value) for value in profile["required_capabilities"]]
    missing = sorted(set(required) - target_capabilities)
    role = str(profile["source_role"])
    if role == _MATERIALIZABLE_ROLE and not missing:
        execution_state = "materialization_required"
    elif role == _MATERIALIZABLE_ROLE:
        execution_state = "blocked_target_capability"
    elif role == "diagnostic_method":
        execution_state = "diagnostic_routing_required"
    elif role == "research_substrate":
        execution_state = "knowledge_only"
    else:
        execution_state = "rejected_source_target_mismatch"
    evidence = [
        _component_evidence(component, source.summary)
        for component in profile["components"]
        if isinstance(component, Mapping)
    ]
    payload = {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-source-transfer-assessment",
        "source_id": source.source_id,
        "source_digest": source.source_digest,
        "source_title": source.title,
        "profile_id": profile["profile_id"],
        "source_role": role,
        "track": profile["track"],
        "transfer_mode": profile["transfer_mode"],
        "execution_state": execution_state,
        "failure_context": list(failure_context),
        "source_evidence": evidence,
        "preserved_components": [row["component_id"] for row in evidence],
        "required_target_capabilities": required,
        "missing_target_capabilities": missing,
        "target_intervention": profile["target_intervention"],
        "forbidden_substitutions": list(profile["forbidden_substitutions"]),
        "falsification_criterion": profile["falsification_criterion"],
        "claim_boundary": (
            "This assessment classifies a source-to-target transfer. It grants no source "
            "mutation, GPU scheduling, or promotion authority."
        ),
    }
    payload["assessment_digest"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _ungrounded_assessment(source: Any, *, failure_context: Sequence[str]) -> dict[str, object]:
    payload = {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-source-transfer-assessment",
        "source_id": source.source_id,
        "source_digest": source.source_digest,
        "source_title": source.title,
        "profile_id": "unclassified",
        "source_role": "incompatible",
        "track": "unclassified",
        "transfer_mode": "none",
        "execution_state": "rejected_source_mechanism_ungrounded",
        "failure_context": list(failure_context),
        "source_evidence": [],
        "preserved_components": [],
        "required_target_capabilities": [],
        "missing_target_capabilities": [],
        "target_intervention": "none",
        "forbidden_substitutions": [
            "Do not invent a generic history, guidance, or action-conditioning method from broad keywords."
        ],
        "falsification_criterion": "No experiment is admissible without a source-grounded mechanism profile.",
        "claim_boundary": (
            "This source remains unclassified and grants no source mutation, GPU scheduling, "
            "or promotion authority."
        ),
    }
    payload["assessment_digest"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _idea_from_assessment(
    config: Mapping[str, object], assessment: Mapping[str, object]
) -> dict[str, object]:
    idea_id = _safe_id(
        f"idea-{assessment['source_id']}-{assessment['profile_id']}-v2"
    )
    payload = {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-research-idea",
        "idea_id": idea_id,
        "source": {
            "source_id": assessment["source_id"],
            "source_digest": assessment["source_digest"],
            "assessment_digest": assessment["assessment_digest"],
        },
        "track": assessment["track"],
        "transfer_mode": assessment["transfer_mode"],
        "failure_context": list(assessment["failure_context"]),
        "source_evidence": list(assessment["source_evidence"]),
        "materialization_contract": {
            "target_model_family": config["target_model_family"],
            "target_intervention": assessment["target_intervention"],
            "preserved_components": list(assessment["preserved_components"]),
            "required_target_capabilities": list(
                assessment["required_target_capabilities"]
            ),
            "forbidden_substitutions": list(assessment["forbidden_substitutions"]),
        },
        "objective": (
            f"Test the source-grounded {assessment['profile_id']} mechanism on "
            f"{config['target_model_family']} under the frozen ACWM contract."
        ),
        "hypothesis": (
            f"Preserving {', '.join(str(value) for value in assessment['preserved_components'])} "
            "improves the primary paired-prediction metric without a protected regression."
        ),
        "falsification_criterion": assessment["falsification_criterion"],
        "required_protocol": ["screen", "confirm", "frozen_verifier"],
        "synthesis_method": "source_component_grounding_v2",
        "execution_state": "materialization_required",
        "claim_boundary": (
            "This is a mechanism-transfer hypothesis, not a claim of paper reproduction. "
            "Every preserved component must map to code without a declared compromise."
        ),
    }
    return payload


def _work_order(
    config: Mapping[str, object],
    contract: Mapping[str, object],
    idea: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-research-materialization-work-order",
        "idea_id": idea["idea_id"],
        "idea_sha256": sha256_bytes(canonical_json_bytes(idea)),
        "policy_id": config["policy_id"],
        "contract": {
            "contract_id": contract["contract_id"],
            "contract_digest": contract["contract_digest"],
        },
        "source_grounding": idea["source"],
        "materialization_contract": idea["materialization_contract"],
        "required_before_gpu": [
            "isolated implementation receipt with source revision",
            "complete source-component-to-code mapping",
            "zero declared compromises",
            "immutable candidate batch bound to the implementation receipt",
            "independent screen and confirm baselines",
        ],
        "required_execution_stages": ["screen", "confirm", "frozen_verifier"],
        "forbidden_actions": [
            "execute external source code",
            "replace a preserved component with a proxy",
            "mutate the frozen evaluator or data split",
            "reuse confirm measurements as a screen baseline",
            "promote a model from screen or confirm evidence",
        ],
        "execution_authority": "none_until_materialized_and_compiled",
        "claim_boundary": (
            "This work order grants no GPU or promotion authority. A materializer must "
            "preserve every declared source component or abstain."
        ),
    }


def _profile_matches(profile: Mapping[str, object], content: str) -> bool:
    groups = profile.get("required_term_groups")
    if not isinstance(groups, list) or not groups:
        return False
    for group in groups:
        if not isinstance(group, list) or not any(str(term).lower() in content for term in group):
            return False
    excluded = profile.get("excluded_terms", [])
    return not any(str(term).lower() in content for term in excluded if isinstance(term, str))


def _component_evidence(component: Mapping[str, object], summary: str) -> dict[str, object]:
    terms = component.get("evidence_terms")
    if not isinstance(terms, list):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_COMPONENT_INVALID")
    matched = [str(term) for term in terms if str(term).lower() in summary.lower()]
    if not matched:
        raise ACWMResearchIntakeV2Error(
            f"ACWM_RESEARCH_V2_COMPONENT_UNGROUNDED:{component.get('component_id')}"
        )
    return {
        "component_id": component["component_id"],
        "description": component["description"],
        "matched_terms": sorted(matched),
        "evidence_snippet": _evidence_sentence(summary, matched[0]),
    }


def _evidence_sentence(summary: str, term: str) -> str:
    sentences = _SENTENCE_BOUNDARY.split(" ".join(summary.split()))
    lowered = term.lower()
    for sentence in sentences:
        if lowered in sentence.lower():
            return sentence[:1000]
    raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_EVIDENCE_SNIPPET_MISSING")


def _source_digest(fields: Mapping[str, str]) -> str:
    payload = {
        key: fields[key]
        for key in (
            "source_id",
            "source_type",
            "title",
            "summary",
            "source_url",
            "query",
        )
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _legacy_intake_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "ctrl_world_acwm_guidance_v1"
        / "research_intake.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verdiwm_ctrl_world_acwm_research_intake_v1_dependency", path
    )
    if spec is None or spec.loader is None:
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_DEPENDENCY_INVALID")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate(schema_name: str, payload: Mapping[str, object], *, root: Path) -> None:
    try:
        validate_document(schema_name, payload, root=root)
    except ContractValidationError as exc:
        raise ACWMResearchIntakeV2Error(
            f"ACWM_RESEARCH_V2_SCHEMA_INVALID:{schema_name}:{exc}"
        ) from exc


def _resume_if_bound(
    destination: Path, input_lock: Mapping[str, object]
) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_OUTPUT_INVALID")
    lock = _load_mapping(
        destination / "input-lock.json", "ACWM_RESEARCH_V2_INPUT_LOCK_INVALID"
    )
    if lock.get("input_sha256") != input_lock.get("input_sha256"):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_INPUT_LOCK_MISMATCH")
    return _load_mapping(
        destination / "manifest.json", "ACWM_RESEARCH_V2_MANIFEST_INVALID"
    )


def _load_mapping(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMResearchIntakeV2Error(code) from exc
    if not isinstance(payload, dict):
        raise ACWMResearchIntakeV2Error(code)
    return payload


def _require_file(path: Path | None, code: str) -> Path:
    if path is None:
        raise ACWMResearchIntakeV2Error(code)
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ACWMResearchIntakeV2Error(code)
    value = value.resolve()
    if not value.is_file():
        raise ACWMResearchIntakeV2Error(code)
    return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ACWMResearchIntakeV2Error("ACWM_RESEARCH_V2_CONFIG_INVALID")
    return child


def _safe_id(value: str) -> str:
    result = _SAFE_ID.sub("-", value.lower()).strip("-")[:110]
    return result or "idea"


def _digest_without(document: Mapping[str, object], field: str) -> str:
    return sha256_bytes(
        canonical_json_bytes({key: value for key, value in document.items() if key != field})
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--failure-context", action="append", default=[])
    args = parser.parse_args(argv)
    result = run_research_intake_v2(
        config_path=args.config,
        contract_path=args.contract,
        output_root=args.output_root,
        project_root=args.project_root,
        failure_context=args.failure_context,
        source_bundle_path=args.source_bundle,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
