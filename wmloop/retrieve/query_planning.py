"""Evidence-bound, bounded literature search plans for each diagnostic round."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.retrieve.query_terms import arxiv_clause, query_terms
from wmloop.storage import atomic_write, canonical_bytes, canonical_json_bytes, checked_path, exclusive_file_lock, sha256_bytes


class LiteratureQueryPlanError(ValueError):
    """A literature query input or immutable plan is invalid."""


# Searchable domain words preserve the old automatic cold-start route without
# treating arbitrary adapter identifiers (such as arm_slot) as scientific tasks.
_DOMAIN_WORDS = frozenset({"video", "image", "audio", "speech", "language", "vision", "2d", "3d"})
_TASK_DOMAIN_WORDS = _DOMAIN_WORDS | frozenset({
    "robot", "robotics", "manipulation", "prediction", "forecasting", "generation",
    "diffusion", "reconstruction", "denoising", "restoration", "segmentation",
    "detection", "tracking", "control", "planning", "grasping", "navigation",
})
_METRIC_WORDS = frozenset({"psnr", "ssim", "lpips", "mse", "mae", "rmse", "iou", "auc", "bleu", "rouge"})
_DATASET_WORDS = frozenset({"agibot", "droid"})


def build_literature_query_plan(
    *,
    explicit_query: str | None,
    retrieval_context: Mapping[str, object],
    probe_manifest: Mapping[str, object] | None,
    irg_failure_signatures: tuple[str, ...] = (),
    irg_model_family: str | None = None,
) -> dict[str, object]:
    """Prioritize the frozen research goal, retaining diagnostic evidence.

    The capsule is the index's bounded projection of verified settled records.
    Paired outcomes can suggest a mechanism lens, but neither a diagnostic
    outcome nor PASS is an effect verdict. Old signatures never override a
    current observation. The first view searches the stated goal (optionally
    informed by verified memory); a current-diagnostic view is the fallback
    after an empty successful response.
    """

    if explicit_query is not None and not isinstance(explicit_query, str):
        raise LiteratureQueryPlanError("LITERATURE_INTENT_INVALID")
    intent = (explicit_query or "").strip()
    probe = dict(probe_manifest or {})
    irg = retrieval_context.get("irg_guided")
    request = irg.get("request", {}) if isinstance(irg, Mapping) else {}
    request = request if isinstance(request, Mapping) else {}
    family = _text(probe.get("model_family")) or _text(request.get("model_family")) or irg_model_family
    current_signatures = _strings(probe.get("failure_signatures"))
    fallback_signatures = _strings(request.get("failure_signatures")) or list(irg_failure_signatures)
    signatures = current_signatures or fallback_signatures
    symptom_groups = [_symptom_terms(signature) for signature in signatures[:3]]
    symptom_groups = [group for group in symptom_groups if group]
    focus_terms = tuple(dict.fromkeys(term for group in symptom_groups for term in group))
    structured_intent = bool(re.search(r"\b(?:all|ti|au|abs|cat|id):", intent))
    intent_words = () if structured_intent else query_terms(intent, maximum=20, intent=True)
    # Goal vocabulary must not depend on whichever symptom happened to be
    # diagnosed this round. Diagnostics have their own fallback view below.
    task_words = tuple(term for term in intent_words if term not in {"quality", "performance", "accuracy"})
    # An adapter family can be an internal identifier (e.g. arm_slot). The
    # user's stated task is a stronger domain anchor than those identifiers.
    family_words = query_terms(family or "", maximum=4)
    explicit_domain_terms = tuple(term for term in task_words if term in _TASK_DOMAIN_WORDS)[:3]
    fallback_task_terms = tuple(term for term in task_words if term not in _METRIC_WORDS and term not in _DATASET_WORDS)
    family_domain_terms = (
        tuple(term for term in family_words if term in _DOMAIN_WORDS or term == "diffusion")[:3]
        if _DOMAIN_WORDS.intersection(family_words) else ()
    )
    domain_terms = explicit_domain_terms or fallback_task_terms[:2] or family_domain_terms
    intent_terms = tuple(
        term for term in intent_words
        if term not in set(domain_terms) and term not in _DATASET_WORDS
    )[:5]
    irg_lenses = _strings(request.get("cross_domain_lenses"))[:2]
    memory = _memory_context(retrieval_context.get("capsule"), current_signatures)
    agibot_psnr_memory = _agibot_psnr_memory_context(
        retrieval_context.get("agibot_psnr_memory")
    )
    memory_terms = tuple(dict.fromkeys(term for row in memory if row["used_as_lens"]
                                      for term in query_terms(" ".join(row["lens_terms"]), maximum=3)))[:3]
    psnr_lens_terms = tuple(
        dict.fromkeys(
            term
            for row in agibot_psnr_memory
            if row["used_as_lens"]
            for term in query_terms(" ".join(row["lens_terms"]), maximum=3)
        )
    )[:3]
    lens_terms = memory_terms or psnr_lens_terms or query_terms(" ".join(irg_lenses), maximum=3)

    goal_clauses = [arxiv_clause(domain_terms, operator="AND")]
    if structured_intent:
        if len(intent) > 512:
            raise LiteratureQueryPlanError("LITERATURE_STRUCTURED_INTENT_TOO_LONG")
        goal_clauses.append(f"({intent})")
    # With no English canonical evidence, do not pretend to translate a goal.
    goal_base = " AND ".join(clause for clause in goal_clauses if clause)
    diagnostic_clauses = list(goal_clauses)
    if symptom_groups:
        diagnostic_clauses.append("(" + " OR ".join(arxiv_clause(group, operator="AND") for group in symptom_groups) + ")")
    diagnostic_query = " AND ".join(clause for clause in diagnostic_clauses if clause)
    queries: list[dict[str, str]] = []
    if goal_base and (domain_terms or structured_intent):
        refinements = [arxiv_clause(intent_terms), arxiv_clause(lens_terms)]
        refinement = " AND ".join(clause for clause in refinements if clause)
        goal_query = goal_base + (" AND " + refinement if refinement else "")
        if refinement:
            queries.append({"view": "evidence_lens" if lens_terms else "goal_focus", "query": goal_query})
        else:
            queries.append({"view": "goal_focus", "query": goal_query})
        if symptom_groups:
            queries.append({"view": "current_diagnostic", "query": diagnostic_query})
    if any(len(row["query"]) > 1024 for row in queries):
        raise LiteratureQueryPlanError("LITERATURE_QUERY_PLAN_TOO_LONG")
    diagnostic = {
        "model_family": family,
        "failure_signatures": current_signatures[:20],
        "manifest_sha256": sha256_bytes(canonical_bytes(probe)) if probe_manifest is not None else None,
        **{key: _text(probe.get(key)) for key in ("probe_id", "archive_trial_id", "receipt_ref", "result_ref")},
    }
    plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-literature-query-plan",
        "state": "planned" if queries else "skipped",
        "reason": None if queries else "LITERATURE_TASK_TERMS_UNAVAILABLE",
        "suggested_action": None if queries else "Add searchable English task words to the research goal, for example: video prediction.",
        "authority": "ranking_only",
        "explicit_intent": {"text": intent[:4096], "sha256": sha256_bytes(intent.encode("utf-8")), "truncated": len(intent) > 4096},
        "current_diagnostic": diagnostic,
        "signature_source": "current_diagnostic" if current_signatures else "irg" if signatures else "none",
        "domain_terms": list(domain_terms),
        "domain_source": "explicit_intent" if explicit_domain_terms or fallback_task_terms else "model_family" if domain_terms else "none",
        "focus_terms": list(focus_terms),
        "intent_terms": list(intent_terms),
        "irg_lenses": [value[:256] for value in irg_lenses],
        "memory_context": memory,
        "agibot_psnr_memory": agibot_psnr_memory,
        "query": queries[0]["query"] if queries else None,
        "queries": queries,
        "fallback_policy": "next_query_only_after_empty_network_response",
        "claim_boundary": "Search terms route untrusted literature candidates only. Current diagnostics take precedence; verified memory and paired signs supply hypotheses, never target effects, success or transfer authority. Untranslated intent remains provenance.",
    }
    validate_document("literature_query_plan", plan)
    return plan


def write_literature_query_plan(path: Path, plan: Mapping[str, object]) -> Path:
    """Persist one immutable query plan; identical retries keep its bytes."""

    validate_document("literature_query_plan", plan)
    destination = checked_path(path, code="LITERATURE_QUERY_PLAN_OUTPUT_INVALID", error=LiteratureQueryPlanError)
    payload = canonical_json_bytes(plan)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(destination.parent / f".{destination.name}.lock"):
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != payload:
                raise LiteratureQueryPlanError("LITERATURE_QUERY_PLAN_INPUT_MISMATCH")
            return destination
        atomic_write(destination, payload)
    return destination


def _memory_context(capsule: object, current_signatures: Sequence[str]) -> list[dict[str, object]]:
    if not isinstance(capsule, Mapping):
        return []
    selected = capsule.get("selected_evidence", [])
    if not isinstance(selected, list):
        raise LiteratureQueryPlanError("LITERATURE_MEMORY_INVALID")
    rows = []
    for row in selected[:3]:
        if not isinstance(row, Mapping) or not isinstance(row.get("evidence"), Mapping):
            raise LiteratureQueryPlanError("LITERATURE_MEMORY_INVALID")
        evidence = row["evidence"]
        paired = row.get("paired_observations", [])
        if not isinstance(paired, list):
            raise LiteratureQueryPlanError("LITERATURE_MEMORY_INVALID")
        observations = []
        lens_terms = []
        for pair in paired[:3]:
            if not isinstance(pair, Mapping):
                raise LiteratureQueryPlanError("LITERATURE_MEMORY_PAIRED_INVALID")
            query = pair.get("query")
            try:
                validate_document("diagnostic_paired_query", query)
            except (ContractValidationError, TypeError) as exc:
                raise LiteratureQueryPlanError("LITERATURE_MEMORY_PAIRED_INVALID") from exc
            outcome = pair.get("outcome")
            if outcome not in {"positive", "negative", "null"}:
                raise LiteratureQueryPlanError("LITERATURE_MEMORY_PAIRED_INVALID")
            observations.append({"primitive": query["primitive"], "metric_name": query["metric_name"], "outcome": outcome})
            lens_terms.extend((query["primitive"], query["metric_name"]))
        compatible = row.get("failure_signature") in current_signatures
        rows.append({
            "failure_signature": str(row.get("failure_signature") or ""),
            "primitive": _text(row.get("primitive")),
            "verdict": str(row.get("verdict") or ""),
            "evidence": {key: str(evidence.get(key) or "") for key in ("archive_trial_id", "receipt_ref", "result_ref", "receipt_hash")},
            "paired_outcomes": observations,
            "used_as_lens": bool(observations and compatible),
            "lens_terms": list(dict.fromkeys(lens_terms))[:6] if compatible else [],
        })
    return rows


def _agibot_psnr_memory_context(value: object) -> list[dict[str, object]]:
    """Project settled PSNR memory into a bounded, ranking-only query lens."""

    if not isinstance(value, Mapping):
        return []
    matches = value.get("matches", [])
    if not isinstance(matches, list):
        raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
    result: list[dict[str, object]] = []
    for row in matches[:5]:
        if not isinstance(row, Mapping):
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        evidence = row.get("evidence")
        statistics = row.get("statistics")
        if not isinstance(evidence, Mapping) or not isinstance(statistics, Mapping):
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        if not isinstance(row.get("trial_id"), str) or not row["trial_id"]:
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        if any(
            not isinstance(evidence.get(key), str) or not evidence[key]
            for key in ("receipt_ref", "verdict_ref", "measurement_ref")
        ):
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        hypothesis = row.get("hypothesis")
        method = row.get("method")
        if not isinstance(hypothesis, str) or not isinstance(method, str):
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        verdict = row.get("verdict")
        if verdict not in {"ACCEPT", "REJECT"}:
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        mean = statistics.get("mean_paired_delta_db")
        lower = statistics.get("lower_bound_db")
        upper = statistics.get("upper_bound_db")
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            for number in (mean, lower, upper)
        ):
            raise LiteratureQueryPlanError("LITERATURE_AGIBOT_PSNR_MEMORY_INVALID")
        result.append(
            {
                "trial_id": str(row["trial_id"]),
                "verdict": verdict,
                "effect_verified": row.get("effect_verified") is True,
                "statistics": {
                    "mean_paired_delta_db": float(mean),
                    "lower_bound_db": float(lower),
                    "upper_bound_db": float(upper),
                },
                "evidence": {
                    "receipt_ref": str(evidence["receipt_ref"]),
                    "verdict_ref": str(evidence["verdict_ref"]),
                    "measurement_ref": str(evidence["measurement_ref"]),
                },
                "hypothesis": hypothesis,
                "method": method,
                "used_as_lens": True,
                "lens_terms": list(
                    dict.fromkeys(query_terms(f"{hypothesis} {method}", maximum=6))
                ),
                "claim_boundary": (
                    "Settled PSNR memory guides ranking only; it does not transfer "
                    "an effect or establish the current round's result."
                ),
            }
        )
    return result


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _symptom_terms(signature: str) -> tuple[str, ...]:
    parts = signature.lower().split("_")
    if parts and parts[0] in {"paired", "diagnostic", "probe"}:
        parts = parts[1:]
    if parts and parts[-1] in {"high", "low", "poor", "detected", "observed"}:
        parts = parts[:-1]
    return query_terms(" ".join(parts), maximum=3)


def _strings(value: object) -> list[str]:
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item)) if isinstance(value, (list, tuple)) else []
