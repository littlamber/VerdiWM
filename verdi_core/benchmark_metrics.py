"""Benchmark-aware metric discovery, selection, and evidence gates.

The Kernel does not assume that every benchmark metric fits every model.  A
catalog entry describes its data/evaluator requirements; an AI provider may
choose among eligible entries, while deterministic validation remains the
authority for promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .runtime import AIProvider


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    benchmark: str
    description: str
    direction: str = "maximize"
    role_candidates: tuple[str, ...] = ("primary", "protected", "diagnostic")
    required_signals: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    architecture_facets: tuple[str, ...] = ()
    cost: str = "medium"
    ground_truth: bool = True
    evaluator_ref: str = ""
    source_refs: tuple[str, ...] = ()
    diagnostic_only: bool = False
    implementation_status: str = "catalogued"

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be maximize or minimize")
        if not self.metric_id or not self.benchmark:
            raise ValueError("metric_id and benchmark are required")
        if self.cost not in {"low", "medium", "high"}:
            raise ValueError("metric cost must be low, medium, or high")

    def eligibility(self, model_report: Mapping[str, Any], available_signals: Iterable[str] = ()) -> tuple[bool, list[str]]:
        signals = {str(value).lower() for value in available_signals}
        for key in ("signals", "available_signals", "metrics", "data_signals"):
            value = model_report.get(key, ())
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                signals.update(str(item).lower() for item in value)
        capabilities = {str(value).lower() for value in model_report.get("capabilities", ())}
        architecture = {str(value).lower() for value in model_report.get("architecture_facets", model_report.get("architecture", ())) or ()}
        reasons: list[str] = []
        missing_signals = sorted(set(self.required_signals) - signals)
        if missing_signals:
            reasons.append("missing_signals:" + ",".join(missing_signals))
        missing_capabilities = sorted(set(self.required_capabilities) - capabilities)
        if missing_capabilities:
            reasons.append("missing_capabilities:" + ",".join(missing_capabilities))
        if self.architecture_facets and architecture and not (architecture & set(self.architecture_facets)):
            reasons.append("architecture_mismatch:" + ",".join(self.architecture_facets))
        if self.implementation_status not in {"implemented", "validated", "catalogued"}:
            reasons.append("implementation_" + self.implementation_status)
        return not reasons, reasons


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


class MetricCatalog:
    """Versioned metric catalog assembled from benchmark sources."""

    def __init__(self, definitions: Iterable[MetricDefinition] = (), *, catalog_id: str = "catalog"):
        self.catalog_id = catalog_id
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions:
            self.add(definition)

    def add(self, definition: MetricDefinition) -> None:
        if definition.metric_id in self._definitions:
            raise ValueError(f"metric already registered: {definition.metric_id}")
        self._definitions[definition.metric_id] = definition

    def get(self, metric_id: str) -> MetricDefinition | None:
        return self._definitions.get(str(metric_id))

    def all(self) -> tuple[MetricDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def digest(self) -> str:
        return _digest([asdict(item) for item in self.all()])

    def candidates(self, model_report: Mapping[str, Any], available_signals: Iterable[str] = ()) -> tuple[list[MetricDefinition], dict[str, list[str]]]:
        eligible: list[MetricDefinition] = []
        rejected: dict[str, list[str]] = {}
        for definition in self.all():
            ok, reasons = definition.eligibility(model_report, available_signals)
            if ok:
                eligible.append(definition)
            else:
                rejected[definition.metric_id] = reasons
        return eligible, rejected

    def as_prompt_payload(self, model_report: Mapping[str, Any], available_signals: Iterable[str] = ()) -> dict[str, Any]:
        eligible, rejected = self.candidates(model_report, available_signals)
        return {
            "catalog_id": self.catalog_id,
            "catalog_digest": self.digest(),
            "eligible": [asdict(item) for item in eligible],
            "rejected": rejected,
        }

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], *, catalog_id: str = "imported") -> "MetricCatalog":
        catalog = cls(catalog_id=catalog_id)
        for record in records:
            if not isinstance(record, Mapping) or not record.get("metric_id"):
                continue
            values = dict(record)
            for key in ("role_candidates", "required_signals", "required_capabilities", "architecture_facets", "source_refs"):
                if isinstance(values.get(key), list):
                    values[key] = tuple(str(item) for item in values[key])
            catalog.add(MetricDefinition(**values))
        return catalog


class WorldArenaMetricCatalog(MetricCatalog):
    """Seed catalog for the action-conditioned predictive WorldArena family.

    The source refs are provenance labels, not a claim that all entries are
    available for every release.  Adapters may extend or replace this catalog
    with records parsed from the benchmark's pinned paper/code release.
    """

    @classmethod
    def default(cls) -> "WorldArenaMetricCatalog":
        source = "worldarena:frozen-metric-policy"
        return cls([
            MetricDefinition("rollout_video_l1", "worldarena", "paired RGB rollout error", "minimize", ("primary", "protected", "diagnostic"), ("video_ground_truth",), ("rollout",), (), "medium", True, "worldarena:evaluator:v1", (source,), False, "validated"),
            MetricDefinition("rollout_video_psnr", "worldarena", "paired rollout reconstruction PSNR", "maximize", ("protected", "diagnostic"), ("video_ground_truth",), ("rollout",), (), "low", True, "worldarena:evaluator:v1", (source,), False, "validated"),
            MetricDefinition("segment_final_mae", "worldarena", "final state error after an action segment", "minimize", ("primary", "protected", "diagnostic"), ("state_ground_truth",), ("rollout",), (), "medium", True, "worldarena:evaluator:v1", (source,), False, "validated"),
            MetricDefinition("segment_view_pair_mae", "worldarena", "paired multi-view segment error", "minimize", ("protected", "diagnostic"), ("state_ground_truth", "multiview"), ("rollout",), (), "medium", True, "worldarena:evaluator:v1", (source,), False, "validated"),
            MetricDefinition("segment_view_fused_mae", "worldarena", "fused multi-view segment error", "minimize", ("protected", "diagnostic"), ("state_ground_truth", "multiview"), ("rollout",), (), "medium", True, "worldarena:evaluator:v1", (source,), False, "validated"),
            MetricDefinition("horizon_drift_slope", "worldarena", "interaction-indexed long-horizon drift slope", "minimize", ("primary", "protected", "diagnostic"), ("interaction_index", "state_ground_truth"), ("rollout",), (), "high", True, "worldarena:evaluator:v2", (source,)),
            MetricDefinition("action_conditioning_sensitivity", "worldarena", "response change under controlled action dose", "maximize", ("diagnostic",), ("action_counterfactual",), ("rollout",), (), "low", False, "worldarena:probe:v1", (source,), True),
            MetricDefinition("temporal_mean_preservation", "worldarena", "temporal mean preservation under intervention", "minimize", ("diagnostic",), ("video_ground_truth",), ("rollout",), (), "low", True, "worldarena:evaluator:v1", (source,), True),
        ], catalog_id="worldarena-action-conditioned-v1")


class MetricCatalogDiscovery:
    """Turn pinned benchmark documentation/code into auditable catalog rows."""

    def __init__(self, ai: AIProvider | None = None):
        self.ai = ai

    def discover(self, documents: Sequence[Mapping[str, Any]], *, benchmark: str = "worldarena") -> dict[str, Any]:
        source_digests = [_digest({"url": item.get("url"), "title": item.get("title"), "digest": item.get("content_digest")}) for item in documents]
        if self.ai is None:
            return {"state": "abstain", "reason": "AI provider not configured", "source_digests": source_digests}
        compact = [
            {"url": item.get("url"), "title": item.get("title"), "content_digest": item.get("content_digest"), "text": str(item.get("text", ""))[:12000]}
            for item in documents[:20]
        ]
        prompt = json.dumps({
            "benchmark": benchmark,
            "documents": compact,
            "instruction": "Extract only explicit benchmark metric definitions. Return JSON object {metrics:[...]}; each metric must include metric_id, benchmark, description, direction, role_candidates, required_signals, required_capabilities, cost, ground_truth, evaluator_ref, source_refs, diagnostic_only, implementation_status. Do not invent formulas or availability; mark uncertain entries implementation_status=unverified.",
        }, sort_keys=True)
        try:
            result = json.loads(self.ai.complete(role="benchmark_catalog_extractor", prompt=prompt))
            records = result.get("metrics", []) if isinstance(result, Mapping) else []
            catalog = MetricCatalog.from_records(records, catalog_id=benchmark + "-discovered")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return {"state": "abstain", "reason": "invalid catalog extraction", "source_digests": source_digests}
        return {"state": "discovered", "catalog_id": catalog.catalog_id, "catalog_digest": catalog.digest(), "source_digests": source_digests, "metrics": [asdict(item) for item in catalog.all()]}


def validate_metric_materialization(
    definition: MetricDefinition,
    receipt: Mapping[str, Any],
    *,
    require_reference_alignment: bool = True,
) -> dict[str, Any]:
    """Gate an AI-authored or adapter-provided evaluator before formal use.

    The receipt is produced inside an isolated worktree by the engineering
    agent or an adapter installer.  Passing unit tests alone is insufficient:
    a ground-truth metric must also identify a frozen evaluator revision and,
    when available, agree with benchmark reference fixtures.
    """
    errors: list[str] = []
    if not definition.ground_truth:
        errors.append("diagnostic_or_subjective_metric_cannot_be_formal")
    if not bool(receipt.get("contract_tests_passed")):
        errors.append("contract_tests_not_passed")
    if not str(receipt.get("evaluator_digest", "")).startswith("sha256:"):
        errors.append("missing_evaluator_digest")
    if not str(receipt.get("frozen_split_digest", "")).startswith("sha256:"):
        errors.append("missing_frozen_split_digest")
    if require_reference_alignment and not bool(receipt.get("reference_alignment_passed")):
        errors.append("reference_alignment_not_passed")
    if not bool(receipt.get("deterministic_repeat_passed")):
        errors.append("deterministic_repeat_not_passed")
    return {
        "state": "validated" if not errors else "abstain",
        "metric_id": definition.metric_id,
        "errors": errors,
        "evaluator_digest": receipt.get("evaluator_digest"),
        "frozen_split_digest": receipt.get("frozen_split_digest"),
        "receipt_digest": _digest(dict(receipt)),
    }


class MetricMaterializer:
    """Ask the bounded engineering agent to add a missing evaluator safely."""

    def __init__(self, engineering_agent: Any | None = None):
        self.engineering_agent = engineering_agent

    def materialize(self, definition: MetricDefinition, *, model_report: Mapping[str, Any], reference_fixtures: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
        if self.engineering_agent is None:
            return {"state": "abstain", "reason": "engineering agent not configured", "metric_id": definition.metric_id}
        result = self.engineering_agent.run(
            objective="Materialize and validate the evaluator for benchmark metric " + definition.metric_id,
            context={
                "metric_definition": asdict(definition),
                "model_report": dict(model_report),
                "reference_fixtures": list(reference_fixtures),
                "required_receipt": {
                    "contract_tests_passed": True,
                    "reference_alignment_passed": True,
                    "deterministic_repeat_passed": True,
                    "evaluator_digest": "sha256:<digest>",
                    "frozen_split_digest": "sha256:<digest>",
                },
                "promotion_rule": "Do not claim formal availability unless every receipt field is independently produced by commands inside the isolated worktree.",
            },
        )
        receipt = result.get("result", {}) if isinstance(result, Mapping) else {}
        if not isinstance(receipt, Mapping):
            return {"state": "abstain", "reason": "missing materialization receipt", "agent_result": result}
        gate = validate_metric_materialization(definition, receipt)
        events = result.get("events", []) if isinstance(result.get("events", []), Sequence) else []
        successful_actions = {
            str(event.get("action"))
            for event in events
            if isinstance(event, Mapping) and isinstance(event.get("result"), Mapping) and event["result"].get("state") == "ok"
        }
        missing_actions = sorted({"run_tests", "collect_artifacts"} - successful_actions)
        if missing_actions:
            gate = {**gate, "state": "abstain", "errors": [*gate.get("errors", []), "missing_engineering_actions:" + ",".join(missing_actions)]}
        return {**gate, "agent_state": result.get("state"), "agent_steps": result.get("steps"), "agent_events": list(events)}


def _improvement(direction: str, baseline: float, candidate: float) -> float:
    return candidate - baseline if direction == "maximize" else baseline - candidate


def validate_metric_selection(selection: Mapping[str, Any], catalog: MetricCatalog, model_report: Mapping[str, Any], available_signals: Iterable[str] = ()) -> dict[str, Any]:
    """Validate an AI selection and return a normalized plan or abstention."""
    eligible, rejected = catalog.candidates(model_report, available_signals)
    eligible_ids = {item.metric_id: item for item in eligible}
    primary = str(selection.get("primary", ""))
    protected = tuple(dict.fromkeys(str(value) for value in selection.get("protected", ())))
    diagnostic = tuple(dict.fromkeys(str(value) for value in selection.get("diagnostic", ())))
    errors: list[str] = []
    if primary not in eligible_ids:
        errors.append("primary_not_eligible:" + primary)
    for metric_id in (*protected, *diagnostic):
        if metric_id not in eligible_ids:
            errors.append("metric_not_eligible:" + metric_id)
    if primary and primary in eligible_ids and ("primary" not in eligible_ids[primary].role_candidates or eligible_ids[primary].diagnostic_only or not eligible_ids[primary].ground_truth):
        errors.append("primary_not_formal_ground_truth_metric:" + primary)
    for metric_id in protected:
        definition = eligible_ids.get(metric_id)
        if definition and ("protected" not in definition.role_candidates or definition.diagnostic_only or not definition.ground_truth):
            errors.append("protected_not_formal_ground_truth_metric:" + metric_id)
    if errors:
        return {"state": "abstain", "reason": "invalid_metric_selection", "errors": errors, "rejected": rejected}
    definitions = [eligible_ids[metric_id] for metric_id in (primary, *protected, *diagnostic) if metric_id]
    formal = [primary, *protected]
    pilot = [metric_id for metric_id in formal if eligible_ids[metric_id].cost != "high"]
    if primary not in pilot:
        pilot.insert(0, primary)
    return {
        "state": "validated",
        "primary": primary,
        "protected": list(protected),
        "diagnostic": list(diagnostic),
        "definitions": [asdict(item) for item in definitions],
        "catalog_id": catalog.catalog_id,
        "catalog_digest": catalog.digest(),
        "rejected": rejected,
        "rationale": str(selection.get("rationale", "")),
        "evaluation_order": list(selection.get("evaluation_order", [primary, *protected, *diagnostic])),
        "evaluation_stages": {
            "pilot_metrics": pilot,
            "promotion_metrics": formal,
            "diagnostic_metrics": list(diagnostic),
            "rule": "run lower-cost pilot metrics first; only a candidate with a non-harmful pilot proceeds to all formal held-out metrics",
        },
    }


def evaluate_metric_bundle(raw_result: Mapping[str, Any], plan: Mapping[str, Any], *, split: str = "heldout") -> dict[str, Any]:
    """Evaluate baseline/candidate metric maps under primary/protected gates.

    This function is intentionally strict: missing or non-finite rows produce
    abstention, and a protected regression cannot be hidden by an aggregate.
    """
    baseline = raw_result.get("baseline_metrics")
    candidate = raw_result.get("candidate_metrics")
    definitions = {str(item["metric_id"]): item for item in plan.get("definitions", ()) if isinstance(item, Mapping) and item.get("metric_id")}
    primary = str(plan.get("primary", ""))
    protected = [str(item) for item in plan.get("protected", ())]
    required = [primary, *protected]
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping) or not definitions or not primary:
        return {"outcome": "abstain", "reason": "metric_bundle_missing", "split": split}
    missing = [metric_id for metric_id in required if metric_id not in baseline or metric_id not in candidate or metric_id not in definitions]
    if missing:
        return {"outcome": "abstain", "reason": "missing_metrics", "missing": missing, "split": split}
    receipts = raw_result.get("metric_receipts", {})
    if not isinstance(receipts, Mapping):
        receipts = {}
    unvalidated = []
    for metric_id in required:
        definition = definitions[metric_id]
        if str(definition.get("implementation_status", "catalogued")) == "validated":
            continue
        values = dict(definition)
        for key in ("role_candidates", "required_signals", "required_capabilities", "architecture_facets", "source_refs"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        gate = validate_metric_materialization(MetricDefinition(**values), receipts.get(metric_id, {}))
        if gate["state"] != "validated":
            unvalidated.append({"metric_id": metric_id, "errors": gate["errors"]})
    if unvalidated:
        return {"outcome": "abstain", "reason": "metric_evaluator_not_validated", "metrics": unvalidated, "split": split}
    deltas: dict[str, float] = {}
    for metric_id, definition in definitions.items():
        try:
            before, after = float(baseline[metric_id]), float(candidate[metric_id])
        except (TypeError, ValueError, KeyError):
            return {"outcome": "abstain", "reason": "non_numeric_metric", "metric": metric_id, "split": split}
        if not math.isfinite(before) or not math.isfinite(after):
            return {"outcome": "abstain", "reason": "non_finite_metric", "metric": metric_id, "split": split}
        deltas[metric_id] = _improvement(str(definition.get("direction", "maximize")), before, after)
    threshold = abs(float(raw_result.get("practical_threshold", plan.get("practical_threshold", 0.0)) or 0.0))
    protected_ok = all(deltas[item] >= -threshold for item in protected)
    primary_delta = deltas[primary]
    outcome = "confirmed_positive" if primary_delta > threshold and protected_ok else ("harmful" if primary_delta < -threshold or not protected_ok else "null")
    return {"outcome": outcome, "delta": primary_delta, "deltas": deltas, "protected_ok": protected_ok, "practical_threshold": threshold, "metric_plan_digest": _digest(plan), "metrics": definitions, "split": split}
