"""AI-assisted autonomous search planning and dual-route idea extraction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .contracts import canonical_digest
from .runtime import AIProvider


@dataclass(frozen=True)
class SearchPlan:
    queries: tuple[str, ...]
    rationale: str
    adjacent_fields: tuple[str, ...]


@dataclass(frozen=True)
class CandidateIdea:
    idea_id: str
    title: str
    mechanism: str
    expected_effect: str
    required_capabilities: tuple[str, ...]
    risks: tuple[str, ...]
    evidence_basis: tuple[str, ...]


def diagnostic_gap_report(graph: dict[str, Any]) -> dict[str, Any]:
    """Summarize diagnosis dimensions that lack positive method evidence."""
    dimensions: set[str] = set()
    methods: list[dict[str, Any]] = []
    compatibility_only: list[str] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        payload = node.get("payload", {}) if isinstance(node.get("payload"), dict) else {}
        if node.get("kind") == "portrait":
            dimensions.update(str(value) for value in payload.get("diagnostic_dimensions", []) if str(value))
        elif node.get("kind") == "fingerprint":
            observation = payload.get("observation", {}) if isinstance(payload.get("observation"), dict) else {}
            if observation.get("status") not in {"evaluated", "measured"}:
                compatibility_only.append(str(payload.get("probe_id", node.get("key", "unknown"))))
        elif node.get("kind") == "method":
            diagnoses = payload.get("diagnostic_dimensions", payload.get("diagnoses", []))
            if isinstance(diagnoses, str):
                diagnoses = [diagnoses]
            methods.append({
                "method_id": str(node.get("id", "")),
                "name": str(node.get("display_name") or node.get("key", "")),
                "status": str(node.get("status") or "unknown"),
                "diagnostic_dimensions": sorted(set(str(value) for value in diagnoses if str(value))),
            })
    coverage = {}
    for dimension in sorted(dimensions):
        relevant = [item for item in methods if dimension in item["diagnostic_dimensions"]]
        positive = [item for item in relevant if item["status"] in {"confirmed_positive", "positive"}]
        coverage[dimension] = {
            "attempted_methods": sorted(set(item["name"] for item in relevant)),
            "positive_methods": sorted(set(item["name"] for item in positive)),
            "state": "positively_covered" if positive else ("attempted_without_positive" if relevant else "unexplored"),
        }
    gaps = [
        {"diagnostic_dimension": key, **value}
        for key, value in coverage.items()
        if value["state"] != "positively_covered"
    ]
    return {
        "diagnostic_dimensions": sorted(dimensions),
        "coverage": coverage,
        "gaps": gaps,
        "compatibility_only_probes": sorted(set(compatibility_only)),
        "claim_boundary": "Coverage gaps prioritize experiments; compatibility-only probes do not establish a measured model defect.",
    }


class GapDrivenIdeaPlanner:
    """Ask the shared AI to propose source-grounded ideas for graph gaps."""

    def __init__(self, ai: AIProvider):
        self.ai = ai

    def propose(
        self,
        *,
        objective: str,
        portrait: dict[str, Any],
        graph: dict[str, Any],
        documents: list[dict[str, Any]],
        count: int,
        training_only: bool = True,
    ) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be positive")
        gaps = diagnostic_gap_report(graph)
        prompt = json.dumps({
            "objective": objective,
            "portrait": portrait,
            "diagnostic_gap_report": gaps,
            "retrieved_sources": documents,
            "constraints": [
                f"Return exactly {count} distinct high-value candidates.",
                "Every candidate must target at least one diagnostic gap and explain why it is not redundant with prior methods.",
                "Every candidate must be a training intervention that creates a new checkpoint; inference-only, sampling-only, and evaluator-only ideas are forbidden." if training_only else "Declare whether the intervention trains weights.",
                "Every candidate must cite at least one retrieved source URL and title.",
                "Use a stable semantic kebab-case idea_id without experiment numbers, version numbers, seeds, or replication suffixes.",
                "The implementation plan must be concrete enough for an autonomous engineering agent to materialize in an isolated worktree.",
                "Do not claim that a compatibility-only probe measured a defect; it only identifies an evidence gap.",
            ],
            "output_schema": {"ideas": [{
                "idea_id": "semantic-kebab-id", "title": "human-readable title",
                "display_name_zh": "concise Chinese community name",
                "intervention_family": "training", "produces_checkpoint": True,
                "target_diagnostic_dimensions": [], "mechanism": "",
                "novelty_against_prior_methods": "", "implementation_plan": "",
                "implementation_surface": [], "source_evidence": [{"title": "", "url": "", "mechanism": ""}],
                "primary_metrics": [], "protected_metrics": [], "risks": [],
                "claim_boundary": "",
            }]},
        }, sort_keys=True)
        raw = self.ai.complete(role="gap_driven_training_idea_planner", prompt=prompt)
        if raw.strip().startswith("```"):
            raw = raw.strip().split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(raw)
        ideas = value.get("ideas", value) if isinstance(value, dict) else value
        if not isinstance(ideas, list) or len(ideas) != count:
            raise RuntimeError(f"expected exactly {count} ideas")
        gap_names = {item["diagnostic_dimension"] for item in gaps["gaps"]}
        retrieved_urls = {
            str(document.get("source_url") or document.get("url"))
            for document in documents
            if isinstance(document, dict) and str(document.get("source_url") or document.get("url", "")).startswith(("http://", "https://"))
        }
        seen: set[str] = set()
        validated: list[dict[str, Any]] = []
        for raw_idea in ideas:
            if not isinstance(raw_idea, dict):
                raise RuntimeError("idea must be an object")
            idea = dict(raw_idea)
            idea_id = str(idea.get("idea_id", "")).strip().lower()
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", idea_id) or re.search(r"(?:^|-)v?\\d+(?:-|$)", idea_id):
                raise RuntimeError(f"invalid semantic idea_id: {idea_id}")
            if idea_id in seen:
                raise RuntimeError(f"duplicate idea_id: {idea_id}")
            if training_only and (idea.get("produces_checkpoint") is not True or str(idea.get("intervention_family", "")).lower() != "training"):
                raise RuntimeError(f"non-training idea rejected: {idea_id}")
            targets = {str(value) for value in idea.get("target_diagnostic_dimensions", [])}
            if gap_names and not targets.intersection(gap_names):
                raise RuntimeError(f"idea does not target a diagnostic gap: {idea_id}")
            sources = idea.get("source_evidence", [])
            if not isinstance(sources, list) or not sources or not all(isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://")) for item in sources):
                raise RuntimeError(f"idea lacks retrieved source evidence: {idea_id}")
            if any(str(item["url"]) not in retrieved_urls for item in sources):
                raise RuntimeError(f"idea cites a source outside the retrieval ledger: {idea_id}")
            idea["objective"] = str(idea.get("objective") or idea.get("title") or idea_id)
            idea["display_name"] = str(idea.get("display_name_zh") or idea.get("title") or idea_id)
            idea["diagnostic_gap_basis"] = [item for item in gaps["gaps"] if item["diagnostic_dimension"] in targets]
            seen.add(idea_id)
            validated.append(idea)
        return validated


def relevant_source_documents(
    documents: list[dict[str, Any]],
    *,
    objective: str,
    portrait: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reject obvious search noise before it can ground an experiment."""
    strong_terms = {
        str(value).lower().replace("_", " ")
        for field in ("architecture_facets", "capabilities")
        for value in portrait.get(field, [])
        if len(str(value)) >= 4
    }
    strong_terms.update({"world model", "video diffusion", "diffusion model", "temporal model"})
    objective_terms = {
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", objective)
        if token.lower() not in {"select", "value", "training", "create", "without", "evidence", "dimensions"}
    }
    selected = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        text = " ".join(str(document.get(field, "")) for field in ("title", "text", "snippet")).lower()
        strong_hits = {term for term in strong_terms if term in text}
        objective_hits = {term for term in objective_terms if term in text}
        if strong_hits and len(strong_hits | objective_hits) >= 2:
            selected.append({**document, "relevance_terms": sorted(strong_hits | objective_hits)})
    return selected


class IdeaRoute(Protocol):
    route_id: str

    def extract(self, documents: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]: ...


class AIJsonRoute:
    """An IdeaRoute backed by the shared OpenAI-compatible provider."""

    def __init__(self, ai: AIProvider, role: str):
        self.ai = ai
        self.route_id = role

    def extract(self, documents: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = json.dumps({"context": context, "documents": documents, "instruction": "Extract research ideas as a JSON list. Each item must contain title, mechanism, expected_effect, required_capabilities, risks, evidence_basis."}, sort_keys=True)
        try:
            value = json.loads(self.ai.complete(role=self.route_id, prompt=prompt))
            return value if isinstance(value, list) else value.get("ideas", [])
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []


class AutonomousResearchPlanner:
    def __init__(self, ai: AIProvider):
        self.ai = ai

    def plan(self, objective: str, portrait: dict[str, Any]) -> SearchPlan:
        prompt = json.dumps({"objective": objective, "portrait": portrait, "instruction": "Choose relevant and adjacent fields autonomously; return JSON with queries, rationale, adjacent_fields."}, sort_keys=True)
        raw = self.ai.complete(role="research_planner", prompt=prompt)
        try:
            value = json.loads(raw)
            queries = tuple(
                str(v.get("query", "")).strip() if isinstance(v, dict) else str(v).strip()
                for v in value.get("queries", [])
                if (str(v.get("query", "")).strip() if isinstance(v, dict) else str(v).strip())
            )
            fields = tuple(str(v) for v in value.get("adjacent_fields", []) if str(v).strip())
            if queries:
                return SearchPlan(queries, str(value.get("rationale", "")), fields)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return SearchPlan((objective,), "fallback objective query", ())


class DualRouteIdeator:
    def __init__(self, routes: tuple[IdeaRoute, IdeaRoute]):
        self.routes = routes

    def extract(self, documents: list[dict[str, Any]], context: dict[str, Any]) -> list[CandidateIdea]:
        candidates: dict[str, CandidateIdea] = {}
        for route in self.routes:
            for raw in route.extract(documents, context):
                title = str(raw.get("title", "")).strip()
                mechanism = str(raw.get("mechanism", "")).strip()
                if not title or not mechanism:
                    continue
                body = {"title": title, "mechanism": mechanism, "expected_effect": raw.get("expected_effect", "")}
                idea = CandidateIdea(
                    idea_id="idea-" + canonical_digest(body)[7:23],
                    title=title,
                    mechanism=mechanism,
                    expected_effect=str(raw.get("expected_effect", "")),
                    required_capabilities=tuple(str(v) for v in raw.get("required_capabilities", [])),
                    risks=tuple(str(v) for v in raw.get("risks", [])),
                    evidence_basis=tuple(str(v) for v in raw.get("evidence_basis", [])),
                )
                candidates[idea.idea_id] = idea
        return list(candidates.values())
