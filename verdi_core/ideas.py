"""AI-assisted autonomous search planning and dual-route idea extraction."""

from __future__ import annotations

import json
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
            queries = tuple(str(v) for v in value.get("queries", []) if str(v).strip())
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
