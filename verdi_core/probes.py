"""Probe registry with evidence-driven, AI-proposed evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contracts import canonical_digest
from .runtime import AIProvider


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    question: str
    dimensions: tuple[str, ...]
    required_capabilities: tuple[str, ...] = ()
    parent_probe_id: str | None = None


class ProbeRegistry:
    def __init__(self, probes: list[ProbeSpec] | None = None):
        self._probes = {probe.probe_id: probe for probe in probes or []}

    def register(self, probe: ProbeSpec) -> bool:
        if probe.probe_id in self._probes:
            return False
        self._probes[probe.probe_id] = probe
        return True

    def all(self) -> list[ProbeSpec]:
        return list(self._probes.values())


class ProbeEvolution:
    def __init__(self, ai: AIProvider | None = None):
        self.ai = ai

    def propose(self, portrait: dict[str, Any], failures: list[dict[str, Any]]) -> list[ProbeSpec]:
        if self.ai is None:
            return []
        prompt = json.dumps({"portrait": portrait, "failures": failures, "instruction": "Invent diagnostic probes that distinguish the observed failure; return JSON list."}, sort_keys=True)
        try:
            values = json.loads(self.ai.complete(role="probe_evolver", prompt=prompt))
            result = []
            for value in values if isinstance(values, list) else values.get("probes", []):
                question = str(value.get("question", "")).strip()
                if not question:
                    continue
                body = {"question": question, "dimensions": value.get("dimensions", [])}
                result.append(ProbeSpec("probe-" + canonical_digest(body)[7:23], question, tuple(map(str, body["dimensions"])), tuple(map(str, value.get("required_capabilities", [])))))
            return result
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []
