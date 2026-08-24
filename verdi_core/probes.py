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

    def set_status(self, probe_id: str, status: str) -> None:
        if probe_id not in self._probes:
            raise KeyError(probe_id)
        if status not in {"proposed", "sandbox_tested", "admitted", "executed", "evaluated", "promoted", "deprecated"}:
            raise ValueError(status)


class ProbeCampaign:
    """Executes admitted probes and promotes only probes with usable evidence."""

    def __init__(self, registry: ProbeRegistry, state: Any | None = None):
        self.registry, self.state = registry, state

    def run(self, adapter: Any, *, admitted: list[str] | None = None) -> list[dict[str, Any]]:
        selected = set(admitted or [probe.probe_id for probe in self.registry.all()])
        results = []
        for probe in self.registry.all():
            if probe.probe_id not in selected:
                continue
            self.registry.set_status(probe.probe_id, "admitted")
            if self.state:
                self.state.put_probe(probe.probe_id, probe.__dict__, "admitted")
            try:
                value = adapter.probe(probe.probe_id)
                result = {"probe_id": probe.probe_id, "status": "evaluated", "result": value}
                self.registry.set_status(probe.probe_id, "evaluated")
                if self.state:
                    self.state.put_probe(probe.probe_id, result, "evaluated")
            except Exception as exc:
                result = {"probe_id": probe.probe_id, "status": "abstain", "error": str(exc)}
            results.append(result)
        return results


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
