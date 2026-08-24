"""Replaceable model adapter protocol and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ModelAdapter(Protocol):
    adapter_id: str
    version: str

    def inspect(self) -> dict[str, Any]: ...

    def probe(self, probe_id: str) -> dict[str, Any]: ...

    def evaluate(self, intervention: dict[str, Any], split: str) -> dict[str, Any]: ...

    def intervene(self, hypothesis: dict[str, Any]) -> dict[str, Any]: ...


class ExperimentWorker(Protocol):
    """Executes a candidate and returns raw, reproducible artifacts."""

    def execute(self, hypothesis: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]: ...


class Evaluator(Protocol):
    """Scores frozen artifacts on a held-out split and classifies evidence."""

    evaluator_id: str

    def evaluate(self, artifacts: dict[str, Any], *, split: str, metrics: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: str
    version: str
    capabilities: tuple[str, ...]
    implementation_digest: str
