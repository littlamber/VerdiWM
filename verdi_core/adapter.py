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


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: str
    version: str
    capabilities: tuple[str, ...]
    implementation_digest: str

