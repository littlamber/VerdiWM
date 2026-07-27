"""Versioned probe-role registry and verdict-evidence projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, validate_document


class ProbeRegistryError(ValueError):
    """Probe roles or report projection violate the frozen registry."""


@dataclass(frozen=True)
class ProbeDefinition:
    probe_id: str
    role: str
    report_field: str
    description: str


@dataclass(frozen=True)
class ProbeRegistry:
    library_version: str
    probes: tuple[ProbeDefinition, ...]

    @property
    def verdict_probe_ids(self) -> tuple[str, ...]:
        return tuple(probe.probe_id for probe in self.probes if probe.role == "verdict")

    def probe(self, probe_id: str) -> ProbeDefinition:
        for probe in self.probes:
            if probe.probe_id == probe_id:
                return probe
        raise ProbeRegistryError(f"PROBE_UNKNOWN:{probe_id}")


def load_probe_registry(path: Path, *, root: Path | None = None) -> ProbeRegistry:
    base = (root or Path(__file__).resolve().parents[2]).resolve()
    raw_path = _resolve_inside(base, path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    try:
        validate_document("probe_registry", payload, root=base)
    except ContractValidationError as exc:
        raise ProbeRegistryError(f"PROBE_REGISTRY_CONTRACT_INVALID:{exc}") from exc
    seen: set[str] = set()
    probes: list[ProbeDefinition] = []
    for item in payload["probes"]:
        probe_id = str(item["id"])
        if probe_id in seen:
            raise ProbeRegistryError(f"PROBE_DUPLICATE:{probe_id}")
        seen.add(probe_id)
        probes.append(
            ProbeDefinition(
                probe_id=probe_id,
                role=str(item["role"]),
                report_field=str(item["report_field"]),
                description=str(item["description"]),
            )
        )
    registry = ProbeRegistry(library_version=str(payload["library_version"]), probes=tuple(probes))
    if not registry.verdict_probe_ids:
        raise ProbeRegistryError("PROBE_VERDICT_SET_EMPTY")
    return registry


def build_verdict_evidence(failure_report: Mapping[str, Any], registry: ProbeRegistry) -> dict[str, object]:
    """Project a full diagnostic report down to verifier-visible evidence."""

    probes: dict[str, object] = {}
    for probe_id in registry.verdict_probe_ids:
        probe = registry.probe(probe_id)
        if probe.report_field not in failure_report:
            raise ProbeRegistryError(f"VERDICT_PROBE_FIELD_MISSING:{probe.report_field}")
        probes[probe.probe_id] = failure_report[probe.report_field]
    evidence = {
        "schema_version": 1,
        "artifact_type": "wmloop-verdict-evidence",
        "env": _string(failure_report, "env"),
        "model_ref": _string(failure_report, "model_ref"),
        "round": _integer(failure_report, "round"),
        "goal_id": _string(failure_report, "goal_id"),
        "verdict_probe_ids": list(registry.verdict_probe_ids),
        "probes": probes,
        "evidence_frames": list(failure_report.get("evidence_frames") or []),
    }
    try:
        validate_document("verdict_evidence", evidence)
    except ContractValidationError as exc:
        raise ProbeRegistryError(f"VERDICT_EVIDENCE_CONTRACT_INVALID:{exc}") from exc
    return evidence


def _resolve_inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ProbeRegistryError("PROBE_REGISTRY_PATH_OUTSIDE_ROOT")
    if not resolved.is_file():
        raise ProbeRegistryError("PROBE_REGISTRY_MISSING")
    return resolved


def _string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ProbeRegistryError(f"VERDICT_EVIDENCE_FIELD_INVALID:{key}")
    return value


def _integer(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProbeRegistryError(f"VERDICT_EVIDENCE_FIELD_INVALID:{key}")
    return value
