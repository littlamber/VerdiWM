"""Build a deterministic evidence-graph projection from VerdiWM artifacts.

The graph is a read-only projection. Source receipts and CAS objects remain
authoritative; nodes carry source paths and hashes so a graph can be rebuilt
after a crash or archive migration without changing scientific state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EvidenceGraphError(ValueError):
    """The graph input or output contract is invalid."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_.:" else "_" for char in value)[:240]


def _node_id(kind: str, key: str) -> str:
    return f"{kind}:{_safe_id(key)}:{_sha256(f'{kind}:{key}'.encode())[:16]}"


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def node(self, kind: str, key: str, *, source: str | None = None, **attrs: Any) -> str:
        identifier = _node_id(kind, key)
        record = self.nodes.setdefault(identifier, {"id": identifier, "kind": kind, "key": key})
        for name, value in attrs.items():
            if value is not None:
                record[name] = value
        if source:
            record.setdefault("sources", set()).add(source)
        return identifier

    def edge(self, source: str, relation: str, target: str, *, evidence: str | None = None) -> None:
        key = f"{source}|{relation}|{target}"
        record = self.edges.setdefault(
            key,
            {"id": _node_id("edge", key), "source": source, "relation": relation, "target": target},
        )
        if evidence:
            record.setdefault("evidence", set()).add(evidence)

    def document(self, *, input_root: Path, source_count: int) -> dict[str, Any]:
        nodes = []
        for value in sorted(self.nodes.values(), key=lambda item: item["id"]):
            item = dict(value)
            if isinstance(item.get("sources"), set):
                item["sources"] = sorted(item["sources"])
            nodes.append(item)
        edges = []
        for value in sorted(self.edges.values(), key=lambda item: item["id"]):
            item = dict(value)
            if isinstance(item.get("evidence"), set):
                item["evidence"] = sorted(item["evidence"])
            edges.append(item)
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-evidence-graph",
            "state": "ready",
            "input_root": str(input_root),
            "source_count": source_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "claim_boundary": (
                "This is a provenance-preserving projection. Only source artifacts that "
                "declare settled or verified state can create verified evidence edges."
            ),
            "nodes": nodes,
            "edges": edges,
        }


def build_evidence_graph(input_root: Path) -> dict[str, Any]:
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise EvidenceGraphError("EVIDENCE_GRAPH_INPUT_INVALID")
    graph = EvidenceGraph()
    source_count = 0
    for path in sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payloads = _load_payloads(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for index, payload in enumerate(payloads):
            if not isinstance(payload, Mapping):
                continue
            source_count += 1
            _project_document(graph, payload, source=str(path), ordinal=index)
    return graph.document(input_root=root, source_count=source_count)


def _load_payloads(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def _project_document(graph: EvidenceGraph, payload: Mapping[str, Any], *, source: str, ordinal: int) -> None:
    artifact = str(payload.get("artifact_type") or "document")
    identity = str(
        payload.get("campaign_id")
        or payload.get("trial_id")
        or payload.get("record_id")
        or payload.get("candidate_id")
        or payload.get("experiment_id")
        or f"{Path(source).name}:{ordinal}"
    )
    root = graph.node("artifact", f"{artifact}:{identity}:{ordinal}", source=source, artifact_type=artifact, state=payload.get("state"), verdict=payload.get("verdict"), settlement_state=payload.get("settlement_state"))
    kind_map = {
        "target_backbone": "backbone",
        "model_family": "backbone",
        "model_ref": "model",
        "environment": "environment",
        "scenario": "scenario",
        "goal_id": "goal",
        "primitive": "primitive",
        "primitive_family": "primitive",
        "probe_id": "probe",
        "candidate_id": "candidate",
        "trial_id": "trial",
        "campaign_id": "campaign",
        "experiment_id": "experiment",
        "receipt_ref": "receipt",
        "verdict_ref": "verdict",
        "certificate_status": "certificate",
    }
    for field, kind in kind_map.items():
        value = payload.get(field)
        if value is None or isinstance(value, (dict, list)):
            continue
        child = graph.node(kind, str(value), source=source, value=value)
        relation = {
            "target_backbone": "targets_backbone",
            "model_family": "uses_backbone",
            "model_ref": "references_model",
            "environment": "evaluated_in",
            "scenario": "evaluated_scenario",
            "goal_id": "optimizes_goal",
            "primitive": "tests_primitive",
            "primitive_family": "tests_primitive_family",
            "probe_id": "uses_probe",
            "candidate_id": "proposes_candidate",
            "trial_id": "settles_trial",
            "campaign_id": "belongs_to_campaign",
            "experiment_id": "belongs_to_experiment",
            "receipt_ref": "supported_by_receipt",
            "verdict_ref": "supported_by_verdict",
            "certificate_status": "has_certificate",
        }[field]
        graph.edge(root, relation, child, evidence=source)
    if payload.get("settlement_state") == "settled" or payload.get("state") in {"settled", "verified", "ready"}:
        verified = graph.node("verified_evidence", f"{artifact}:{identity}:{ordinal}", source=source, artifact_type=artifact)
        graph.edge(root, "provides_verified_evidence", verified, evidence=source)
    if payload.get("certificate_status") == "licensed" or payload.get("status") == "licensed":
        licensed = graph.node("transfer_license", f"{artifact}:{identity}:{ordinal}", source=source, status="licensed")
        graph.edge(root, "licenses_transfer", licensed, evidence=source)
    for field, relation in (("evidence_refs", "cites_evidence"), ("source_map_ids", "derived_from"), ("parent_campaign_id", "reproduces")):
        values = payload.get(field)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, (str, int)):
                child = graph.node("evidence", str(value), source=source)
                graph.edge(root, relation, child, evidence=source)


def write_evidence_graph(*, input_root: Path, output_root: Path) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = build_evidence_graph(input_root)
    temporary = Path(tempfile.mkdtemp(prefix=".evidence-graph-", dir=destination))
    try:
        (temporary / "graph.json").write_bytes(_canonical(report) + b"\n")
        (temporary / "manifest.json").write_bytes(_canonical({k: report[k] for k in ("schema_version", "artifact_type", "state", "source_count", "node_count", "edge_count", "claim_boundary")}) + b"\n")
        for name in ("graph.json", "manifest.json"):
            os.replace(temporary / name, destination / name)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return report


def query_evidence_graph(
    graph_path: Path,
    *,
    entity: str,
    filters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Query graph nodes or edges using exact scalar filters.

    The query surface intentionally stays small and deterministic. It is an
    index-like read API, not a second source of scientific truth.
    """

    path = Path(graph_path).expanduser().resolve()
    if path.is_dir():
        path = path / "graph.json"
    if not path.is_file() or path.is_symlink():
        raise EvidenceGraphError("EVIDENCE_GRAPH_NOT_FOUND")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "verdiwm-evidence-graph":
        raise EvidenceGraphError("EVIDENCE_GRAPH_CONTRACT_INVALID")
    if entity not in {"nodes", "edges"}:
        raise EvidenceGraphError("EVIDENCE_GRAPH_ENTITY_INVALID")
    rows = payload.get(entity)
    if not isinstance(rows, list):
        raise EvidenceGraphError("EVIDENCE_GRAPH_ROWS_INVALID")
    normalized = {str(key): str(value) for key, value in (filters or {}).items() if key not in {"limit", "offset"}}
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and all(str(row.get(key)) == value for key, value in normalized.items())
    ]
    try:
        offset = max(0, int((filters or {}).get("offset", "0")))
        limit = min(1000, max(1, int((filters or {}).get("limit", "100"))))
    except (TypeError, ValueError) as exc:
        raise EvidenceGraphError("EVIDENCE_GRAPH_PAGING_INVALID") from exc
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-evidence-graph-query",
        "entity": entity,
        "filters": normalized,
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "items": selected[offset : offset + limit],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a VerdiWM evidence graph projection")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_evidence_graph(input_root=args.input_root, output_root=args.output_root), sort_keys=True))
    return 0
