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
import sqlite3
import tempfile
from collections import Counter
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
        kind_counts = Counter(str(item["kind"]) for item in nodes)
        relation_counts = Counter(str(item["relation"]) for item in edges)
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-evidence-graph",
            "state": "ready",
            "input_root": str(input_root),
            "source_count": source_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_kind_counts": dict(sorted(kind_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "claim_boundary": (
                "This is a provenance-preserving projection. Only source artifacts that "
                "declare settled or verified state can create verified evidence edges."
            ),
            "nodes": nodes,
            "edges": edges,
        }


def build_evidence_graph(
    input_root: Path,
    *,
    archive_db: Path | None = None,
) -> dict[str, Any]:
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
    if archive_db is not None:
        source_count += _project_archive(graph, Path(archive_db).expanduser().resolve())
    return graph.document(input_root=root, source_count=source_count)


def _project_archive(graph: EvidenceGraph, archive_db: Path) -> int:
    if not archive_db.is_file() or archive_db.is_symlink():
        raise EvidenceGraphError("EVIDENCE_GRAPH_ARCHIVE_INVALID")
    source = str(archive_db)
    try:
        connection = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT trial_id, proposal_id, goal_id, library_version, failure_context_ref, verdict_ref, receipt_ref, settlement_json FROM trials ORDER BY trial_id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise EvidenceGraphError("EVIDENCE_GRAPH_ARCHIVE_READ_FAILED") from exc
    finally:
        try:
            connection.close()
        except (NameError, UnboundLocalError):
            pass
    for ordinal, row in enumerate(rows):
        payload: dict[str, Any] = {
            "artifact_type": "verdiwm-archive-settled-trial",
            "trial_id": row["trial_id"],
            "proposal_id": row["proposal_id"],
            "goal_id": row["goal_id"],
            "library_version": row["library_version"],
            "failure_context_ref": row["failure_context_ref"],
            "verdict_ref": row["verdict_ref"],
            "receipt_ref": row["receipt_ref"],
            "settlement_state": "settled",
        }
        try:
            settlement = json.loads(str(row["settlement_json"]))
        except (TypeError, json.JSONDecodeError):
            settlement = {}
        if isinstance(settlement, Mapping):
            payload["settlement"] = settlement
            payload["evidence_scope"] = settlement.get("evidence_scope")
        _project_document(graph, payload, source=source, ordinal=ordinal)
    return len(rows)


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
        "source_id": "research_source",
        "assessment_digest": "source_assessment",
        "implementation_revision": "implementation",
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
            "source_id": "derived_from_research_source",
            "assessment_digest": "bound_to_source_assessment",
            "implementation_revision": "implemented_by_revision",
        }[field]
        graph.edge(root, relation, child, evidence=source)
    _project_portable_experience(graph, payload, root=root, source=source)
    settlement = payload.get("settlement")
    settlement_state = settlement.get("state") if isinstance(settlement, Mapping) else None
    evidence_scope = (
        settlement.get("evidence_scope")
        if isinstance(settlement, Mapping)
        else payload.get("evidence_scope")
    )
    is_settled = (
        payload.get("settlement_state") == "settled"
        or settlement_state == "settled"
        or payload.get("verification_state") in {"settled", "verified"}
    )
    stage = str(
        payload.get("stage")
        or (settlement.get("stage") if isinstance(settlement, Mapping) else "")
        or ""
    )
    verification_state = str(payload.get("verification_state") or "")
    verdict_ref = payload.get("verdict_ref")
    frozen_verifier_bound = (
        verification_state == "verified"
        and stage == "confirm"
        and isinstance(verdict_ref, str)
        and _is_content_addressed(verdict_ref)
    )
    outcome = str(payload.get("outcome") or "")
    verified_negative = (
        verification_state == "verified"
        and outcome in {"rejected_at_screen", "rejected_at_confirm"}
        and isinstance(verdict_ref, str)
        and _is_content_addressed(verdict_ref)
    )
    verified_operational_failure = (
        verification_state == "verified"
        and outcome == "operational_failure"
        and isinstance(verdict_ref, str)
        and _is_content_addressed(verdict_ref)
    )
    if verified_negative:
        negative = graph.node(
            "verified_negative_evidence",
            f"{artifact}:{identity}:{ordinal}",
            source=source,
            artifact_type=artifact,
            outcome=outcome,
        )
        graph.edge(root, "provides_verified_negative_boundary", negative, evidence=source)
    elif verified_operational_failure:
        operational = graph.node(
            "verified_operational_failure",
            f"{artifact}:{identity}:{ordinal}",
            source=source,
            artifact_type=artifact,
        )
        graph.edge(root, "records_verified_operational_failure", operational, evidence=source)
    elif is_settled and (evidence_scope == "exploratory" or stage in {"screen", "gate"}):
        exploratory = graph.node(
            "exploratory_evidence",
            f"{artifact}:{identity}:{ordinal}",
            source=source,
            artifact_type=artifact,
        )
        graph.edge(
            root,
            "provides_exploratory_evidence",
            exploratory,
            evidence=source,
        )
    elif frozen_verifier_bound:
        verified = graph.node("verified_evidence", f"{artifact}:{identity}:{ordinal}", source=source, artifact_type=artifact)
        graph.edge(root, "provides_verified_evidence", verified, evidence=source)
    elif is_settled and stage == "confirm":
        confirmed = graph.node(
            "confirmation_pending_verifier",
            f"{artifact}:{identity}:{ordinal}",
            source=source,
            artifact_type=artifact,
        )
        graph.edge(root, "provides_target_confirmation", confirmed, evidence=source)
    elif is_settled:
        unclassified = graph.node(
            "settled_unclassified_evidence",
            f"{artifact}:{identity}:{ordinal}",
            source=source,
            artifact_type=artifact,
        )
        graph.edge(root, "provides_unclassified_evidence", unclassified, evidence=source)
    licensed_certificate = (
        payload.get("certificate_status") == "licensed"
        and payload.get("stage") == "confirm"
        and frozen_verifier_bound
    ) or (
        payload.get("artifact_type") == "verdiwm-transfer-certificate"
        and payload.get("status") == "licensed"
    )
    if licensed_certificate:
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


def _project_portable_experience(
    graph: EvidenceGraph,
    payload: Mapping[str, Any],
    *,
    root: str,
    source: str,
) -> None:
    if payload.get("artifact_type") != "verdiwm-transferable-experience":
        return
    knowledge = payload.get("portable_knowledge")
    evidence_ir = payload.get("evidence_ir")
    if not isinstance(knowledge, Mapping) or not isinstance(evidence_ir, Mapping):
        return
    mappings = (
        ("model_family", "backbone", "observed_on_backbone"),
        ("capability_class", "capability", "requires_capability"),
        ("goal_protocol", "goal_protocol", "evaluated_under_goal"),
        ("outcome_protocol", "outcome_protocol", "measured_by_outcome"),
        ("dataset_regime", "dataset_regime", "observed_in_regime"),
        ("primitive", "primitive", "tests_primitive"),
    )
    for field, kind, relation in mappings:
        value = knowledge.get(field)
        if isinstance(value, str) and value:
            child = graph.node(kind, value, source=source, value=value)
            graph.edge(root, relation, child, evidence=source)
    for condition in payload.get("anti_conditions", []):
        if isinstance(condition, str) and condition:
            child = graph.node("anti_condition", condition, source=source)
            graph.edge(root, "bounded_by", child, evidence=source)
    status = evidence_ir.get("status")
    authority = evidence_ir.get("authority")
    if not isinstance(status, Mapping) or not isinstance(authority, Mapping):
        return
    if (
        status.get("state") == "transfer_licensed"
        and authority.get("claim_scope") == "transfer_prior"
        and _is_content_addressed(authority.get("goal_binding"))
        and _is_content_addressed(authority.get("evaluator_binding"))
    ):
        licensed = graph.node(
            "transfer_license",
            str(evidence_ir.get("evidence_id") or payload.get("portable_experience_id")),
            source=source,
            status="licensed",
        )
        graph.edge(root, "licenses_transfer_prior", licensed, evidence=source)


def _is_content_addressed(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("cas://", "urn:")):
        return len(value.split(":", 1)[1]) > 0
    return value.startswith("sha256:") and len(value) == len("sha256:") + 64


def write_evidence_graph(
    *,
    input_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    source_root = Path(input_root).expanduser().resolve()
    if destination == source_root or source_root in destination.parents:
        raise EvidenceGraphError("EVIDENCE_GRAPH_OUTPUT_OVERLAPS_INPUT")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = build_evidence_graph(input_root, archive_db=archive_db)
    temporary = Path(tempfile.mkdtemp(prefix=".evidence-graph-", dir=destination))
    try:
        (temporary / "graph.json").write_bytes(_canonical(report) + b"\n")
        (temporary / "manifest.json").write_bytes(_canonical({k: report[k] for k in ("schema_version", "artifact_type", "state", "source_count", "node_count", "edge_count", "claim_boundary")}) + b"\n")
        _write_index(temporary / "graph.db", report)
        for name in ("graph.json", "graph.db", "manifest.json"):
            os.replace(temporary / name, destination / name)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return report


def _write_index(path: Path, report: Mapping[str, Any]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(
            """
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY NOT NULL,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX nodes_kind_key ON nodes(kind, key);
            CREATE TABLE edges (
                id TEXT PRIMARY KEY NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX edges_relation ON edges(relation);
            CREATE INDEX edges_source ON edges(source);
            CREATE INDEX edges_target ON edges(target);
            CREATE TABLE metadata (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL);
            """
        )
        connection.executemany(
            "INSERT INTO nodes(id, kind, key, payload_json) VALUES (?, ?, ?, ?)",
            [
                (
                    str(row["id"]),
                    str(row["kind"]),
                    str(row["key"]),
                    _canonical(row).decode(),
                )
                for row in report["nodes"]
            ],
        )
        connection.executemany(
            "INSERT INTO edges(id, source, relation, target, payload_json) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    str(row["id"]),
                    str(row["source"]),
                    str(row["relation"]),
                    str(row["target"]),
                    _canonical(row).decode(),
                )
                for row in report["edges"]
            ],
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", str(report["schema_version"])),
                ("artifact_type", str(report["artifact_type"])),
                ("node_count", str(report["node_count"])),
                ("edge_count", str(report["edge_count"])),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(path, 0o600)


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
        index_path = path / "graph.db"
        if index_path.is_file() and not index_path.is_symlink():
            return _query_index(index_path, entity=entity, filters=filters)
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


def _query_index(
    path: Path,
    *,
    entity: str,
    filters: Mapping[str, str] | None,
) -> dict[str, Any]:
    if entity not in {"nodes", "edges"}:
        raise EvidenceGraphError("EVIDENCE_GRAPH_ENTITY_INVALID")
    allowed = (
        {"id", "kind", "key"}
        if entity == "nodes"
        else {"id", "source", "relation", "target"}
    )
    supplied = filters or {}
    normalized = {
        str(key): str(value)
        for key, value in supplied.items()
        if key not in {"limit", "offset"}
    }
    unsupported = sorted(set(normalized) - allowed)
    if unsupported:
        raise EvidenceGraphError(
            f"EVIDENCE_GRAPH_FILTER_INVALID:{','.join(unsupported)}"
        )
    try:
        offset = max(0, int(supplied.get("offset", "0")))
        limit = min(1000, max(1, int(supplied.get("limit", "100"))))
    except (TypeError, ValueError) as exc:
        raise EvidenceGraphError("EVIDENCE_GRAPH_PAGING_INVALID") from exc
    clauses = [f"{name} = ?" for name in sorted(normalized)]
    values = [normalized[name] for name in sorted(normalized)]
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        total = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {entity}{where}", values
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"SELECT payload_json FROM {entity}{where} ORDER BY id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
    except sqlite3.Error as exc:
        raise EvidenceGraphError("EVIDENCE_GRAPH_INDEX_READ_FAILED") from exc
    finally:
        try:
            connection.close()
        except (NameError, UnboundLocalError):
            pass
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-evidence-graph-query",
        "entity": entity,
        "filters": normalized,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [json.loads(str(row[0])) for row in rows],
        "query_backend": "sqlite",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a VerdiWM evidence graph projection")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--archive-db", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_evidence_graph(input_root=args.input_root, output_root=args.output_root, archive_db=args.archive_db), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
