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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class EvidenceGraphError(ValueError):
    """The graph input or output contract is invalid."""


_DISPLAY_KINDS = {
    "artifact": "工件",
    "backbone": "模型骨干",
    "model": "模型",
    "campaign": "任务",
    "experiment": "实验",
    "goal": "目标",
    "environment": "环境",
    "scenario": "场景",
    "primitive": "方法原语",
    "probe": "探针",
    "candidate": "候选方案",
    "trial": "试验",
    "receipt": "运行回执",
    "verdict": "判定",
    "research_source": "文献来源",
    "source_assessment": "来源评估",
    "implementation": "实现版本",
    "evidence": "证据引用",
    "verified_evidence": "已验证正证据",
    "verified_negative_evidence": "已验证负边界",
    "verified_operational_failure": "已验证运维失败",
    "exploratory_evidence": "探索性证据",
    "confirmation_pending_verifier": "待验证确认",
    "settled_unclassified_evidence": "未分类已结算证据",
    "transfer_license": "迁移许可",
}

_DISPLAY_ARTIFACT_TYPES = {
    "verdiwm-evidence-graph": "证据图谱",
    "verdiwm-artifact-lint-report": "产物合规报告",
    "verdiwm-transfer-certificate": "迁移证书",
    "verdiwm-transferable-experience": "可迁移经验",
    "verdiwm-archive-settled-trial": "已结算试验",
    "verdiwm-campaign-revision-record": "任务修订记录",
    "verdiwm-campaign-dispatch": "任务调度清单",
    "verdiwm-adapter-repair-manifest": "适配器修复清单",
}

_TECHNICAL_NODE_KINDS = {"artifact", "evidence", "source_assessment", "implementation"}
_PRESENTATION_PROJECTION_TYPES = {
    "verdiwm-artifact-lint-report",
    "verdiwm-atlas",
    "verdiwm-evidence-graph",
    "verdiwm-evidence-graph-query",
    "verdiwm-mechanism-board",
    "verdiwm-portable-knowledge-graph",
    "verdiwm-portable-knowledge-graph-query",
}


def _semantic_slug(value: Any) -> str:
    """Turn an identity into a compact human label without exposing paths."""
    text = str(value or "").strip()
    if not text:
        return "未命名"
    if text.startswith(("cas://", "urn:", "sha256:")):
        return "内容寻址对象"
    return text.replace("_", "-")[:96]


def _artifact_label(artifact_type: Any, key: Any) -> str:
    artifact = str(artifact_type or "document")
    title = _DISPLAY_ARTIFACT_TYPES.get(artifact)
    if title is None:
        words = artifact.removeprefix("verdiwm-").replace("-", " ").replace("_", " ")
        title = words.strip().capitalize() or "文档"
    parts = str(key or "").split(":")
    identity = parts[1] if len(parts) > 1 and parts[1] not in {"0", ""} else ""
    return f"{title} · {_semantic_slug(identity)}" if identity else title


def _display_label(node: Mapping[str, Any]) -> str:
    kind = str(node.get("kind") or "node")
    key = node.get("key")
    if kind == "artifact":
        return _artifact_label(node.get("artifact_type"), key)
    if kind in {"model", "backbone"}:
        return _semantic_slug(
            node.get("family") or node.get("model_name") or node.get("value") or key
        )
    if kind in _DISPLAY_KINDS:
        value = node.get("value") or key
        text = str(value)
        if _is_content_addressed(text) or (
            kind in {"source_assessment", "implementation"}
            and len(text) >= 32
            and all(char in "0123456789abcdefABCDEF" for char in text)
        ):
            return _DISPLAY_KINDS[kind]
        return _semantic_slug(value)
    return _semantic_slug(node.get("value") or key)


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
        kind_ordinals: dict[str, int] = {}
        for value in sorted(self.nodes.values(), key=lambda item: item["id"]):
            item = dict(value)
            if isinstance(item.get("sources"), set):
                item["sources"] = sorted(item["sources"])
            item["display_kind"] = _DISPLAY_KINDS.get(
                str(item.get("kind")), str(item.get("kind"))
            )
            item["display_label"] = _display_label(item)
            item["ui_tier"] = (
                "technical" if str(item.get("kind")) in _TECHNICAL_NODE_KINDS else "primary"
            )
            if item["display_label"] == item["display_kind"]:
                kind = str(item.get("kind"))
                kind_ordinals[kind] = kind_ordinals.get(kind, 0) + 1
                item["display_label"] = f"{item['display_label']} #{kind_ordinals[kind]}"
            nodes.append(item)
        edges = []
        for value in sorted(self.edges.values(), key=lambda item: item["id"]):
            item = dict(value)
            if isinstance(item.get("evidence"), set):
                item["evidence"] = sorted(item["evidence"])
            edges.append(item)
        kind_counts = Counter(str(item["kind"]) for item in nodes)
        tier_counts = Counter(str(item["ui_tier"]) for item in nodes)
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
            "node_tier_counts": dict(sorted(tier_counts.items())),
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
    include_payload: Callable[[Mapping[str, Any], str, int], bool] | None = None,
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
            if payload.get("artifact_type") in _PRESENTATION_PROJECTION_TYPES:
                continue
            if include_payload is not None and not include_payload(payload, str(path), index):
                continue
            source_count += 1
            _project_document(graph, payload, source=str(path), ordinal=index)
    if archive_db is not None:
        source_count += _project_archive(graph, Path(archive_db).expanduser().resolve())
    return graph.document(input_root=root, source_count=source_count)


def load_evidence_graph(input_root: Path) -> dict[str, Any] | None:
    """Load a previously materialized graph and add the current UI projection.

    Older runs often retain the immutable ``graph.json`` but not every raw
    receipt. Decorating that graph keeps the historical evidence usable in the
    workbench without rewriting any source artifact.
    """
    root = Path(input_root).expanduser().resolve()
    paths = [root / "graph.json"]
    for pattern in ("*/graph.json", "*/*/graph.json", "*/*/*/graph.json"):
        paths.extend(sorted(root.glob(pattern)))
    paths = [path for index, path in enumerate(paths) if path not in paths[:index]]
    paths = [path for path in paths if path.is_file() and not path.is_symlink()]
    if not paths:
        return None
    documents: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and payload.get("artifact_type") == "verdiwm-evidence-graph":
            documents.append(payload)
    if not documents:
        return None
    payload = documents[0]
    raw_nodes = [node for document in documents for node in (document.get("nodes") or [])]
    raw_edges = [edge for document in documents for edge in (document.get("edges") or [])]
    raw_nodes = list({str(node.get("id")): node for node in raw_nodes if isinstance(node, Mapping)}.values())
    raw_edges = list({str(edge.get("id")): edge for edge in raw_edges if isinstance(edge, Mapping)}.values())
    nodes: list[dict[str, Any]] = []
    kind_ordinals: dict[str, int] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            continue
        node = dict(raw)
        node["display_kind"] = _DISPLAY_KINDS.get(str(node.get("kind")), str(node.get("kind")))
        node["display_label"] = _display_label(node)
        node["ui_tier"] = "technical" if str(node.get("kind")) in _TECHNICAL_NODE_KINDS else "primary"
        if node["display_label"] == node["display_kind"]:
            kind = str(node.get("kind"))
            kind_ordinals[kind] = kind_ordinals.get(kind, 0) + 1
            node["display_label"] = f"{node['display_label']} #{kind_ordinals[kind]}"
        nodes.append(node)
    result = dict(payload)
    result["nodes"] = nodes
    result["node_count"] = len(nodes)
    result["edge_count"] = len(raw_edges)
    result["node_kind_counts"] = dict(sorted(Counter(str(node.get("kind")) for node in nodes).items()))
    result["node_tier_counts"] = dict(sorted(Counter(str(node["ui_tier"]) for node in nodes).items()))
    result["input_root"] = str(root)
    result["presentation_source"] = "materialized_graph"
    return result


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
    root = graph.node(
        "artifact",
        f"{artifact}:{identity}:{ordinal}",
        source=source,
        artifact_type=artifact,
        state=payload.get("state"),
        status=payload.get("status"),
        verdict=payload.get("verdict"),
        outcome=payload.get("outcome"),
        stage=payload.get("stage"),
        settlement_state=payload.get("settlement_state"),
        model_family=payload.get("model_family"),
        model_name=payload.get("model_name"),
        summary=payload.get("summary"),
        claim_boundary=payload.get("claim_boundary"),
    )
    kind_map = {
        "target_backbone": "backbone",
        "model_family": "backbone",
        "model_name": "model",
        "model_ref": "model",
        "environment": "environment",
        "env": "environment",
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
            "model_name": "names_model",
            "model_ref": "references_model",
            "environment": "evaluated_in",
            "env": "evaluated_in",
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
    # Give content-addressed model nodes a human-meaningful family attribute
    # whenever the same artifact names its backbone family.
    model_ref = payload.get("model_ref")
    if isinstance(model_ref, str) and model_ref:
        family = payload.get("model_family") or payload.get("model_name") or payload.get("target_backbone")
        if isinstance(family, str) and family:
            graph.node("model", model_ref, source=source, family=family)
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
