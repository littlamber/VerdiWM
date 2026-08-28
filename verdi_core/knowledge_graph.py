"""Layered, portable knowledge-graph projections and static viewer export.

The SQLite tables are the local query projection.  Append-only records and
content-addressed artifact references remain the portable interchange surface;
the graph document can therefore be rebuilt without trusting a UI or a path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import re
import tempfile
import html as html_lib
from pathlib import Path
from typing import Any, Mapping, Sequence


GRAPH_SCHEMA_VERSION = 1
LAYERS = {
    "L0": "ontology",
    "L1": "model_portrait",
    "L2": "method_knowledge",
    "L3": "experiment_evidence",
    "L4": "transfer_reasoning",
    "L5": "provenance",
}

PUBLIC_PROBE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "probe_id": "action_scaling",
        "semantic_variable": "action_conditioning_magnitude",
        "diagnostic_dimensions": ("action_responsiveness", "controllability_stability"),
        "dose_unit": "relative_action_gain",
    },
    {
        "probe_id": "controlled_context_retention",
        "semantic_variable": "retained_history_fraction",
        "diagnostic_dimensions": ("history_dependence", "forgetting_onset", "context_loss_long_horizon_drift"),
        "dose_unit": "retained_history_fraction_delta",
    },
    {
        "probe_id": "first_frame_anchoring_strength",
        "semantic_variable": "first_frame_conditioning_gain",
        "diagnostic_dimensions": ("identity_drift", "background_drift", "scene_layout_drift"),
        "dose_unit": "relative_anchor_gain",
    },
    {
        "probe_id": "sampler_noise_stress",
        "semantic_variable": "sampler_noise_strength_or_schedule",
        "diagnostic_dimensions": ("sampling_robustness", "model_sampler_instability_attribution"),
        "dose_unit": "relative_sampler_noise_stress",
    },
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _portable(value: Any) -> Any:
    """Remove machine-local paths while retaining deterministic references."""
    if isinstance(value, Mapping):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, tuple):
        return [_portable(v) for v in value]
    if isinstance(value, str):
        def artifact_ref(raw: str) -> str:
            path = Path(raw)
            digest = _sha256(path.read_bytes()) if path.is_file() and not path.is_symlink() else _sha256(raw.encode("utf-8"))
            return f"artifact://sha256:{digest}"

        def replace_path(match: re.Match[str]) -> str:
            return artifact_ref(match.group(0))

        if value.startswith("/"):
            return artifact_ref(value)
        return re.sub(r"/(?:share|root|home|tmp|mnt|workspace)/[^\s,;]+", replace_path, value)
    return value


def _portable_key(value: str) -> str:
    if "/" in value or "\\" in value:
        return "key://sha256:" + _sha256(value.encode("utf-8"))
    return value


def _row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("payload_json", "{}")))
    except (TypeError, json.JSONDecodeError):
        return {}


_KIND_NAMES = {
    "model": "模型",
    "architecture": "架构",
    "portrait": "模型画像",
    "fingerprint": "模型指纹",
    "probe": "探针观测",
    "probe_family": "公共探针",
    "method": "方法",
    "diagnosis": "诊断问题",
    "conclusion": "结论",
    "evidence": "验证证据",
    "metric": "评测指标",
    "metric_plan": "指标方案",
    "evaluator": "评测器",
    "source": "来源",
    "experiment": "实验",
    "run": "运行",
    "evaluation": "评估",
    "artifact": "产物",
    "transfer_assessment": "迁移评估",
}

_KNOWN_PUBLIC_NAMES = {
    "background-preservation-training": "背景保持训练",
    "denoising-path-agreement": "去噪路径一致性",
    "long-horizon-context-replay": "长时上下文回放",
    "geometry-aware-layout-consistency": "几何布局一致性",
    "schedule-robust-diffusion-training": "调度稳健扩散训练",
    "context-retention-training": "上下文保持训练",
}


def public_node_name(kind: str, key: str, payload: Mapping[str, Any] | None = None) -> str:
    """Return a stable, human-readable community label.

    Internal batch prefixes and replication suffixes remain in ``key`` and
    provenance payloads, but are intentionally absent from the primary label.
    """
    data = payload or {}
    for field in ("display_name", "title", "name", "method_name"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    clean = str(key)
    if clean in _KNOWN_PUBLIC_NAMES:
        return _KNOWN_PUBLIC_NAMES[clean]
    if kind in {"evidence", "experiment", "model", "run", "artifact"} and re.match(r"^(?:evidence|experiment|model|run|artifact)[-:]", clean, re.I):
        return _KIND_NAMES.get(kind, kind)
    clean = re.sub(r"^(experiments-v\d+-|experiment-v\d+-)", "", clean)
    clean = re.sub(r"-(?:heldout|rep\\d+|independent-model|100-seedA)(?:-.+)?$", "", clean)
    clean = clean.replace("_", " ").replace("-", " ").strip()
    if not clean:
        return _KIND_NAMES.get(kind, kind)
    return clean[:1].upper() + clean[1:]


# These fields identify a run or reproduction rather than shared knowledge.
# Keep them available to auditors, but out of the primary community payload.
_PROVENANCE_FIELDS = {
    "revision", "version", "version_id", "variant_of", "seed", "seed_id",
    "experiment_id", "evidence_id", "run_id", "campaign_id", "replication",
    "record_digest", "verifier_digest", "artifact_digest", "artifact_ref",
    "evidence_ref", "split", "source", "created_at", "attempt", "batch_id",
}


def _public_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Separate stable public facts from run-specific provenance metadata."""
    public: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key, value in payload.items():
        (provenance if str(key) in _PROVENANCE_FIELDS else public)[str(key)] = value
    if provenance:
        public["provenance"] = provenance
    return public


def _legacy_graph_projection(state: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project pre-layered evidence tables into the public graph vocabulary.

    Older campaigns persisted only ``evidence`` and ``knowledge_edges``.  This
    read-only projection lets those campaigns participate in community exports
    without mutating their historical SQLite state.
    """
    try:
        evidence_rows = state.list_rows("evidence", limit=100000)
        legacy_edges = state.list_rows("knowledge_edges", limit=100000)
    except (AttributeError, ValueError):
        return [], []
    if not evidence_rows:
        return [], []

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(kind: str, key: str, *, layer: str, status: str | None = None, payload: Mapping[str, Any] | None = None) -> str:
        node_id = _node_id(kind, key)
        body = dict(payload or {})
        nodes.setdefault(node_id, {
            "id": node_id,
            "kind": kind,
            "key": key,
            "display_name": public_node_name(kind, key, body),
            "layer": layer,
            "status": status,
            "content_digest": "sha256:" + _sha256(json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "payload": body,
        })
        return node_id

    def add_edge(source: str, relation: str, target: str, *, evidence_id: str | None = None, payload: Mapping[str, Any] | None = None) -> None:
        identity = {"source": source, "relation": relation, "target": target, "evidence": evidence_id}
        edge_id = "edge-" + _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))[:16]
        edges.setdefault(edge_id, {
            "id": edge_id,
            "source": source,
            "relation": relation,
            "target": target,
            "evidence_id": evidence_id,
            "content_digest": "sha256:" + _sha256(json.dumps({**identity, "payload": dict(payload or {})}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "payload": dict(payload or {}),
        })

    for row in evidence_rows:
        payload = _row_payload(row)
        evidence_id = str(row.get("evidence_id") or payload.get("evidence_id") or "unknown-evidence")
        model_id = str(row.get("model_id") or payload.get("model_id") or "unknown-model")
        hypothesis_id = str(payload.get("hypothesis_id") or row.get("experiment_id") or evidence_id)
        experiment_id = str(row.get("experiment_id") or payload.get("experiment_id") or evidence_id)
        outcome = str(row.get("outcome") or payload.get("outcome") or "unknown")
        model = add_node("model", model_id, layer="L1", payload={"model_id": model_id})
        diagnosis = add_node("diagnosis", hypothesis_id, layer="L2", status="observed", payload={
            "display_name": "诊断问题：" + public_node_name("method", hypothesis_id),
            "problem": hypothesis_id,
        })
        method = add_node("method", hypothesis_id, layer="L2", status=outcome, payload={
            "display_name": "方法：" + public_node_name("method", hypothesis_id),
            "method_name": hypothesis_id,
            "claim": payload.get("claim", ""),
        })
        conclusion_text = {"confirmed_positive": "支持", "positive": "支持", "null": "未观察到改善", "abstain": "暂不下结论"}.get(outcome, outcome)
        conclusion = add_node("conclusion", hypothesis_id + ":" + outcome, layer="L3", status=outcome, payload={
            "display_name": "结论：" + conclusion_text,
            "outcome": outcome,
            "delta": row.get("delta", payload.get("delta", 0.0)),
            "protected_ok": bool(row.get("protected_ok", payload.get("protected_ok", False))),
            "claim_boundary": payload.get("claim_boundary", ""),
        })
        evidence = add_node("evidence", evidence_id, layer="L3", status=outcome, payload=_public_payload({
            "outcome": outcome,
            "delta": row.get("delta", payload.get("delta", 0.0)),
            "protected_ok": bool(row.get("protected_ok", payload.get("protected_ok", False))),
            **payload,
        }))
        experiment = add_node("experiment", experiment_id, layer="L5", status=outcome, payload={
            # The experiment id is provenance; expose the underlying question
            # as the community label instead of a dated/versioned batch name.
            "display_name": "实验：" + public_node_name("method", hypothesis_id),
            "provenance": {"experiment_id": experiment_id},
        })
        add_edge(diagnosis, "addressed_by", method)
        add_edge(method, "led_to", conclusion)
        add_edge(conclusion, "supported_by", evidence, evidence_id=evidence_id)
        add_edge(method, "validated_by", experiment)
        add_edge(experiment, "produced", evidence, evidence_id=evidence_id)
        add_edge(method, "tested_on", model)

    # Preserve predicates from the old edge table as provenance links where an
    # endpoint was not already represented by the semantic projection.
    for row in legacy_edges:
        subject = str(row.get("subject", "")); object_ = str(row.get("object", "")); predicate = str(row.get("predicate", "related_to"))
        evidence_id = row.get("evidence_id")
        source = _node_id("experiment", subject)
        target = _node_id("evidence", object_)
        if source in nodes and target in nodes:
            add_edge(source, predicate, target, evidence_id=str(evidence_id) if evidence_id else None)
    return list(nodes.values()), list(edges.values())


def build_graph_document(state: Any, *, portable: bool = False) -> dict[str, Any]:
    """Build a deterministic nodes/edges document from a SQLiteState."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for row in state.graph_nodes(limit=100000):
        item = {
            "id": row["node_id"],
            "kind": row["kind"],
            "key": _portable_key(str(row["node_key"])) if portable else row["node_key"],
            "display_name": public_node_name(str(row["kind"]), str(row["node_key"]), _row_payload(row)),
            "layer": row["layer"],
            "status": row.get("status"),
            "content_digest": row["content_digest"],
            "payload": _public_payload(_row_payload(row)),
        }
        nodes.append(_portable(item) if portable else item)
    for row in state.graph_edges(limit=100000):
        item = {
            "id": row["edge_id"],
            "source": row["source_id"],
            "relation": row["relation"],
            "target": row["target_id"],
            "evidence_id": row.get("evidence_id"),
            "content_digest": row["content_digest"],
            "payload": _row_payload(row),
        }
        edges.append(_portable(item) if portable else item)
    # Campaigns created before the layered graph schema have no explicit graph
    # rows.  Add their semantic projection while preserving any newer rows.
    legacy_nodes, legacy_edges = _legacy_graph_projection(state)
    known_nodes = {item["id"] for item in nodes}
    known_edges = {item["id"] for item in edges}
    nodes.extend(item for item in legacy_nodes if item["id"] not in known_nodes)
    edges.extend(item for item in legacy_edges if item["id"] not in known_edges)
    if portable:
        nodes = [_portable(item) for item in nodes]
        edges = [_portable(item) for item in edges]
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "artifact_type": "verdiwm-layered-knowledge-graph",
        "layers": LAYERS,
        "portable": portable,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(edges, key=lambda item: item["id"]),
    }


def build_transfer_index(state: Any, *, portable: bool = True) -> dict[str, Any]:
    """Emit a compact method/portrait index used by the migration router."""
    selected = []
    for row in state.graph_nodes(kind="method", limit=100000):
        payload = _row_payload(row)
        selected.append(
            {
                "method_id": row["node_id"],
                "key": row["node_key"],
                "display_name": public_node_name("method", str(row["node_key"]), payload),
                "status": row.get("status"),
                "mechanism": payload.get("mechanism", payload.get("claim", "")),
                "architecture_facets": payload.get("architecture_facets", []),
                "diagnostic_dimensions": payload.get("diagnostic_dimensions", payload.get("diagnoses", [])),
                "required_capabilities": payload.get("required_capabilities", []),
                "anti_conditions": payload.get("anti_conditions", []),
                "evidence_count": payload.get("evidence_count", payload.get("replications", 0)),
            }
        )
    result = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "artifact_type": "verdiwm-transfer-index",
        "purpose": "ranking-only migration retrieval; target-side validation remains mandatory",
        "methods": sorted(selected, key=lambda item: item["method_id"]),
    }
    return _portable(result) if portable else result


def _node_id(kind: str, key: str) -> str:
    return f"{kind}:{_sha256((kind + ':' + key).encode('utf-8'))[:16]}"


def project_model_portrait(
    state: Any,
    *,
    model_id: str,
    revision: str,
    capabilities: Sequence[str] = (),
    hooks: Sequence[str] = (),
    architecture_facets: Sequence[str] = (),
    probe_results: Sequence[Mapping[str, Any]] = (),
    portrait_id: str | None = None,
) -> str:
    """Persist a model, its probe observations and derived portrait as graph nodes."""
    model_node = _node_id("model", model_id)
    state.put_graph_node(model_node, kind="model", key=model_id, layer="L1", payload={
        "model_id": model_id, "revision": revision, "capabilities": sorted(set(map(str, capabilities))),
        "hooks": sorted(set(map(str, hooks))), "architecture_facets": sorted(set(map(str, architecture_facets))),
    })
    state.put_graph_node(_node_id("architecture", model_id), kind="architecture", key=model_id, layer="L1", payload={"facets": sorted(set(map(str, architecture_facets)))})
    state.add_graph_edge(model_node, "has_architecture", _node_id("architecture", model_id))
    diagnostics: set[str] = set()
    fingerprint_ids: list[str] = []
    for result in probe_results:
        probe_id = str(result.get("probe_id", "unknown"))
        probe_node = _node_id("probe", probe_id)
        state.put_graph_node(probe_node, kind="probe", key=probe_id, layer="L0", status=str(result.get("status", "unknown")), payload=dict(result))
        state.add_graph_edge(model_node, "evaluated_probe", probe_node)
        dims = result.get("dimensions") or result.get("diagnostic_dimensions") or result.get("failure_signatures") or []
        diagnostics.update(map(str, dims if isinstance(dims, Sequence) and not isinstance(dims, str) else [dims]))
        fingerprint_key = model_id + ":" + probe_id + ":" + str(result.get("response_digest", result.get("result", "")))
        fingerprint_node = _node_id("fingerprint", fingerprint_key)
        fingerprint_ids.append(fingerprint_node)
        state.put_graph_node(fingerprint_node, kind="fingerprint", key=fingerprint_key, layer="L1", payload={"model_id": model_id, "probe_id": probe_id, "dimensions": list(map(str, dims)) if isinstance(dims, Sequence) and not isinstance(dims, str) else [str(dims)], "observation": dict(result)})
        state.add_graph_edge(fingerprint_node, "generated_by", probe_node)
        state.add_graph_edge(model_node, "has_fingerprint", fingerprint_node)
    portrait_key = portrait_id or model_id + ":" + revision
    portrait_node = _node_id("portrait", portrait_key)
    state.put_graph_node(portrait_node, kind="portrait", key=portrait_key, layer="L1", payload={"model_id": model_id, "revision": revision, "architecture_facets": sorted(set(map(str, architecture_facets))), "capabilities": sorted(set(map(str, capabilities))), "hooks": sorted(set(map(str, hooks))), "diagnostic_dimensions": sorted(diagnostics), "fingerprint_ids": fingerprint_ids, "readiness": "ready_for_experiment"})
    state.add_graph_edge(model_node, "has_portrait", portrait_node)
    return portrait_node


def project_metric_plan(state: Any, *, model_id: str, plan: Mapping[str, Any], source: str = "benchmark_selector") -> str:
    """Record metric eligibility, selection, and evaluator provenance in L0-L5."""
    model_node = _node_id("model", model_id)
    plan_key = model_id + ":" + str(plan.get("catalog_digest", "")) + ":" + str(plan.get("primary", ""))
    plan_node = _node_id("metric_plan", plan_key)
    state.put_graph_node(plan_node, kind="metric_plan", key=plan_key, layer="L4", status=str(plan.get("state", "unknown")), payload={"model_id": model_id, "source": source, **dict(plan)})
    state.add_graph_edge(model_node, "selected_metric_plan", plan_node)
    for definition in plan.get("definitions", ()):
        if not isinstance(definition, Mapping) or not definition.get("metric_id"):
            continue
        metric_id = str(definition["metric_id"])
        metric_node = _node_id("metric", metric_id)
        state.put_graph_node(metric_node, kind="metric", key=metric_id, layer="L0", status=str(definition.get("implementation_status", "catalogued")), payload=dict(definition))
        state.add_graph_edge(plan_node, "uses_metric", metric_node)
        if definition.get("evaluator_ref"):
            evaluator_key = str(definition["evaluator_ref"])
            evaluator_node = _node_id("evaluator", evaluator_key)
            state.put_graph_node(evaluator_node, kind="evaluator", key=evaluator_key, layer="L5", status="declared", payload={"evaluator_ref": evaluator_key, "metric_id": metric_id})
            state.add_graph_edge(metric_node, "evaluated_by", evaluator_node)
    return plan_node


def import_settlement_entries(state: Any, entries: Sequence[Mapping[str, Any]], *, model_id: str = "ctrl-world", campaign_id: str = "ctrl-world") -> dict[str, int]:
    """Import compact settlement entries without leaking local paths in exports."""
    counts = {"methods": 0, "evidence": 0, "edges": 0}
    model_node = _node_id("model", model_id)
    state.put_graph_node(model_node, kind="model", key=model_id, layer="L1", payload={"model_id": model_id, "campaign_id": campaign_id})
    for entry in entries:
        idea_id = str(entry.get("idea_id") or entry.get("method_id") or "unknown")
        method_node = _node_id("method", idea_id)
        payload = dict(entry)
        payload.setdefault("campaign_id", campaign_id)
        payload.setdefault("evidence_count", int(entry.get("replications", 0) or 0))
        state.append_knowledge_record(payload, record_type="settlement", layer="L3", status=str(entry.get("status", "unknown")))
        state.put_graph_node(method_node, kind="method", key=idea_id, layer="L2", status=str(entry.get("status", "unknown")), payload=payload)
        state.add_graph_edge(method_node, "evaluated_on", model_node)
        counts["methods"] += 1; counts["edges"] += 1
        paths = entry.get("evidence_paths", [])
        if not isinstance(paths, Sequence) or isinstance(paths, str):
            paths = []
        replications = int(entry.get("replications", len(paths)) or 0)
        for index in range(replications or len(paths)):
            ref = str(paths[index]) if index < len(paths) else f"replication:{index + 1}"
            # Keep local paths out of node identities as well as payloads.
            evidence_key = idea_id + ":replication:" + str(index + 1) + ":" + _sha256(ref.encode("utf-8"))[:16]
            evidence_node = _node_id("evidence", evidence_key)
            delta = entry.get("replicate_deltas", {})
            state.put_graph_node(evidence_node, kind="evidence", key=evidence_key, layer="L3", status=str(entry.get("status", "unknown")), payload={"idea_id": idea_id, "replication": index + 1, "evidence_ref": ref, "delta": delta, "claim": entry.get("claim", ""), "boundary": entry.get("boundary", "")})
            state.add_graph_edge(method_node, "supported_by", evidence_node)
            counts["evidence"] += 1; counts["edges"] += 1
    return counts


def seed_public_ontology(state: Any) -> int:
    """Install the stable layer-0 probe vocabulary into a graph projection."""
    count = 0
    for item in PUBLIC_PROBE_FAMILIES:
        probe_id = str(item["probe_id"])
        node_id = _node_id("probe_family", probe_id)
        state.put_graph_node(node_id, kind="probe_family", key=probe_id, layer="L0", status="public", payload={
            "probe_id": probe_id,
            "semantic_variable": item["semantic_variable"],
            "diagnostic_dimensions": list(item["diagnostic_dimensions"]),
            "dose_unit": item["dose_unit"],
            "inference_only": True,
            "reversible": True,
            "support_policy": "adapter_declares_supported_or_unsupported; AI may materialize hooks",
        })
        count += 1
    return count


def write_static_viewer(graph: Mapping[str, Any], output: Path, *, title: str = "VerdiWM Knowledge Graph") -> Path:
    """Write a dependency-free interactive graph viewer."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(graph, ensure_ascii=True, separators=(",", ":"))
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_lib.escape(title)}</title><style>
html,body{{margin:0;height:100%;font:14px system-ui,sans-serif;background:#111827;color:#e5e7eb;overflow:hidden}}
#bar{{height:48px;display:flex;gap:8px;align-items:center;padding:0 12px;background:#1f2937;box-sizing:border-box}}
input,select,button{{background:#374151;color:#f9fafb;border:1px solid #4b5563;border-radius:4px;padding:6px 8px}}
#q{{width:260px}}#count{{margin-left:auto;color:#9ca3af}}#stage{{position:relative;height:calc(100% - 48px)}}
svg{{width:100%;height:100%;cursor:grab}}svg.dragging{{cursor:grabbing}}.edge{{stroke:#6b7280;stroke-width:1.2;opacity:.65}}
.node{{cursor:grab;stroke:#111827;stroke-width:1.5}}.node.selected{{stroke:#fbbf24;stroke-width:3}}
.label{{pointer-events:none;fill:#f3f4f6;font-size:11px;text-anchor:middle}}#info{{position:absolute;right:12px;top:12px;width:300px;max-height:calc(100% - 24px);overflow:auto;background:#1f2937;border:1px solid #4b5563;padding:12px;display:none;white-space:pre-wrap}}
</style></head><body><div id="bar"><input id="q" placeholder="Search nodes"><select id="layer"><option value="">All layers</option></select><select id="kind"><option value="">All kinds</option></select><button id="reset">Reset</button><span id="count"></span></div><div id="stage"><svg id="svg"></svg><pre id="info"></pre></div>
<script>const G={data};const svg=document.getElementById('svg'),ns='http://www.w3.org/2000/svg';const q=document.getElementById('q'),layer=document.getElementById('layer'),kind=document.getElementById('kind'),info=document.getElementById('info'),count=document.getElementById('count');
const layers=Object.keys(G.layers||{{}});layers.forEach(x=>layer.add(new Option(x+' '+G.layers[x],x)));[...new Set(G.nodes.map(n=>n.kind))].sort().forEach(x=>kind.add(new Option(x,x)));
let scale=1,pan={{x:0,y:0}},drag=null,selected=null;const colors={{ontology:'#60a5fa',model:'#34d399',probe:'#fbbf24',fingerprint:'#f59e0b',portrait:'#a78bfa',method:'#f472b6',source:'#fb7185',experiment:'#22d3ee',run:'#2dd4bf',evaluation:'#c084fc',evidence:'#f87171',artifact:'#9ca3af',transfer_assessment:'#fde047'}};
function visible(n){{const text=(n.key+' '+JSON.stringify(n.payload)).toLowerCase();return(!q.value||text.includes(q.value.toLowerCase()))&&(!layer.value||n.layer===layer.value)&&(!kind.value||n.kind===kind.value)}}
function draw(){{svg.replaceChildren();const W=svg.clientWidth||1000,H=svg.clientHeight||700;const nsn=G.nodes.filter(visible),ids=new Set(nsn.map(n=>n.id));const pos=new Map(nsn.map((n,i)=>[n.id,{{x:W/2+(i%8-3.5)*120,y:H/2+(Math.floor(i/8)-Math.ceil(nsn.length/16))*100}}]));
const g=document.createElementNS(ns,'g');g.setAttribute('transform',`translate(${{pan.x}},${{pan.y}}) scale(${{scale}})`);svg.append(g);G.edges.filter(e=>ids.has(e.source)&&ids.has(e.target)).forEach(e=>{{const a=pos.get(e.source),b=pos.get(e.target),line=document.createElementNS(ns,'line');line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.setAttribute('class','edge');g.append(line)}});
nsn.forEach(n=>{{const p=pos.get(n.id),c=document.createElementNS(ns,'circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',n.kind==='method'?18:13);c.setAttribute('fill',colors[n.kind]||'#94a3b8');c.setAttribute('class','node'+(selected===n.id?' selected':''));c.onmousedown=ev=>{{ev.stopPropagation();drag={{n,p,dx:ev.clientX-p.x,dy:ev.clientY-p.y}}}};c.onclick=()=>{{selected=n.id;info.style.display='block';info.textContent=JSON.stringify(n,null,2);draw()}};g.append(c);const t=document.createElementNS(ns,'text');t.setAttribute('x',p.x);t.setAttribute('y',p.y+32);t.setAttribute('class','label');t.textContent=(n.display_name||n.key).length>22?(n.display_name||n.key).slice(0,21)+'…':(n.display_name||n.key);g.append(t)}});count.textContent=nsn.length+' nodes / '+G.edges.filter(e=>ids.has(e.source)&&ids.has(e.target)).length+' edges'}}
svg.onmousemove=ev=>{{if(!drag)return;drag.p.x=ev.clientX-drag.dx;drag.p.y=ev.clientY-drag.dy;draw()}};svg.onmouseup=()=>drag=null;svg.onwheel=ev=>{{ev.preventDefault();scale=Math.max(.25,Math.min(3,scale*(ev.deltaY<0?1.1:.9)));draw()}};[q,layer,kind].forEach(x=>x.oninput=draw);document.getElementById('reset').onclick=()=>{{q.value='';layer.value='';kind.value='';scale=1;pan={{x:0,y:0}};selected=null;info.style.display='none';draw()}};draw();</script></body></html>'''
    destination.write_text(html, encoding="utf-8")
    return destination


def export_bundle(state: Any, output_root: Path, *, portable: bool = True, include_sqlite: bool = True) -> dict[str, Any]:
    """Export the community bundle from one local state projection."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    graph = build_graph_document(state, portable=portable)
    transfer = build_transfer_index(state, portable=portable)
    graph_path = root / "graph.json"
    transfer_path = root / "transfer_index.json"
    graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transfer_path.write_text(json.dumps(transfer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_static_viewer(graph, root / "graph.html")
    records = state.list_rows("knowledge_records", limit=100000)
    if not records:
        # Legacy campaigns predate knowledge_records; retain their evidence in
        # the portable record stream instead of silently exporting an empty one.
        records = [
            {
                "record_id": row.get("evidence_id"),
                "record_type": "legacy_evidence",
                "layer": "L3",
                "status": row.get("outcome", "unknown"),
                "payload_json": row.get("payload_json", "{}"),
                "record_digest": "",
                "created_at": "",
            }
            for row in state.list_rows("evidence", limit=100000)
        ]
    (root / "records").mkdir(exist_ok=True)
    (root / "records" / "knowledge.jsonl").write_text(
        "".join(json.dumps(_portable({**row, "payload_json": None, "payload": _row_payload(row)}) if portable else {**row, "payload": _row_payload(row)}, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    if include_sqlite:
        destination = root / "knowledge.sqlite3"
        if portable:
            _write_portable_sqlite(state, destination, graph, records)
        else:
            source = sqlite3.connect(state.path)
            target = sqlite3.connect(destination)
            source.backup(target)
            target.close(); source.close()
    manifest = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "artifact_type": "verdiwm-knowledge-community-bundle",
        "portable": portable,
        "layers": LAYERS,
        "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
        "graph_sha256": _sha256(graph_path.read_bytes()),
        "transfer_index_sha256": _sha256(transfer_path.read_bytes()),
        "claim_boundary": "Transfer entries rank bounded target-side experiments; they never replace target validation.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_portable_sqlite(state: Any, destination: Path, graph: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    from .storage import SQLiteState

    handle = tempfile.NamedTemporaryFile(prefix="knowledge-", suffix=".sqlite3", dir=destination.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        target = SQLiteState(temporary)
        for node in graph.get("nodes", []):
            target.put_graph_node(str(node["id"]), kind=str(node["kind"]), key=str(node["key"]), layer=str(node["layer"]), status=node.get("status"), payload=dict(node.get("payload", {})))
        for edge in graph.get("edges", []):
            target.add_graph_edge(str(edge["source"]), str(edge["relation"]), str(edge["target"]), evidence_id=edge.get("evidence_id"), payload=dict(edge.get("payload", {})))
        for row in records:
            payload = _portable(_row_payload(row))
            target.append_knowledge_record(payload, record_type=str(row.get("record_type", "evidence")), layer=str(row.get("layer", "L3")), status=str(row.get("status", "unknown")))
        for row in state.list_rows("transfer_assessments", limit=100000):
            payload = _portable(_row_payload(row))
            target.put_transfer_assessment(str(row["assessment_id"]), source_method_id=str(row["source_method_id"]), target_model_id=str(row["target_model_id"]), state=str(row["state"]), score=float(row["score"]), payload=payload)
        target.close()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
