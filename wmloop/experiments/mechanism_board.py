"""Discover mechanism-relation artifacts and build a board projection for UIs.

Relations are first-class portable artifacts (``verdiwm-mechanism-relation``).
This module only reads and projects them; the source JSONL/JSON documents and
the EffectMemory settlement path remain authoritative.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from wmloop.geometry.mechanism_relations import validate_mechanism_relation

_MAX_SCAN_FILES = 5000
_MAX_RELATIONS = 256


class MechanismBoardError(ValueError):
    """The mechanism board input or discovery contract is invalid."""


def discover_mechanism_relations(root: Path) -> list[dict[str, Any]]:
    """Return validated relation documents found under root."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir() or base.is_symlink():
        raise MechanismBoardError("MECHANISM_BOARD_ROOT_INVALID")
    relations: dict[str, dict[str, Any]] = {}
    scanned = 0
    for path in sorted(base.rglob("*.json")) + sorted(base.rglob("*.jsonl")):
        if scanned >= _MAX_SCAN_FILES or len(relations) >= _MAX_RELATIONS:
            break
        if path.is_symlink() or not path.is_file():
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if '"verdiwm-mechanism-relation"' not in text:
            continue
        payloads: list[Any]
        if path.suffix == ".jsonl":
            try:
                payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError:
                continue
        else:
            try:
                payloads = [json.loads(text)]
            except json.JSONDecodeError:
                continue
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            if payload.get("artifact_type") != "verdiwm-mechanism-relation":
                continue
            try:
                validate_mechanism_relation(payload)
            except Exception:
                continue
            payload = dict(payload)
            payload["source_path"] = str(path)
            relations[str(payload["relation_id"])] = payload
    return [relations[key] for key in sorted(relations)]


def build_mechanism_board(root: Path) -> dict[str, Any]:
    """Build the mechanism-relation board consumed by the workbench UI."""

    base = Path(root).expanduser().resolve()
    relations = discover_mechanism_relations(base)
    if not relations:
        raise MechanismBoardError("MECHANISM_RELATIONS_NOT_FOUND")
    mechanisms: dict[str, dict[str, Any]] = {}
    for rel in relations:
        for role in ("source_mechanism_id", "target_mechanism_id"):
            mechanism_id = str(rel[role])
            entry = mechanisms.setdefault(
                mechanism_id,
                {"mechanism_id": mechanism_id, "relation_count": 0, "states": set()},
            )
            entry["relation_count"] += 1
            entry["states"].add(str(rel["verification_state"]))
    mechanism_rows = []
    for mechanism_id in sorted(mechanisms):
        entry = mechanisms[mechanism_id]
        mechanism_rows.append(
            {
                "mechanism_id": mechanism_id,
                "relation_count": entry["relation_count"],
                "states": sorted(entry["states"]),
            }
        )
    type_counts = Counter(str(rel["relation_type"]) for rel in relations)
    state_counts = Counter(str(rel["verification_state"]) for rel in relations)
    # Ranking-only queue of pairs whose four-cell evidence is still incomplete.
    # The score is a transparent heuristic (|interaction| / uncertainty), never
    # claim authority: admission still requires the settled four cells.
    queue = []
    for rel in relations:
        if str(rel["verification_state"]) not in {"candidate", "screened"}:
            continue
        uncertainty = max(float(rel["uncertainty"]), 1e-9)
        queue.append(
            {
                "relation_id": rel["relation_id"],
                "source_mechanism_id": rel["source_mechanism_id"],
                "target_mechanism_id": rel["target_mechanism_id"],
                "relation_type": rel["relation_type"],
                "verification_state": rel["verification_state"],
                "interaction_effect": rel["interaction_effect"],
                "uncertainty": rel["uncertainty"],
                "replication_count": rel["replication_count"],
                "missing_gates": sorted(
                    gate for gate, ok in dict(rel["validity_gates"]).items() if not ok
                ),
                "priority_score": abs(float(rel["interaction_effect"])) / uncertainty,
            }
        )
    queue.sort(
        key=lambda item: (-item["priority_score"], str(item["relation_id"]))
    )
    return {
        "artifact_type": "verdiwm-mechanism-board",
        "schema_version": 1,
        "state": "ready",
        "input_root": str(base),
        "relation_count": len(relations),
        "mechanism_count": len(mechanism_rows),
        "relation_type_counts": dict(sorted(type_counts.items())),
        "verification_state_counts": dict(sorted(state_counts.items())),
        "claim_boundary": (
            "Relations in candidate or screened state are ranking knowledge only. "
            "Only confirmed or rejected relations with all validity gates and named "
            "ablations carry claim authority. The discovery queue priority is a "
            "transparent heuristic and cannot promote a relation."
        ),
        "mechanisms": mechanism_rows,
        "relations": relations,
        "discovery_queue": queue,
    }
