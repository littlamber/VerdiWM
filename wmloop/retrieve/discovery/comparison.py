"""Evidence-bound mechanism discovery components: comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.retrieve.discovery.common import _tag_set
from wmloop.retrieve.discovery.types import _AXES


def compare_mechanism_signature(
    extraction: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    ontology: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    reference_kinds: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compare controlled mechanism concepts, never titles or method names."""

    axes = extraction.get("axes")
    if extraction.get("extraction_state") != "evidence_supported" or not isinstance(
        axes, Mapping
    ):
        return {
            "novelty_state": "unresolved",
            "nearest_primitives": [],
            "novel_axes": [],
            "novelty_basis": "Typed mechanism semantics are incomplete or not evidence-supported.",
        }
    candidate = {
        axis: _canonical_concepts(axis, _tag_set(axes.get(axis, [])), ontology)
        for axis in _AXES
    }
    if sum(bool(tags) for tags in candidate.values()) < 4:
        return {
            "novelty_state": "unresolved",
            "nearest_primitives": [],
            "novel_axes": [],
            "novelty_basis": "Fewer than four mechanism axes are populated.",
        }
    scored: list[dict[str, object]] = []
    known_tags: dict[str, set[str]] = {axis: set() for axis in _AXES}
    for primitive, profile in sorted(profiles.items()):
        overlaps: list[str] = []
        differences: list[str] = []
        matched_concepts: dict[str, list[str]] = {}
        intersection = 0
        union = 0
        for axis in _AXES:
            reference = _canonical_concepts(
                axis,
                _tag_set(profile.get(axis, [])),
                ontology,
            )
            known_tags[axis].update(reference)
            current = candidate[axis]
            shared = current & reference
            if shared:
                overlaps.append(axis)
                matched_concepts[axis] = sorted(shared)
                intersection += 1
            if current != reference:
                differences.append(axis)
            if current or reference:
                union += 1
        score = intersection / union if union else 0.0
        scored.append(
            {
                "primitive": primitive,
                "reference_kind": str(
                    (reference_kinds or {}).get(primitive, "registered_primitive")
                ),
                "structural_similarity": round(score, 6),
                "matching_axes": overlaps,
                "differing_axes": differences,
                "matched_concepts": matched_concepts,
            }
        )
    scored.sort(
        key=lambda row: (-float(row["structural_similarity"]), str(row["primitive"]))
    )
    nearest = scored[:3]
    best = float(nearest[0]["structural_similarity"]) if nearest else 0.0
    novel_axes = [
        axis
        for axis in _AXES
        if candidate[axis] and not candidate[axis].issubset(known_tags[axis])
    ]
    if best >= 0.85 and not novel_axes:
        state = "equivalent"
    elif best >= 0.45 and novel_axes:
        state = "extension"
    elif (
        len([row for row in nearest if float(row["structural_similarity"]) >= 0.25])
        >= 2
    ):
        state = "composition"
    else:
        state = "structural_candidate"
    return {
        "novelty_state": state,
        "nearest_primitives": nearest,
        "novel_axes": novel_axes,
        "novelty_basis": (
            "Classification uses ontology-canonicalized operator concepts, evidence-backed "
            "annotations, registered primitives, and supplied prior-mechanism references; "
            "it is a screening label, not a publication-level novelty claim."
        ),
    }


def _canonical_concepts(
    axis: str,
    tags: set[str],
    ontology: Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> set[str]:
    if ontology is None:
        return set(tags)
    concepts = ontology.get(axis, {})
    aliases = {
        str(alias): str(concept)
        for concept, values in concepts.items()
        for alias in values
    }
    return {aliases.get(tag, f"unmapped:{tag}") for tag in tags}
