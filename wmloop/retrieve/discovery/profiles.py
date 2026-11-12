"""Evidence-bound mechanism discovery components: profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.retrieve.discovery.common import _load_mapping, _valid_tag
from wmloop.retrieve.discovery.types import _AXES, MechanismDiscoveryError


def _load_profiles(path: Path) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_PROFILES_INVALID")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
    result: dict[str, Mapping[str, Sequence[str]]] = {}
    for name, profile in profiles.items():
        if (
            not isinstance(name, str)
            or not isinstance(profile, Mapping)
            or set(profile) != set(_AXES)
        ):
            raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
        for values in profile.values():
            if not isinstance(values, list) or any(
                not isinstance(tag, str) or not _valid_tag(tag) for tag in values
            ):
                raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
        result[name] = profile  # type: ignore[assignment]
    return result


def _load_ontology(path: Path) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_ONTOLOGY_INVALID")
    if payload.get("artifact_type") != "wmloop-mechanism-tag-ontology":
        raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
    axes = payload.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != set(_AXES):
        raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
    result: dict[str, Mapping[str, Sequence[str]]] = {}
    for axis, concepts in axes.items():
        if not isinstance(concepts, Mapping) or not concepts:
            raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
        seen: set[str] = set()
        typed: dict[str, Sequence[str]] = {}
        for concept, aliases in concepts.items():
            if (
                not isinstance(concept, str)
                or not _valid_tag(concept)
                or not isinstance(aliases, list)
                or not aliases
                or any(
                    not isinstance(alias, str) or not _valid_tag(alias)
                    for alias in aliases
                )
            ):
                raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
            duplicate = seen & set(aliases)
            if duplicate:
                raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_ALIAS_DUPLICATE")
            seen.update(aliases)
            typed[concept] = tuple(aliases)
        result[str(axis)] = typed
    return result


def _load_reference_atlas_profiles(
    path: Path,
) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_REFERENCE_ATLAS_INVALID")
    entries = payload.get("entries")
    if (
        payload.get("artifact_type") != "wmloop-mechanism-discovery-atlas"
        or payload.get("state") != "ready"
        or not isinstance(entries, list)
    ):
        raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
    profiles: dict[str, Mapping[str, Sequence[str]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        mechanism = entry.get("mechanism")
        source = entry.get("source")
        if not isinstance(mechanism, Mapping) or not isinstance(source, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        if mechanism.get("extraction_state") != "evidence_supported":
            continue
        axes = _validated_profile_axes(
            mechanism.get("axes"),
            code="MECHANISM_REFERENCE_ATLAS_INVALID",
        )
        arxiv_id = str(source.get("arxiv_id") or "")
        if not arxiv_id:
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        profiles[f"literature:{arxiv_id}"] = axes
    return profiles


def _load_reference_profiles(
    path: Path,
) -> tuple[dict[str, Mapping[str, Sequence[str]]], dict[str, str]]:
    payload = _load_mapping(path, "MECHANISM_REFERENCE_PROFILES_INVALID")
    rows = payload.get("profiles")
    if (
        payload.get("artifact_type") != "wmloop-mechanism-reference-profiles"
        or not isinstance(rows, Mapping)
        or not rows
    ):
        raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
    profiles: dict[str, Mapping[str, Sequence[str]]] = {}
    kinds: dict[str, str] = {}
    for name, row in rows.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
        kind = row.get("reference_kind")
        if not isinstance(kind, str) or not kind:
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
        reference_id = f"settled:{name}"
        profiles[reference_id] = _validated_profile_axes(
            row.get("axes"),
            code="MECHANISM_REFERENCE_PROFILES_INVALID",
        )
        kinds[reference_id] = kind
    return profiles, kinds


def _validated_profile_axes(
    value: object,
    *,
    code: str,
) -> Mapping[str, Sequence[str]]:
    if not isinstance(value, Mapping) or set(value) != set(_AXES):
        raise MechanismDiscoveryError(code)
    result: dict[str, Sequence[str]] = {}
    for axis, tags in value.items():
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not _valid_tag(tag) for tag in tags
        ):
            raise MechanismDiscoveryError(code)
        result[str(axis)] = tuple(tags)
    return result


def _merge_profiles(
    destination: dict[str, Mapping[str, Sequence[str]]],
    incoming: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    source: Path,
) -> None:
    duplicate = set(destination) & set(incoming)
    if duplicate:
        names = ",".join(sorted(duplicate))
        raise MechanismDiscoveryError(f"MECHANISM_REFERENCE_DUPLICATE:{source}:{names}")
    destination.update(incoming)
