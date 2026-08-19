"""Auditable world-model training recipes.

Public training numbers are useful as priors, but they are not authority for a
local run.  This module keeps provenance and disclosure status attached to each
recipe and refuses to turn ``shadow_only`` literature into launch parameters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class TrainingRecipeError(ValueError):
    """The recipe registry is invalid or not admitted for execution."""


ADMITTED_STATUSES = {"local_validated", "reusable_optimization_memory"}


def load_training_recipe_registry(
    path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    """Load and validate a checked-in registry without network access."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingRecipeError("TRAINING_RECIPE_REGISTRY_INVALID") from exc
    if not isinstance(payload, dict):
        raise TrainingRecipeError("TRAINING_RECIPE_REGISTRY_INVALID")
    try:
        validate_document("world_model_training_recipe_registry", payload, root=root)
    except ContractValidationError as exc:
        raise TrainingRecipeError(f"TRAINING_RECIPE_REGISTRY_SCHEMA_INVALID:{exc}") from exc

    source_ids = [row["source_id"] for row in payload["provenance"]["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise TrainingRecipeError("TRAINING_RECIPE_SOURCE_ID_DUPLICATE")
    known_sources = set(source_ids)
    recipe_ids: set[str] = set()
    for recipe in payload["recipes"]:
        recipe_id = recipe["recipe_id"]
        if recipe_id in recipe_ids:
            raise TrainingRecipeError("TRAINING_RECIPE_ID_DUPLICATE")
        recipe_ids.add(recipe_id)
        if not set(recipe["source_ids"]).issubset(known_sources):
            raise TrainingRecipeError(f"TRAINING_RECIPE_SOURCE_UNKNOWN:{recipe_id}")
    payload["registry_path"] = str(source)
    payload["registry_sha256"] = _sha256(source)
    return payload


def find_training_recipe(
    registry: Mapping[str, Any], recipe_id: str
) -> Mapping[str, Any]:
    for recipe in registry.get("recipes", []):
        if isinstance(recipe, Mapping) and recipe.get("recipe_id") == recipe_id:
            return recipe
    raise TrainingRecipeError(f"TRAINING_RECIPE_NOT_FOUND:{recipe_id}")


def summarize_training_recipes(
    registry: Mapping[str, Any], *, recipe_id: str | None = None
) -> dict[str, Any]:
    """Return a stable, compact index suitable for CLI and retrieval ranking."""

    recipes = registry.get("recipes", [])
    if recipe_id is not None:
        recipes = [find_training_recipe(registry, recipe_id)]
    items = []
    for recipe in recipes:
        items.append(
            {
                "recipe_id": recipe["recipe_id"],
                "backbone": recipe["backbone"],
                "family": recipe["family"],
                "phase": recipe["phase"],
                "status": recipe["status"],
                "evidence_tier": recipe["evidence_tier"],
                "source_ids": list(recipe["source_ids"]),
                "planner_policy": recipe["planner_policy"],
                "known_fields": list(recipe["known_fields"]),
                "undisclosed_fields": list(recipe["undisclosed_fields"]),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-world-model-training-recipe-index",
        "registry_id": registry["registry_id"],
        "registry_path": registry.get("registry_path"),
        "registry_sha256": registry.get("registry_sha256"),
        "items": items,
        "execution_policy": (
            "shadow_only recipes are ranking-only; execution requires a local "
            "validation receipt and explicit admission"
        ),
    }


def require_admitted_recipe(
    registry: Mapping[str, Any], recipe_id: str
) -> Mapping[str, Any]:
    recipe = find_training_recipe(registry, recipe_id)
    if recipe["status"] not in ADMITTED_STATUSES or recipe["planner_policy"] != "admitted":
        raise TrainingRecipeError(
            "TRAINING_RECIPE_EXTERNAL_NOT_ADMITTED:"
            f"{recipe_id}:{recipe['status']}"
        )
    return recipe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
