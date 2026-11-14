"""Training planning, recipe lookup, and attribution CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from wmloop.control.commands.io import emit_json, repository_root
from wmloop.experiments.training_gain_attribution import (
    build_training_gain_attribution,
    write_training_gain_attribution,
)
from wmloop.experiments.training_scale import (
    TrainingScaleError,
    build_training_ladder,
    build_training_scale_plan,
    write_training_scale_plan,
)
from wmloop.retrieve.training_recipes import (
    find_training_recipe,
    load_training_recipe_registry,
    require_admitted_recipe,
    summarize_training_recipes,
)


def _root(args: argparse.Namespace) -> Path:
    return (
        (args.schema_root or args.repo_root or repository_root()).expanduser().resolve()
    )


def plan(args: argparse.Namespace) -> int:
    dataset_root = getattr(args, "dataset_root", None)
    train_manifest = getattr(args, "train_manifest", None)
    val_manifest = getattr(args, "val_manifest", None)
    if dataset_root is None and val_manifest is None:
        raise TrainingScaleError("TRAINING_SCALE_VAL_MANIFEST_REQUIRED")
    if dataset_root is not None and val_manifest is not None:
        raise TrainingScaleError("TRAINING_SCALE_DATASET_INPUTS_CONFLICT")
    root = _root(args)
    profile = None
    if args.training_recipe:
        registry = load_training_recipe_registry(
            args.training_recipe_registry, root=root
        )
        profile = require_admitted_recipe(registry, args.training_recipe)
    result = build_training_scale_plan(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        dataset_root=dataset_root,
        stage=args.stage,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        world_size=args.world_size,
        sequence_length=args.sequence_length,
        requested_seed_count=args.seed_count,
        seed=args.seed,
        training_profile=profile,
        root=root,
    )
    if args.output:
        write_training_scale_plan(result, args.output)
    emit_json(result)
    return 0 if result["state"] == "ready" else 3


def ladder(args: argparse.Namespace) -> int:
    result = build_training_ladder(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        current_stage=args.current_stage,
        target_stage=args.target_stage,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        world_size=args.world_size,
        sequence_length=args.sequence_length,
        requested_seed_count=args.seed_count,
        seed=args.seed,
        root=_root(args),
    )
    if args.output:
        write_training_scale_plan(result, args.output)
    emit_json(result)
    return 0 if result["state"] == "ready" else 3


def recipes(args: argparse.Namespace) -> int:
    root = (args.repo_root or repository_root()).expanduser().resolve()
    registry_path = (
        args.registry
        or root / "configs" / "retrieval" / "world_model_training_recipes_v1.json"
    )
    registry = load_training_recipe_registry(registry_path, root=root)
    if args.recipe_id:
        emit_json(
            {
                "index": summarize_training_recipes(registry, recipe_id=args.recipe_id),
                "recipe": find_training_recipe(registry, args.recipe_id),
            }
        )
    else:
        emit_json(summarize_training_recipes(registry))
    return 0


def diagnose_gain(args: argparse.Namespace) -> int:
    result = build_training_gain_attribution(
        training_receipt_path=args.training_receipt,
        screen_settlement_path=args.screen_settlement,
        confirm_settlement_path=args.confirm_settlement,
        verifier_manifest_path=args.verifier_manifest,
    )
    if args.output is not None:
        write_training_gain_attribution(result, args.output)
    emit_json(result)
    return 0
