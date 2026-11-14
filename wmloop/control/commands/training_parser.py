"""CLI grammar for bounded training planning and evidence commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path


def add_training_plan_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]],
    repository_root: Callable[[], Path],
) -> None:
    """Register the training-scale planning commands."""

    plan_training = commands.add_parser(
        "plan-training",
        help="derive a bounded world-model training scale from sample manifests",
    )
    plan_training.add_argument("--train-manifest", type=Path)
    plan_training.add_argument("--val-manifest", type=Path)
    plan_training.add_argument(
        "--dataset-root", "--agibot-root", dest="dataset_root", type=Path
    )
    plan_training.add_argument(
        "--stage",
        choices=(
            "engineering_diagnostic",
            "probe",
            "smoke",
            "screen",
            "pilot",
            "confirm",
        ),
        default="screen",
    )
    plan_training.add_argument("--batch-size", type=int, default=1)
    plan_training.add_argument("--gradient-accumulation", type=int, default=1)
    plan_training.add_argument("--world-size", type=int, default=1)
    plan_training.add_argument("--sequence-length", type=int)
    plan_training.add_argument("--seed-count", type=int)
    plan_training.add_argument("--seed", type=int, default=0)
    plan_training.add_argument(
        "--training-recipe",
        help="admitted local recipe id; shadow-only literature is rejected",
    )
    plan_training.add_argument(
        "--training-recipe-registry",
        type=Path,
        default=repository_root()
        / "configs"
        / "retrieval"
        / "world_model_training_recipes_v1.json",
    )
    plan_training.add_argument("--schema-root", type=Path)
    plan_training.add_argument("--repo-root", type=Path)
    plan_training.add_argument("--output", type=Path)
    plan_training.set_defaults(handler=handlers["_plan_training"])

    plan_training_ladder = commands.add_parser(
        "plan-training-ladder",
        help="plan automatic probe-to-formal training upgrades without launching a model",
    )
    plan_training_ladder.add_argument("--train-manifest", type=Path, required=True)
    plan_training_ladder.add_argument("--val-manifest", type=Path, required=True)
    plan_training_ladder.add_argument(
        "--current-stage",
        choices=("probe", "smoke", "screen", "pilot"),
        default="probe",
    )
    plan_training_ladder.add_argument(
        "--target-stage", choices=("screen", "pilot", "confirm"), default="confirm"
    )
    plan_training_ladder.add_argument("--batch-size", type=int, default=1)
    plan_training_ladder.add_argument("--gradient-accumulation", type=int, default=1)
    plan_training_ladder.add_argument("--world-size", type=int, default=1)
    plan_training_ladder.add_argument("--sequence-length", type=int)
    plan_training_ladder.add_argument("--seed-count", type=int)
    plan_training_ladder.add_argument("--seed", type=int, default=0)
    plan_training_ladder.add_argument("--schema-root", type=Path)
    plan_training_ladder.add_argument("--repo-root", type=Path)
    plan_training_ladder.add_argument("--output", type=Path)
    plan_training_ladder.set_defaults(handler=handlers["_plan_training_ladder"])


def add_training_recipes_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]],
) -> None:
    training_recipes = commands.add_parser(
        "training-recipes",
        help="list auditable world-model training recipes and their evidence tiers",
    )
    training_recipes.add_argument("--registry", type=Path)
    training_recipes.add_argument("--recipe-id")
    training_recipes.add_argument("--repo-root", type=Path)
    training_recipes.set_defaults(handler=handlers["_training_recipes"])


def add_diagnose_training_gain_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]],
) -> None:
    diagnose_gain = commands.add_parser(
        "diagnose-training-gain",
        help="plan experiments that distinguish optimization, data, capacity, and mechanism limits",
    )
    diagnose_gain.add_argument("--training-receipt", type=Path, required=True)
    diagnose_gain.add_argument("--screen-settlement", type=Path, required=True)
    diagnose_gain.add_argument("--confirm-settlement", type=Path, required=True)
    diagnose_gain.add_argument("--verifier-manifest", type=Path, required=True)
    diagnose_gain.add_argument("--output", type=Path)
    diagnose_gain.set_defaults(handler=handlers["_diagnose_training_gain"])
