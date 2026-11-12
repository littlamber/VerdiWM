"""Project onboarding and external evaluator command implementations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from wmloop.control.commands.io import emit_json
from wmloop.control.evaluator_discovery import EvaluatorDiscoveryError, discover_evaluator_candidates
from wmloop.control.evaluator_overlay import (
    EvaluatorOverlayError,
    materialize_evaluator_overlay,
)
from wmloop.control.first_contact import initialize_project, inspect_project
from wmloop.control.onboarding_assistant import (
    build_onboarding_questionnaire,
    write_onboarding_questionnaire,
)
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_request import ResearchRequestError


def _configured() -> dict[str, Any]:
    # A missing project is a valid input for ``guide-model``; a malformed one
    # must remain an actionable error rather than being mistaken for absence.
    try:
        return load_project_config(cwd=Path.cwd()).values
    except ProjectConfigError as exc:
        raise ResearchRequestError(str(exc)) from exc


def init(args: argparse.Namespace) -> int:
    result = initialize_project(
        root=Path.cwd(),
        model=args.model,
        source=args.source,
        data=args.data,
        goal=args.goal,
        budget=args.budget,
        mode=args.mode,
        target_metrics=args.target_metrics,
        evaluator_contract=str(args.evaluator_contract)
        if args.evaluator_contract
        else None,
        runtime_python=str(args.runtime_python) if args.runtime_python else None,
        project_file=args.project_file,
        force=args.force,
    )
    emit_json(result)
    return 0 if result["state"] == "ready" else 3


def check_model(args: argparse.Namespace) -> int:
    configured = _configured()
    result = inspect_project(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        evaluator_contract=str(args.evaluator_contract)
        if args.evaluator_contract
        else configured.get("evaluator_contract"),
        runtime_python=str(args.runtime_python)
        if args.runtime_python
        else configured.get("runtime_python"),
    )
    emit_json(result)
    return 0 if result["state"] == "ready_for_conformance" else 2


def guide(args: argparse.Namespace) -> int:
    configured = _configured()
    questionnaire = build_onboarding_questionnaire(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        goal=args.goal or configured.get("goal"),
        evaluator_contract=args.evaluator_contract
        or configured.get("evaluator_contract"),
        runtime_python=args.runtime_python or configured.get("runtime_python"),
    )
    if args.output is not None:
        questionnaire["questionnaire_path"] = str(
            write_onboarding_questionnaire(args.output, questionnaire)
        )
    emit_json(questionnaire)
    return 0 if questionnaire["state"] == "awaiting_confirmation" else 2


def wan22_droid(args: argparse.Namespace) -> int:
    from wmloop.wan22_droid import (
        Wan22DroidError,
        validate_contract,
        write_sample_manifest,
    )

    try:
        if args.phase == "prepare":
            output = args.output_root.expanduser().resolve()
            train = write_sample_manifest(
                args.data_root,
                "train",
                output / "train.json",
                horizon_frames=args.horizon_frames,
                stride=args.stride,
            )
            val = write_sample_manifest(
                args.data_root,
                "val",
                output / "val.json",
                horizon_frames=args.horizon_frames,
                stride=args.stride,
            )
            emit_json(
                {
                    "state": "ready",
                    "output_root": str(output),
                    "train": train["record_count"],
                    "validation": val["record_count"],
                }
            )
            return 0
        report = validate_contract(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            model=args.model,
            source=args.source,
            evaluator_contract=args.evaluator_contract,
            adapter=args.adapter,
            horizon_frames=args.horizon_frames,
        )
        emit_json(report)
        return 0 if report["state"] == "ready_for_execution" else 2
    except Wan22DroidError as exc:
        emit_json({"state": "blocked", "error": str(exc)})
        return 2


def evaluator_overlay(args: argparse.Namespace) -> int:
    try:
        result = materialize_evaluator_overlay(
            args.candidate,
            source_root=args.source_root,
            model_root=args.model_root,
            data_root=args.data_root,
            output_root=args.output_root,
            confirm=args.confirm,
        )
    except EvaluatorOverlayError as exc:
        emit_json({"state": "blocked", "error": str(exc)})
        return 2
    emit_json(result)
    return 0 if result["state"] == "frozen" else 1


def evaluator_discover(args: argparse.Namespace) -> int:
    try:
        emit_json(discover_evaluator_candidates(args.source_root))
    except EvaluatorDiscoveryError as exc:
        raise ResearchRequestError(str(exc)) from exc
    return 0
