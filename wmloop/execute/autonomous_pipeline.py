"""Run or resume onboarding through settled GPU experiment execution."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import validate_document
from wmloop.control.method_candidate_compiler import MethodCandidateCompilerError
from wmloop.control.onboarding import (
    OnboardingError,
    OnboardingOptions,
    compute_asset_fingerprint,
    compute_source_revision,
    compute_source_tree_revision,
    run_onboarding,
)
from wmloop.control.onboarding_compiler import (
    OnboardingCompilerError,
    compile_and_plan,
)
from wmloop.control.onboarding_conformance import (
    ConformanceOptions,
    ModelConformanceError,
    run_conformance,
)
from wmloop.control.research_modes import ResearchModeError, normalize_research_mode
from wmloop.control.research_state import ResearchJournal
from wmloop.diagnose.probe_campaign import DiagnosticProbeError, run_diagnostic_probe
from wmloop.execute.auto_experiment import AutoExperimentError
from wmloop.execute.experiment_scheduler import (
    ExperimentSchedulerError,
    run_selected_queue,
)
from wmloop.execute.gpu_lease import GpuLeaseError
from wmloop.execute.literature_materialization import (
    LiteratureMaterializationError,
    run_literature_method_materialization,
)
from wmloop.execute.pipeline.budget import (
    _pipeline_budget_total as _pipeline_budget_total,
)
from wmloop.execute.pipeline.catalog import (
    _candidate_asset_parameters as _candidate_asset_parameters,
    _merge_method_catalogs as _merge_method_catalogs,
)
from wmloop.execute.pipeline.io import (
    _canonical_json as _canonical_json,
    _load_json as _load_json,
    _optional_file as _optional_file,
    _overlaps as _overlaps,
    _resolved as _resolved,
    _sha256 as _sha256,
    _write_json_atomic as _write_json_atomic,
)
from wmloop.execute.pipeline.journal import (
    _execution_evidence_refs as _execution_evidence_refs,
    _initial_research_state as _initial_research_state,
    _scheduler_template_path as _scheduler_template_path,
)
from wmloop.execute.pipeline.policy import (
    _nondefault_resource_policy as _nondefault_resource_policy,
)
from wmloop.execute.pipeline.retrieval import (
    _runtime_retrieval_manifest as _runtime_retrieval_manifest,
    build_irg_context,
    build_literature_query,
)
from wmloop.execute.pipeline.state import (
    _bind_output_root as _bind_output_root,
    _manifest as _manifest,
    _settle_pipeline as _settle_pipeline,
)
from wmloop.execute.pipeline.types import (
    AutonomousPipelineError as AutonomousPipelineError,
    AutonomousPipelineOptions as AutonomousPipelineOptions,
)
from wmloop.experiments.cpbe import CPBEError, publish_cpbe_plan
from wmloop.experiments.cpbe_materializer import (
    CPBEMaterializerError,
    publish_cpbe_materialization,
)
from wmloop.retrieve.evidence_capsule import (
    EvidenceCapsuleError,
    build_evidence_capsule,
    write_evidence_capsule,
)
from wmloop.retrieve.index import ProbeRetrievalError, retrieve_probe_experiences
from wmloop.retrieve.literature import (
    LiteratureRetrievalError,
    run_literature_retrieval,
)
from wmloop.retrieve.method_staging import (
    LiteratureMethodStagingError,
    run_literature_method_prompt_batch,
    run_literature_method_staging,
)
from wmloop.runtime_env import default_lock_root
from wmloop.storage import checked_path, exclusive_file_lock


def run_autonomous_pipeline(
    options: AutonomousPipelineOptions,
) -> dict[str, object]:
    """Run one writer per output root, regardless of CLI or daemon entry point."""
    destination = checked_path(
        options.output_root,
        code="AUTONOMOUS_PIPELINE_OUTPUT_INVALID",
        error=AutonomousPipelineError,
    )
    repo = Path(options.repo_root).expanduser().resolve()
    if _overlaps(destination, repo):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_OUTPUT_OVERLAPS_SOURCE")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The lock lives beside the output so acquiring it does not create an
    # unbound output directory. Never unlink it: concurrent callers share its inode.
    lock = destination.parent / (
        ".verdiwm-pipeline-" + _sha256(str(destination).encode())[:24] + ".lock"
    )
    with ExitStack() as stack:
        try:
            stack.enter_context(exclusive_file_lock(lock, blocking=False))
        except BlockingIOError as exc:
            raise AutonomousPipelineError(
                "AUTONOMOUS_PIPELINE_ALREADY_RUNNING"
            ) from exc
        return _run_autonomous_pipeline(options)


def _run_autonomous_pipeline(options: AutonomousPipelineOptions) -> dict[str, object]:
    """Run or resume the declared closed loop under the output execution lock."""

    repo = Path(options.repo_root).expanduser().resolve()
    destination = Path(options.output_root).expanduser().resolve()
    evaluator = Path(options.evaluator_contract).expanduser().resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_REPOSITORY_INVALID")
    if not evaluator.is_file() or evaluator.is_symlink():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_EVALUATOR_INVALID")
    if _overlaps(destination, repo):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_OUTPUT_OVERLAPS_SOURCE")

    retrieval_db = _resolved(options.retrieval_db, destination / "retrieval.db")
    shared_store_root = (
        retrieval_db.parent if options.retrieval_db is not None else destination
    )
    archive_db = _resolved(options.archive_db, shared_store_root / "archive.db")
    cas_root = _resolved(options.cas_root, shared_store_root / "cas")
    lock_root = Path(options.lock_root).expanduser().resolve()
    budget_db = _resolved(
        options.budget_db, destination / "compiled" / "queue" / "budget.db"
    )
    probe_contract = (
        Path(options.probe_contract).expanduser().resolve()
        if options.probe_contract is not None
        else None
    )
    if probe_contract is not None and (
        not probe_contract.is_file() or probe_contract.is_symlink()
    ):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_PROBE_CONTRACT_INVALID")
    candidate_catalog = _optional_file(
        options.candidate_catalog,
        "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID",
    )
    candidate_asset_parameters = _candidate_asset_parameters(candidate_catalog)
    settlement_manifest = _optional_file(
        options.settlement_manifest,
        "AUTONOMOUS_PIPELINE_SETTLEMENT_MANIFEST_INVALID",
    )
    cpbe_request = _optional_file(
        options.cpbe_request,
        "AUTONOMOUS_PIPELINE_CPBE_REQUEST_INVALID",
    )
    cpbe_history = _optional_file(
        options.cpbe_history,
        "AUTONOMOUS_PIPELINE_CPBE_HISTORY_INVALID",
    )
    model_irg_path = _optional_file(
        options.model_irg_path,
        "AUTONOMOUS_PIPELINE_MODEL_IRG_INVALID",
    )
    if model_irg_path is not None and not options.irg_protected_metrics:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_IRG_PROTECTED_METRICS_REQUIRED"
        )
    research_mode = None
    if options.research_mode is not None:
        try:
            research_mode = normalize_research_mode(options.research_mode)
        except ResearchModeError as exc:
            raise AutonomousPipelineError(str(exc)) from exc
    if research_mode == "causal_discovery" and (
        probe_contract is None or cpbe_request is None or cpbe_history is None
    ):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_CAUSAL_INPUTS_REQUIRED")
    if options.literature_max_results < 1 or options.literature_max_results > 50:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_LITERATURE_LIMIT_INVALID")
    if (
        options.literature_timeout_seconds <= 0
        or options.literature_timeout_seconds > 60
    ):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_LITERATURE_TIMEOUT_INVALID")
    if (
        not math.isfinite(options.budget_max_trial_gpu_hours)
        or options.budget_max_trial_gpu_hours <= 0
        or options.budget_high_trial_limit < 0
    ):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_RESOURCE_POLICY_INVALID")
    budget_total_gpu_hours = _pipeline_budget_total(
        evaluator=evaluator,
        probe_contract=probe_contract,
        override=options.budget_total_gpu_hours,
    )
    for output in (archive_db, cas_root, lock_root, budget_db, retrieval_db):
        if _overlaps(output, repo):
            raise AutonomousPipelineError(
                "AUTONOMOUS_PIPELINE_PERSISTENT_OUTPUT_OVERLAPS_SOURCE"
            )
    if probe_contract is not None and _overlaps(probe_contract, repo):
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_PROBE_CONTRACT_OVERLAPS_SOURCE"
        )

    input_document = _input_document(
        options,
        repo=repo,
        evaluator=evaluator,
        archive_db=archive_db,
        cas_root=cas_root,
        lock_root=lock_root,
        budget_db=budget_db,
        retrieval_db=retrieval_db,
        probe_contract=probe_contract,
        candidate_catalog=candidate_catalog,
        settlement_manifest=settlement_manifest,
        research_mode=research_mode,
        cpbe_request=cpbe_request,
        cpbe_history=cpbe_history,
        model_irg_path=model_irg_path,
        irg_protected_metrics=options.irg_protected_metrics,
        budget_total_gpu_hours=budget_total_gpu_hours,
    )
    input_hash = _sha256(_canonical_json(input_document))
    _bind_output_root(destination, input_document=input_document, input_hash=input_hash)

    sidecar = destination / "onboarding"
    conformance = destination / "conformance"
    diagnostic_probe_root = destination / "diagnostic-probe"
    literature_root = destination / "literature"
    literature_method_root = destination / "literature-methods"
    literature_method_prompt_root = destination / "literature-method-prompts"
    literature_method_materialization_root = (
        destination / "literature-method-materialization"
    )
    compiled = destination / "compiled"
    cpbe_root = destination / "causal-discovery"
    journal = ResearchJournal.open_or_create(
        destination / "research",
        input_hash=input_hash,
        initial=_initial_research_state(
            evaluator, budget_total_gpu_hours, model_irg_path
        ),
    )
    current = journal.read()
    if current["phase"] == "settled":
        manifest = _load_json(
            destination / "pipeline-manifest.json",
            "AUTONOMOUS_PIPELINE_MANIFEST_INVALID",
        )
        validate_document("autonomous_pipeline_manifest", manifest)
        if manifest["input_hash"] != input_hash or manifest["state"] != "settled":
            raise AutonomousPipelineError(
                "AUTONOMOUS_PIPELINE_RESEARCH_SETTLEMENT_MISMATCH"
            )
        return manifest
    if current["phase"] != "observe":
        if current["phase"] != "blocked":
            journal.advance(
                "blocked", "Previous pipeline attempt interrupted before settlement."
            )
        journal.advance(
            "observe", "Resume bound inputs and existing experiment receipts."
        )
    stage = "onboarding"
    try:
        if sidecar.exists() or sidecar.is_symlink():
            onboarding_manifest = _load_json(
                sidecar / "manifest.json", "AUTONOMOUS_PIPELINE_ONBOARDING_INVALID"
            )
        else:
            onboarding_manifest = run_onboarding(
                OnboardingOptions(
                    repo_root=repo,
                    output_root=sidecar,
                    runtime_python=options.runtime_python,
                    evaluator_contract=evaluator,
                    asset_bindings=options.asset_bindings,
                    additional_asset_parameters=(
                        *candidate_asset_parameters,
                        *options.adapter_asset_parameters,
                    ),
                    adapter_contract_ready=options.adapter_contract_ready,
                    probe_imports=options.probe_imports,
                    max_files=options.max_files,
                )
            )
        if onboarding_manifest.get("state") != "ready_for_conformance_smoke":
            return _settle_pipeline(
                destination,
                input_hash=input_hash,
                journal=journal,
                state="blocked",
                verdict="BLOCKED",
                blocked_stage="onboarding",
            )

        stage = "conformance"
        conformance_manifest = run_conformance(
            ConformanceOptions(
                sidecar_root=sidecar,
                output_root=conformance,
                timeout_seconds=options.conformance_timeout_seconds,
            )
        )
        if conformance_manifest.get("verdict") != "PASS":
            return _settle_pipeline(
                destination,
                input_hash=input_hash,
                journal=journal,
                state="blocked",
                verdict="BLOCKED",
                blocked_stage="conformance",
            )

        probe_manifest: dict[str, object] | None = None
        retrieval_context: dict[str, object] = {"state": "cold_start", "matches": []}
        irg_failure_signatures: tuple[str, ...] = ()
        irg_model_family: str | None = None
        irg_guided_context: dict[str, object] | None = None
        irg_guided_context, irg_failure_signatures, irg_model_family = (
            build_irg_context(
                model_irg_path,
                options.irg_protected_metrics,
            )
        )
        if irg_guided_context is not None:
            retrieval_context["irg_guided"] = irg_guided_context
        literature_manifest: dict[str, object] | None = None
        literature_method_manifest: dict[str, object] | None = None
        literature_method_prompt_manifest: dict[str, object] | None = None
        literature_method_materialization_manifest: dict[str, object] | None = None
        cpbe_manifest: dict[str, object] | None = None
        if probe_contract is not None:
            stage = "diagnostic_probe"
            probe_manifest = run_diagnostic_probe(
                contract_path=probe_contract,
                sidecar_root=sidecar,
                conformance_root=conformance,
                output_root=diagnostic_probe_root,
                workspace_root=repo,
                archive_db=archive_db,
                cas_root=cas_root,
                lock_root=lock_root,
                budget_db=budget_db,
                budget_total_gpu_hours=budget_total_gpu_hours,
                budget_max_trial_gpu_hours=options.budget_max_trial_gpu_hours,
                budget_high_trial_limit=options.budget_high_trial_limit,
                budget_require_high_cost_approval=(
                    options.budget_require_high_cost_approval
                ),
                retrieval_db=retrieval_db,
                campaign_id=destination.name,
            )
            if probe_manifest.get("verdict") != "PASS":
                return _settle_pipeline(
                    destination,
                    input_hash=input_hash,
                    journal=journal,
                    state="blocked",
                    verdict="BLOCKED",
                    blocked_stage="diagnostic_probe",
                    diagnostic_probe=probe_manifest,
                )
            signatures = [
                str(value)
                for value in probe_manifest.get("failure_signatures", [])
                if isinstance(value, str)
            ]
            matches = retrieve_probe_experiences(
                database_path=retrieval_db,
                model_family=str(probe_manifest["model_family"]),
                runtime_capability=str(probe_manifest["runtime_capability"]),
                failure_signatures=signatures,
                archive_db=archive_db,
                cas_root=cas_root,
                exclude_archive_trial_id=(
                    str(probe_manifest.get("archive_trial_id") or "") or None
                ),
            )
            retrieval_context = {
                "state": "matched" if matches else "cold_start",
                "query": {
                    "model_family": probe_manifest["model_family"],
                    "runtime_capability": probe_manifest["runtime_capability"],
                    "failure_signatures": signatures,
                },
                "matches": [row.to_dict() for row in matches],
                "index_path": str(retrieval_db),
            }
            # Preserve the model-conditioned IRG plan when probe retrieval is
            # also active.  It is part of the same immutable routing context,
            # not a competing retrieval mode.
            if irg_guided_context is not None:
                retrieval_context["irg_guided"] = irg_guided_context
        capsule = build_evidence_capsule(
            probe=probe_manifest,
            matches=[
                row
                for row in retrieval_context.get("matches", [])
                if isinstance(row, Mapping)
            ],
        )
        capsule_path = write_evidence_capsule(
            destination / "retrieval" / "evidence-capsule.json", capsule
        )
        retrieval_context["capsule"] = capsule
        retrieval_context["capsule_path"] = str(capsule_path)
        journal.advance(
            "hypothesize",
            "Diagnostic observations and retrieval evidence are available; causes remain hypotheses.",
            evidence_refs=[*journal.read()["evidence_refs"], str(capsule_path)],
        )
        literature_query = build_literature_query(
            explicit_query=options.literature_query,
            retrieval_context=retrieval_context,
            probe_manifest=probe_manifest,
            irg_failure_signatures=irg_failure_signatures,
            irg_model_family=irg_model_family,
        )
        if literature_query and research_mode != "causal_discovery":
            stage = "literature_retrieval"
            literature_manifest = run_literature_retrieval(
                query=literature_query,
                output_root=literature_root,
                max_results=options.literature_max_results,
                timeout_seconds=options.literature_timeout_seconds,
            )
            stage = "literature_method_staging"
            literature_method_manifest = run_literature_method_staging(
                literature_manifest=literature_root / "manifest.json",
                output_root=literature_method_root,
                repo_root=repo,
                failure_signatures=tuple(
                    dict.fromkeys(
                        [
                            str(value)
                            for value in (probe_manifest or {}).get(
                                "failure_signatures", []
                            )
                            if isinstance(value, str)
                        ]
                        + list(irg_failure_signatures)
                    )
                ),
                model_family=(
                    str((probe_manifest or {}).get("model_family"))
                    if (probe_manifest or {}).get("model_family")
                    else irg_model_family
                ),
            )
            stage = "literature_method_prompts"
            literature_method_prompt_manifest = run_literature_method_prompt_batch(
                method_staging_manifest=literature_method_root / "manifest.json",
                output_root=literature_method_prompt_root,
                repo_root=Path(__file__).resolve().parents[2],
                archive_db=archive_db,
                cas_root=cas_root,
            )
            # Unknown literature methods are automatically materialized in
            # isolated snapshots. A failed method becomes a capability gap;
            # it must not prevent other methods or the baseline from running.
            try:
                literature_method_materialization_manifest = (
                    run_literature_method_materialization(
                        method_staging_manifest=literature_method_root
                        / "manifest.json",
                        output_root=literature_method_materialization_root,
                        source_root=repo,
                        project_root=Path(__file__).resolve().parents[2],
                        evaluator_contract=evaluator,
                        runtime_python=options.runtime_python,
                        archive_db=archive_db,
                        cas_root=cas_root,
                        max_candidates=options.literature_max_results,
                    )
                )
                generated_catalog = literature_method_materialization_manifest.get(
                    "candidate_catalog_path"
                )
                if (
                    isinstance(generated_catalog, str)
                    and Path(generated_catalog).is_file()
                ):
                    generated_path = Path(generated_catalog).resolve()
                    if candidate_catalog is None:
                        candidate_catalog = generated_path
                    else:
                        candidate_catalog = _merge_method_catalogs(
                            candidate_catalog,
                            generated_path,
                            destination
                            / "literature-method-materialization"
                            / "merged-candidate-catalog.json",
                        )
            except LiteratureMaterializationError as exc:
                literature_method_materialization_manifest = {
                    "state": "blocked",
                    "error": str(exc),
                    "claim_boundary": "No literature method received execution authority.",
                }

        if (
            research_mode in {"causal_discovery", "hybrid"}
            and probe_contract is not None
            and cpbe_request is not None
            and cpbe_history is not None
        ):
            stage = "causal_discovery_plan"
            plan_manifest = publish_cpbe_plan(
                request_path=cpbe_request,
                history_path=cpbe_history,
                output_root=cpbe_root / "plan",
                archive_db=archive_db,
                cas_root=cas_root,
            )
            cpbe_manifest = {"plan": plan_manifest}
            try:
                stage = "causal_discovery_materialization"
                cpbe_manifest["materialization"] = publish_cpbe_materialization(
                    plan_path=cpbe_root / "plan" / "cpbe-plan.json",
                    output_root=cpbe_root / "materialization",
                )
                cpbe_manifest["state"] = "awaiting_runtime_canary"
            except CPBEMaterializerError as exc:
                cpbe_manifest["state"] = "materialization_blocked"
                cpbe_manifest["blocker"] = str(exc)
            cpbe_manifest["claim_boundary"] = (
                "CPBE output is diagnostic-only until a target adapter executes and settles "
                "runtime canary, expanded selector, and repair confirmation receipts."
            )
            if research_mode == "causal_discovery":
                blocker = (
                    "causal_discovery_materialization"
                    if cpbe_manifest["state"] == "materialization_blocked"
                    else "causal_discovery_runtime_canary"
                )
                return _settle_pipeline(
                    destination,
                    input_hash=input_hash,
                    journal=journal,
                    state="blocked",
                    verdict="BLOCKED",
                    blocked_stage=blocker,
                    diagnostic_probe=probe_manifest,
                    retrieval=_runtime_retrieval_manifest(retrieval_context),
                    cpbe=cpbe_manifest,
                )

        journal.advance(
            "select",
            "Compile implementable candidates under the bound evaluator and budget.",
        )
        stage = "compilation"
        compilation_manifest = compile_and_plan(
            sidecar_root=sidecar,
            conformance_root=conformance,
            output_root=compiled,
            diagnostic_probe_manifest=(
                (diagnostic_probe_root / "manifest.json")
                if probe_manifest is not None
                else None
            ),
            retrieval_context=retrieval_context,
            literature_manifest=(
                (literature_root / "manifest.json")
                if literature_manifest is not None
                else None
            ),
            literature_method_manifest=(
                (literature_method_root / "manifest.json")
                if literature_method_manifest is not None
                else None
            ),
            candidate_catalog=candidate_catalog,
            settlement_manifest=settlement_manifest,
        )
        method_candidate_report = None
        method_candidate_path = compilation_manifest.get(
            "method_candidate_compilation_path"
        )
        if isinstance(method_candidate_path, str) and method_candidate_path:
            method_candidate_report = _load_json(
                Path(method_candidate_path),
                "AUTONOMOUS_PIPELINE_METHOD_COMPILATION_INVALID",
            )
        queue_path = Path(str(compilation_manifest["queue_path"]))

        queue = _load_json(queue_path, "AUTONOMOUS_PIPELINE_QUEUE_INVALID")
        journal.advance(
            "implement",
            "Execute the selected candidate plans; interface checks do not establish repair effects.",
            candidate_actions=[
                dict(row)
                for row in queue.get("selected", [])
                if isinstance(row, Mapping)
            ],
        )
        stage = "execution"
        execution = run_selected_queue(
            queue_path=queue_path,
            workspace_root=repo,
            archive_db=archive_db,
            cas_root=cas_root,
            lock_root=lock_root,
            budget_db=budget_db,
            budget_total_gpu_hours=budget_total_gpu_hours,
            budget_max_trial_gpu_hours=options.budget_max_trial_gpu_hours,
            budget_high_trial_limit=options.budget_high_trial_limit,
            budget_require_high_cost_approval=(
                options.budget_require_high_cost_approval
            ),
        )
        journal.advance(
            "verify",
            "Inspect scheduler completion; scientific effects remain governed by frozen trial verdicts.",
        )
        candidate_states = execution.get("candidate_states")
        passed = (
            isinstance(candidate_states, Mapping)
            and bool(candidate_states)
            and all(value == "completed" for value in candidate_states.values())
        )
        journal.advance(
            "remember",
            "Retain all available execution outcomes, including blocked candidates; no positive effect is inferred.",
            evidence_refs=_execution_evidence_refs(journal, compiled),
        )
        return _settle_pipeline(
            destination,
            input_hash=input_hash,
            journal=journal,
            state="settled" if passed else "blocked",
            verdict="PASS" if passed else "BLOCKED",
            blocked_stage=None if passed else "execution",
            candidate_states=(
                dict(candidate_states) if isinstance(candidate_states, Mapping) else {}
            ),
            diagnostic_probe=probe_manifest,
            retrieval=_runtime_retrieval_manifest(retrieval_context),
            literature=literature_manifest,
            literature_methods=literature_method_manifest,
            literature_method_prompts=literature_method_prompt_manifest,
            literature_method_materialization=literature_method_materialization_manifest,
            method_candidates=method_candidate_report,
            cpbe=cpbe_manifest,
        )
    except Exception as exc:
        if journal.read()["phase"] not in {"blocked", "settled"}:
            journal.advance(
                "blocked",
                f"Pipeline interrupted at {stage}: {type(exc).__name__}",
                evidence_refs=_execution_evidence_refs(journal, compiled),
            )
        _write_json_atomic(
            destination / "pipeline-manifest.json",
            _manifest(
                destination,
                input_hash=input_hash,
                state="interrupted",
                verdict=None,
                blocked_stage=stage,
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            ),
        )
        raise


def _input_document(
    options: AutonomousPipelineOptions,
    *,
    repo: Path,
    evaluator: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path,
    budget_db: Path,
    retrieval_db: Path,
    probe_contract: Path | None,
    candidate_catalog: Path | None,
    settlement_manifest: Path | None,
    research_mode: str | None,
    cpbe_request: Path | None,
    cpbe_history: Path | None,
    model_irg_path: Path | None,
    irg_protected_metrics: Sequence[str],
    budget_total_gpu_hours: float,
) -> dict[str, object]:
    bindings: list[dict[str, str]] = []
    parameters: set[str] = set()
    for raw_parameter, raw_path in options.asset_bindings:
        parameter = (
            raw_parameter if raw_parameter.startswith("--") else f"--{raw_parameter}"
        )
        if parameter in parameters:
            raise AutonomousPipelineError(
                f"AUTONOMOUS_PIPELINE_ASSET_DUPLICATE:{parameter}"
            )
        parameters.add(parameter)
        path = Path(raw_path).expanduser().resolve()
        try:
            fingerprint = compute_asset_fingerprint(path)
        except OnboardingError as exc:
            raise AutonomousPipelineError(
                f"AUTONOMOUS_PIPELINE_ASSET_INVALID:{parameter}"
            ) from exc
        bindings.append(
            {
                "parameter": parameter,
                "path": str(path),
                "fingerprint": fingerprint,
            }
        )
    runtime = (
        str(Path(options.runtime_python).expanduser().resolve())
        if options.runtime_python is not None
        else None
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-pipeline-input",
        "repo_root": str(repo),
        "source_revision": compute_source_revision(repo, max_files=options.max_files),
        "source_tree_revision": compute_source_tree_revision(
            repo, max_files=options.max_files
        ),
        "runtime_python": runtime,
        "evaluator_contract": str(evaluator),
        "evaluator_sha256": _sha256(evaluator.read_bytes()),
        "scheduler_template_sha256": _sha256(
            _scheduler_template_path(evaluator).read_bytes()
        ),
        "adapter_contract_ready": options.adapter_contract_ready,
        "asset_bindings": sorted(bindings, key=lambda row: row["parameter"]),
        "adapter_asset_parameters": sorted(
            {
                parameter if parameter.startswith("--") else f"--{parameter}"
                for parameter in options.adapter_asset_parameters
            }
        ),
        "probe_imports": options.probe_imports,
        "max_files": options.max_files,
        "conformance_timeout_seconds": options.conformance_timeout_seconds,
        "archive_db": str(archive_db),
        "cas_root": str(cas_root),
        "lock_root": str(lock_root),
        "budget_db": str(budget_db),
        "budget_total_gpu_hours": budget_total_gpu_hours,
        **_nondefault_resource_policy(options),
        "retrieval_db": str(retrieval_db),
        "probe_contract": str(probe_contract) if probe_contract is not None else None,
        "probe_contract_sha256": (
            _sha256(probe_contract.read_bytes()) if probe_contract is not None else None
        ),
        "literature_query": options.literature_query,
        "literature_max_results": options.literature_max_results,
        "literature_timeout_seconds": options.literature_timeout_seconds,
        "candidate_catalog": (
            str(candidate_catalog) if candidate_catalog is not None else None
        ),
        "candidate_catalog_sha256": (
            _sha256(candidate_catalog.read_bytes())
            if candidate_catalog is not None
            else None
        ),
        "settlement_manifest": (
            str(settlement_manifest) if settlement_manifest is not None else None
        ),
        "settlement_manifest_sha256": (
            _sha256(settlement_manifest.read_bytes())
            if settlement_manifest is not None
            else None
        ),
        "research_mode": research_mode,
        "cpbe_request": str(cpbe_request) if cpbe_request is not None else None,
        "cpbe_request_sha256": (
            _sha256(cpbe_request.read_bytes()) if cpbe_request is not None else None
        ),
        "cpbe_history": str(cpbe_history) if cpbe_history is not None else None,
        "cpbe_history_sha256": (
            _sha256(cpbe_history.read_bytes()) if cpbe_history is not None else None
        ),
        "model_irg_path": str(model_irg_path) if model_irg_path is not None else None,
        "model_irg_sha256": (
            _sha256(model_irg_path.read_bytes()) if model_irg_path is not None else None
        ),
        "irg_protected_metrics": list(irg_protected_metrics),
    }


def _parse_asset(value: str) -> tuple[str, Path]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_ASSET_ARGUMENT_INVALID")
    return parameter.strip(), Path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--asset", action="append", default=[], metavar="PARAM=PATH")
    parser.add_argument("--adapter-asset-parameter", action="append", default=[])
    parser.add_argument("--adapter-contract-ready", action="store_true")
    parser.add_argument("--no-import-probe", action="store_true")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--conformance-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    parser.add_argument("--lock-root", type=Path, default=default_lock_root())
    parser.add_argument("--budget-db", type=Path)
    parser.add_argument("--budget-total-gpu-hours", type=float)
    parser.add_argument("--budget-max-trial-gpu-hours", type=float, default=120.0)
    parser.add_argument("--budget-high-trial-limit", type=int, default=2)
    parser.add_argument("--auto-approve-high-cost", action="store_true")
    parser.add_argument("--probe-contract", type=Path)
    parser.add_argument("--retrieval-db", type=Path)
    parser.add_argument("--literature-query")
    parser.add_argument("--literature-max-results", type=int, default=8)
    parser.add_argument("--literature-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--candidate-catalog", type=Path)
    parser.add_argument("--settlement-manifest", type=Path)
    parser.add_argument(
        "--research-mode",
        choices=("quick_start", "causal_discovery", "hybrid"),
    )
    parser.add_argument("--cpbe-request", type=Path)
    parser.add_argument("--cpbe-history", type=Path)
    parser.add_argument(
        "--model-irg",
        type=Path,
        help="model-conditioned IRG artifact used to guide cross-domain retrieval",
    )
    parser.add_argument(
        "--irg-protected-metric",
        action="append",
        default=[],
        help="metric protected while deriving IRG-guided methods (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = run_autonomous_pipeline(
            AutonomousPipelineOptions(
                repo_root=args.repo_root,
                output_root=args.output_root,
                evaluator_contract=args.evaluator_contract,
                runtime_python=args.runtime_python,
                asset_bindings=tuple(_parse_asset(value) for value in args.asset),
                adapter_asset_parameters=tuple(args.adapter_asset_parameter),
                adapter_contract_ready=args.adapter_contract_ready,
                probe_imports=not args.no_import_probe,
                max_files=args.max_files,
                conformance_timeout_seconds=args.conformance_timeout_seconds,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                lock_root=args.lock_root,
                budget_db=args.budget_db,
                budget_total_gpu_hours=args.budget_total_gpu_hours,
                budget_max_trial_gpu_hours=args.budget_max_trial_gpu_hours,
                budget_high_trial_limit=args.budget_high_trial_limit,
                budget_require_high_cost_approval=not args.auto_approve_high_cost,
                probe_contract=args.probe_contract,
                retrieval_db=args.retrieval_db,
                literature_query=args.literature_query,
                literature_max_results=args.literature_max_results,
                literature_timeout_seconds=args.literature_timeout_seconds,
                candidate_catalog=args.candidate_catalog,
                settlement_manifest=args.settlement_manifest,
                research_mode=args.research_mode,
                cpbe_request=args.cpbe_request,
                cpbe_history=args.cpbe_history,
                model_irg_path=args.model_irg,
                irg_protected_metrics=tuple(args.irg_protected_metric),
            )
        )
    except (
        AutonomousPipelineError,
        AutoExperimentError,
        ExperimentSchedulerError,
        GpuLeaseError,
        ModelConformanceError,
        OnboardingCompilerError,
        OnboardingError,
        DiagnosticProbeError,
        LiteratureRetrievalError,
        LiteratureMethodStagingError,
        MethodCandidateCompilerError,
        EvidenceCapsuleError,
        ProbeRetrievalError,
        CPBEError,
        CPBEMaterializerError,
        ResearchModeError,
    ) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0 if manifest.get("verdict") == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
