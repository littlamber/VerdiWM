"""Resumable pipeline state and evidence components: types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wmloop.control.onboarding_scan.types import DEFAULT_RUNTIME_TIMEOUT_SECONDS
from wmloop.runtime_env import default_lock_root


class AutonomousPipelineError(RuntimeError):
    """The immutable pipeline inputs or output root are invalid."""


@dataclass(frozen=True)
class AutonomousPipelineOptions:
    """Inputs for one resumable external-model experiment transaction."""

    repo_root: Path
    output_root: Path
    evaluator_contract: Path
    runtime_python: Path | None = None
    asset_bindings: tuple[tuple[str, Path], ...] = ()
    adapter_asset_parameters: tuple[str, ...] = ()
    adapter_contract_ready: bool = False
    probe_imports: bool = True
    max_files: int = 20_000
    conformance_timeout_seconds: float = 30.0
    archive_db: Path | None = None
    cas_root: Path | None = None
    lock_root: Path = field(default_factory=default_lock_root)
    budget_db: Path | None = None
    budget_total_gpu_hours: float | None = None
    budget_max_trial_gpu_hours: float = 120.0
    budget_high_trial_limit: int = 2
    budget_require_high_cost_approval: bool = True
    probe_contract: Path | None = None
    retrieval_db: Path | None = None
    literature_query: str | None = None
    literature_max_results: int = 8
    literature_timeout_seconds: float = 10.0
    candidate_catalog: Path | None = None
    settlement_manifest: Path | None = None
    research_mode: str | None = None
    cpbe_request: Path | None = None
    cpbe_history: Path | None = None
    model_irg_path: Path | None = None
    irg_protected_metrics: tuple[str, ...] = ()
    runtime_timeout_seconds: float = DEFAULT_RUNTIME_TIMEOUT_SECONDS
