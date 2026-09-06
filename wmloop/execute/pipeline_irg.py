"""IRG lifecycle boundary used by the autonomous execution pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from wmloop.control.model_irg_materializer import (
    ModelIRGMaterializationError,
    materialize_model_irg,
)
from wmloop.control.probe_evolution_cycle import (
    ProbeEvolutionCycleError,
    plan_probe_evolution_cycle,
)
from wmloop.retrieve.coordinator import (
    DiscoveryCoordinatorError,
    prepare_irg_discovery,
)


class PipelineIRGError(RuntimeError):
    """IRG inputs or lifecycle stages are invalid for the autonomous pipeline."""


@dataclass(frozen=True)
class PipelineIRGInputs:
    materialized: Path | None
    portrait: Path | None
    asset: Path | None
    diagnostic_axes: Path | None
    method_effects: Path | None
    comparisons: tuple[Path, ...]

    @property
    def enabled(self) -> bool:
        return self.materialized is not None or self.portrait is not None

    def input_document(self) -> dict[str, object]:
        return {
            "model_irg_path": str(self.materialized) if self.materialized else None,
            "model_irg_sha256": _sha256_file(self.materialized),
            "model_portrait_path": str(self.portrait) if self.portrait else None,
            "model_portrait_sha256": _sha256_file(self.portrait),
            "irg_asset_path": str(self.asset) if self.asset else None,
            "irg_asset_sha256": _sha256_file(self.asset),
            "irg_diagnostic_axes_path": (
                str(self.diagnostic_axes) if self.diagnostic_axes else None
            ),
            "irg_diagnostic_axes_sha256": _sha256_file(self.diagnostic_axes),
            "irg_method_effects_path": (
                str(self.method_effects) if self.method_effects else None
            ),
            "irg_method_effects_sha256": _sha256_file(self.method_effects),
            "comparison_model_irgs": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in self.comparisons
            ],
        }


@dataclass(frozen=True)
class PipelineIRGContext:
    model_irg_path: Path
    model_family: str
    failure_signatures: tuple[str, ...]
    literature_queries: tuple[str, ...]
    retrieval_projection: dict[str, object]


def resolve_pipeline_irg_inputs(
    *,
    model_irg_path: Path | None,
    model_portrait_path: Path | None,
    irg_asset_path: Path | None,
    irg_diagnostic_axes_path: Path | None,
    irg_method_effects_path: Path | None,
    comparison_model_irg_paths: Sequence[Path],
) -> PipelineIRGInputs:
    """Resolve mutually exclusive IRG input forms without model-specific logic."""

    materialized = _optional_file(model_irg_path, "PIPELINE_MODEL_IRG_INVALID")
    portrait = _optional_file(model_portrait_path, "PIPELINE_MODEL_PORTRAIT_INVALID")
    asset = _optional_file(irg_asset_path, "PIPELINE_IRG_ASSET_INVALID")
    axes = _optional_file(
        irg_diagnostic_axes_path, "PIPELINE_IRG_DIAGNOSTIC_AXES_INVALID"
    )
    effects = _optional_file(
        irg_method_effects_path, "PIPELINE_IRG_METHOD_EFFECTS_INVALID"
    )
    if materialized is not None and any((portrait, asset, axes, effects)):
        raise PipelineIRGError("PIPELINE_IRG_INPUTS_CONFLICT")
    if (portrait is None) != (asset is None):
        raise PipelineIRGError("PIPELINE_IRG_INPUTS_INCOMPLETE")
    if (axes is not None or effects is not None) and portrait is None:
        raise PipelineIRGError("PIPELINE_IRG_INPUTS_INCOMPLETE")
    comparisons = tuple(
        dict.fromkeys(
            _file(path, "PIPELINE_COMPARISON_IRG_INVALID")
            for path in comparison_model_irg_paths
        )
    )
    if comparisons and materialized is None and portrait is None:
        raise PipelineIRGError("PIPELINE_COMPARISON_IRG_REQUIRES_TARGET")
    return PipelineIRGInputs(
        materialized=materialized,
        portrait=portrait,
        asset=asset,
        diagnostic_axes=axes,
        method_effects=effects,
        comparisons=comparisons,
    )


def prepare_pipeline_irg(
    *,
    inputs: PipelineIRGInputs,
    protected_metrics: Sequence[str],
    output_root: Path,
    control_root: Path,
    enable_external_discovery: bool,
    max_results: int,
    timeout_seconds: float,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> PipelineIRGContext:
    """Materialize, compare, and plan discovery for one target model IRG."""

    if not inputs.enabled:
        raise PipelineIRGError("PIPELINE_IRG_INPUT_REQUIRED")
    root = Path(output_root).resolve()
    projection: dict[str, object] = {}
    model_irg_path = inputs.materialized
    try:
        if inputs.portrait is not None and inputs.asset is not None:
            materialization = materialize_model_irg(
                portrait_path=inputs.portrait,
                irg_asset_path=inputs.asset,
                diagnostic_axes_path=inputs.diagnostic_axes,
                method_effects_path=inputs.method_effects,
                output_root=root / "model-irg",
                repo_root=control_root,
                archive_db=archive_db,
                cas_root=cas_root,
            )
            projection["model_irg_materialization"] = materialization
            model_irg_path = Path(str(materialization["model_irg_path"])).resolve(
                strict=True
            )
        if model_irg_path is None:
            raise PipelineIRGError("PIPELINE_IRG_INPUT_REQUIRED")
        if inputs.comparisons:
            projection["probe_evolution"] = plan_probe_evolution_cycle(
                model_irg_paths=(model_irg_path, *inputs.comparisons),
                output_root=root / "probe-evolution",
                repo_root=control_root,
                archive_db=archive_db,
                cas_root=cas_root,
            )
        discovery = prepare_irg_discovery(
            model_irg_path=model_irg_path,
            protected_metrics=protected_metrics,
            output_root=root / "irg-mechanism-discovery",
            control_root=control_root,
            enable_external_discovery=enable_external_discovery,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
    except (
        DiscoveryCoordinatorError,
        ModelIRGMaterializationError,
        ProbeEvolutionCycleError,
        OSError,
        ValueError,
    ) as exc:
        raise PipelineIRGError(f"PIPELINE_IRG_LIFECYCLE_INVALID:{exc}") from exc
    projection["irg_guided"] = discovery.manifest
    return PipelineIRGContext(
        model_irg_path=model_irg_path,
        model_family=discovery.model_family,
        failure_signatures=discovery.failure_signatures,
        literature_queries=discovery.literature_queries,
        retrieval_projection=projection,
    )


def _optional_file(path: Path | None, code: str) -> Path | None:
    return None if path is None else _file(path, code)


def _file(path: Path, code: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise PipelineIRGError(code)
    return value


def _sha256_file(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None
