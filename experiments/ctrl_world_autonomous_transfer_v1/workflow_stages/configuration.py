"""Configuration stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adaptive_observation import AdaptiveObservationError, load_observation_abi_registry
from wmloop.control.module_composition import ModuleCompositionError, load_module_abi_registry
from .planning import _bound_portable_knowledge_graph, _load_bound_hypothesis_batch, _load_resource_policy
from .common import AutonomousTransferWorkflowError, _canonical_bytes, _load, _load_bound_portrait, _paths, _require_directory, _require_file


def config_digest(config: Mapping[str, object]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_digest"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def load_and_validate_config(path: Path, *, project_root: Path) -> dict[str, object]:
    config_path = _require_file(path, "AUTONOMOUS_CONFIG_INVALID")
    config = _load(config_path, "AUTONOMOUS_CONFIG_INVALID")
    try:
        validate_document("ctrl_world_autonomous_transfer_loop", config, root=project_root)
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_CONFIG_SCHEMA_INVALID:{exc}"
        ) from exc
    if config.get("config_digest") != config_digest(config):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CONFIG_DIGEST_MISMATCH")
    gpu_indices = config["gpu_indices"]
    assert isinstance(gpu_indices, list)
    if (
        len(gpu_indices) > 8
        or len(gpu_indices) != len(set(gpu_indices))
        or int(config["max_parallel_gpu_jobs"]) > len(gpu_indices)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GPU_PARALLELISM_INVALID")
    if isinstance(config.get("portfolio_planning"), Mapping) and not isinstance(
        config.get("resource_portfolio"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_PORTFOLIO_REQUIRED"
        )
    if isinstance(config.get("closed_loop"), Mapping) and not isinstance(
        config.get("portrait_gate"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CLOSED_LOOP_PORTRAIT_GATE_REQUIRED"
        )
    if isinstance(config.get("observation_planning"), Mapping) != isinstance(
        config.get("observation_execution"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_PAIR_REQUIRED"
        )
    if isinstance(config.get("resource_portfolio"), Mapping):
        _load_resource_policy(config, project_root=project_root)
    _validate_local_method_discovery(config, project_root=project_root)
    _validate_metric_evolution(config, project_root=project_root)
    _validate_paths(config, project_root=project_root)
    graph_root = Path(str(config["knowledge_graph_root"])).expanduser().resolve()
    if graph_root == project_root or project_root in graph_root.parents:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GRAPH_INSIDE_SOURCE")
    portable_graph_value = config.get("portable_knowledge_root")
    if portable_graph_value is not None:
        portable_graph_root = Path(str(portable_graph_value)).expanduser().resolve()
        if (
            portable_graph_root == project_root
            or project_root in portable_graph_root.parents
            or portable_graph_root == graph_root
            or graph_root in portable_graph_root.parents
            or portable_graph_root in graph_root.parents
        ):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_GRAPH_ROOT_INVALID")
    portable_records_value = config.get("portable_knowledge_records_root")
    if portable_records_value is not None:
        portable_records_root = Path(str(portable_records_value)).expanduser().resolve()
        if portable_records_root == project_root or project_root in portable_records_root.parents:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INSIDE_SOURCE")
    return config


def _validate_local_method_discovery(
    config: Mapping[str, object], *, project_root: Path
) -> None:
    settings = config.get("local_method_discovery")
    if settings is None:
        return
    if not isinstance(settings, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_DISCOVERY_INVALID")
    if settings.get("enabled") is not True:
        return
    intake_path = _require_file(
        Path(str(settings.get("intake_config") or "")),
        "AUTONOMOUS_LOCAL_METHOD_INTAKE_CONFIG_INVALID",
    )
    profiles = settings.get("materializer_profiles")
    materializers = config.get("materializers")
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(value, str) or not value for value in profiles)
        or len(set(profiles)) != len(profiles)
        or not isinstance(materializers, Mapping)
        or any(str(value) not in materializers for value in profiles)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_PROFILE_BINDING_INVALID")
    maximum = settings.get("max_methods")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_MAX_INVALID")


def _validate_metric_evolution(
    config: Mapping[str, object], *, project_root: Path
) -> None:
    settings = config.get("metric_evolution")
    if settings is None:
        return
    if not isinstance(settings, Mapping) or settings.get("mode") != "shadow_only":
        raise AutonomousTransferWorkflowError("AUTONOMOUS_METRIC_EVOLUTION_SHADOW_ONLY_REQUIRED")
    protected = settings.get("protected_metric_ids")
    candidates = settings.get("candidate_metrics")
    if (
        not isinstance(protected, list)
        or any(not isinstance(value, str) or not value for value in protected)
        or not isinstance(candidates, list)
        or any(not isinstance(value, Mapping) for value in candidates)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_METRIC_EVOLUTION_CONFIG_INVALID")
    for candidate in candidates:
        try:
            validate_document("metric_adequacy", candidate, root=project_root)
        except ContractValidationError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_METRIC_EVOLUTION_CONTRACT_INVALID:{exc}"
            ) from exc


def _validate_paths(config: Mapping[str, object], *, project_root: Path) -> None:
    paths = _paths(config)
    if paths["project_root"] != project_root.resolve():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PROJECT_ROOT_MISMATCH")
    for name in (
        "ctrl_world_root",
        "dataset_root",
        "svd_model",
        "clip_model",
    ):
        _require_directory(paths[name], f"AUTONOMOUS_PATH_INVALID:{name}")
    for name in (
        "data_stat",
        "checkpoint",
        "runtime_python",
        "intake_config",
        "contract",
        "evaluator",
        "base_evaluator",
        "screen_baseline",
        "confirm_baseline",
        "verifier_policy",
    ):
        _require_file(paths[name], f"AUTONOMOUS_PATH_INVALID:{name}")
    if config.get("source_bundle_path") is not None:
        _require_file(
            Path(str(config["source_bundle_path"])),
            "AUTONOMOUS_SOURCE_BUNDLE_INVALID",
        )
    materializers = config["materializers"]
    assert isinstance(materializers, Mapping)
    for profile, row in materializers.items():
        if not isinstance(row, Mapping):
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MATERIALIZER_INVALID:{profile}"
            )
        _require_file(
            Path(str(row["plan_template"])),
            f"AUTONOMOUS_MATERIALIZER_TEMPLATE_INVALID:{profile}",
        )
        if row.get("evaluator") is not None:
            _require_file(
                Path(str(row["evaluator"])),
                f"AUTONOMOUS_MATERIALIZER_EVALUATOR_INVALID:{profile}",
            )
        if row.get("trainer") is not None:
            _require_file(
                Path(str(row["trainer"])),
                f"AUTONOMOUS_MATERIALIZER_TRAINER_INVALID:{profile}",
            )
        training = row.get("training")
        if training is not None and not isinstance(training, Mapping):
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MATERIALIZER_TRAINING_INVALID:{profile}"
            )
    automatic = config.get("automatic_module_generation")
    if automatic is not None:
        if not isinstance(automatic, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_CONFIG_INVALID"
            )
        _require_file(
            Path(str(automatic["model_capability_ir"])),
            "AUTONOMOUS_AUTOMATIC_MODULE_CAPABILITY_IR_INVALID",
        )
        _require_file(
            Path(str(automatic["abi_registry"])),
            "AUTONOMOUS_AUTOMATIC_MODULE_ABI_REGISTRY_INVALID",
        )
        if not isinstance(automatic.get("llm_adapter"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_ADAPTER_INVALID"
            )
        if not isinstance(config.get("portrait_gate"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_PORTRAIT_GATE_REQUIRED"
            )
        if not isinstance(config.get("gap_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_GAP_PLANNING_REQUIRED"
            )
    open_generation = config.get("open_method_generation")
    if open_generation is not None:
        if not isinstance(open_generation, Mapping) or not isinstance(
            open_generation.get("llm_adapter"), Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OPEN_METHOD_CONFIG_INVALID"
            )
        if not isinstance(config.get("portrait_gate"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OPEN_METHOD_PORTRAIT_GATE_REQUIRED"
            )
    gate = config.get("portrait_gate")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_GATE_INVALID")
        _load_bound_portrait(gate, project_root=project_root)
        if not isinstance(config.get("gap_planning"), Mapping) and not isinstance(
            config.get("open_method_generation"), Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTRAIT_GAP_PLANNING_REQUIRED"
            )
    gap_planning = config.get("gap_planning")
    if gap_planning is not None:
        if not isinstance(gap_planning, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_PLANNING_INVALID")
        registry_path = _require_file(
            Path(str(gap_planning.get("abi_registry") or "")),
            "AUTONOMOUS_GAP_ABI_REGISTRY_INVALID",
        )
        try:
            registry = load_module_abi_registry(registry_path, root=project_root)
        except ModuleCompositionError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_GAP_ABI_REGISTRY_INVALID:{exc}"
            ) from exc
        if registry["registry_digest"] != gap_planning.get("abi_registry_digest"):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_ABI_REGISTRY_DIGEST_MISMATCH"
            )
        profiles = gap_planning.get("profile_requirements")
        assert isinstance(profiles, list)
        profile_ids = [str(row["profile_id"]) for row in profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_PROFILE_DUPLICATE"
            )
        _bound_portable_knowledge_graph(gap_planning)
        if not isinstance(config.get("portfolio_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_PORTFOLIO_PLANNING_REQUIRED"
            )
    portfolio_planning = config.get("portfolio_planning")
    if portfolio_planning is not None:
        if not isinstance(portfolio_planning, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTFOLIO_PLANNING_INVALID"
            )
        if not isinstance(config.get("gap_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTFOLIO_GAP_PLANNING_REQUIRED"
            )
        _load_bound_hypothesis_batch(
            portfolio_planning,
            project_root=project_root,
        )
    observation = config.get("observation_planning")
    if observation is not None:
        if not isinstance(observation, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PLANNING_INVALID"
            )
        registry_path = _require_file(
            Path(str(observation.get("abi_registry") or "")),
            "AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID",
        )
        try:
            observation_registry = load_observation_abi_registry(
                registry_path, root=project_root
            )
        except AdaptiveObservationError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID:{exc}"
            ) from exc
        _validate_observation_execution_config(
            config,
            registry=observation_registry,
            project_root=project_root,
        )
    closed_loop = config.get("closed_loop")
    if closed_loop is not None:
        if not isinstance(closed_loop, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_INVALID")
        archive_value = closed_loop.get("archive_root")
        if not isinstance(archive_value, str) or not archive_value.strip():
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_ARCHIVE_INVALID")
        archive_root = Path(archive_value).expanduser().resolve()
        if archive_root == project_root or project_root in archive_root.parents:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_ARCHIVE_INSIDE_SOURCE")
        active_path = closed_loop.get("active_portrait_path")
        active_hash = closed_loop.get("active_portrait_sha256")
        if (active_path is None) != (active_hash is None):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_BINDING_INVALID")
        if active_path is not None:
            if not isinstance(active_path, str) or not active_path.strip():
                raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_PATH_INVALID")
            active = Path(active_path).expanduser().resolve()
            if active == project_root or project_root in active.parents:
                raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_INSIDE_SOURCE")


def _validate_observation_execution_config(
    config: Mapping[str, object],
    *,
    registry: Mapping[str, object],
    project_root: Path,
) -> None:
    execution = config.get("observation_execution")
    if not isinstance(execution, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_REQUIRED"
        )
    archive_root = Path(str(execution.get("archive_root") or "")).expanduser()
    if archive_root.is_symlink():
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INVALID"
        )
    archive_root = archive_root.resolve()
    if archive_root == project_root or project_root in archive_root.parents:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INSIDE_SOURCE"
        )
    adapters = execution.get("adapters")
    runtime_bindings = execution.get("runtime_bindings")
    shadow_adapter = execution.get("shadow_llm_adapter")
    if (
        not isinstance(adapters, Mapping)
        or not isinstance(runtime_bindings, Mapping)
        or not isinstance(shadow_adapter, Mapping)
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_CONFIG_INVALID"
        )
    rows = [row for row in registry["abis"] if isinstance(row, Mapping)]
    by_id = {str(row["abi_id"]): row for row in rows}
    admitted = {
        abi_id
        for abi_id, row in by_id.items()
        if row.get("admission_state") == "admitted"
    }
    if set(str(value) for value in adapters) != admitted:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ADAPTER_COVERAGE_INVALID"
        )
    if any(str(value) not in admitted for value in runtime_bindings):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_RUNTIME_BINDING_UNKNOWN"
        )
    for abi_id in sorted(admitted):
        adapter = adapters[abi_id]
        assert isinstance(adapter, Mapping)
        abi = by_id[abi_id]
        if (
            abi.get("execution_mode") == "cpu_only"
            and adapter.get("resource_class") != "cpu_only"
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_READ_ONLY_GPU_FORBIDDEN:" + abi_id
            )
        command = adapter.get("command")
        if not isinstance(command, list) or not any(
            "{request_path}" in str(value) for value in command
        ) or not any("{response_path}" in str(value) for value in command):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_ADAPTER_PLACEHOLDERS_REQUIRED:" + abi_id
            )
