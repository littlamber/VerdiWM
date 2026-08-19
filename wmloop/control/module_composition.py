"""Resolve admitted module ABIs without importing generated implementation code."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document


class ModuleCompositionError(RuntimeError):
    """A module composition crossed a registry, type, or authority boundary."""


_AUTHORITY_RANK = {"L0": 0, "L1": 1, "L2": 2}
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_CONSTRAINT = re.compile(r"^(>=|<=|>|<|==|=)?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def module_abi_digest(abi: Mapping[str, object]) -> str:
    """Return the semantic digest of one ABI entry."""

    payload = {key: value for key, value in abi.items() if key != "abi_digest"}
    return _sha256(_canonical_json(payload))


def module_registry_digest(registry: Mapping[str, object]) -> str:
    """Return the semantic digest of a module ABI registry."""

    payload = {key: value for key, value in registry.items() if key != "registry_digest"}
    return _sha256(_canonical_json(payload))


def module_composition_receipt_digest(receipt: Mapping[str, object]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    return _sha256(_canonical_json(payload))


def load_module_abi_registry(
    registry_path: Path,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    path = Path(registry_path).expanduser().resolve()
    if not path.is_file():
        raise ModuleCompositionError("MODULE_COMPOSITION_REGISTRY_INVALID")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModuleCompositionError("MODULE_COMPOSITION_REGISTRY_INVALID") from exc
    if not isinstance(payload, dict):
        raise ModuleCompositionError("MODULE_COMPOSITION_REGISTRY_INVALID")
    validate_module_abi_registry(payload, root=root)
    return payload


def validate_module_abi_registry(
    registry: Mapping[str, object],
    *,
    root: Path | None = None,
) -> None:
    """Validate the v2 contract and all semantic cross-field invariants."""

    try:
        validate_document("automatic_module_abi_registry_v2", registry, root=root)
    except ContractValidationError as exc:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_REGISTRY_INVALID:{exc}") from exc
    if registry.get("registry_digest") != module_registry_digest(registry):
        raise ModuleCompositionError("MODULE_COMPOSITION_REGISTRY_DIGEST_MISMATCH")
    rows = registry.get("abis")
    assert isinstance(rows, list)
    abi_ids: set[str] = set()
    versions: set[tuple[str, str]] = set()
    for raw in rows:
        assert isinstance(raw, Mapping)
        abi_id = str(raw["abi_id"])
        version_key = (str(raw["module_id"]), str(raw["abi_version"]))
        if abi_id in abi_ids or version_key in versions:
            raise ModuleCompositionError("MODULE_COMPOSITION_ABI_DUPLICATE")
        abi_ids.add(abi_id)
        versions.add(version_key)
        _validate_abi(raw)


def compose_module_abis(
    *,
    registry_path: Path,
    requested_capabilities: Sequence[str],
    external_ports: Mapping[str, str],
    maximum_authority_level: str,
    expected_registry_digest: str | None = None,
    eligible_abi_ids: Sequence[str] | None = None,
    abi_locks: Mapping[str, Mapping[str, str]] | None = None,
    output_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Resolve a deterministic capability closure and emit its admission receipt."""

    path = Path(registry_path).expanduser().resolve()
    registry = load_module_abi_registry(path, root=root)
    if (
        expected_registry_digest is not None
        and expected_registry_digest != registry["registry_digest"]
    ):
        raise ModuleCompositionError("MODULE_COMPOSITION_REGISTRY_DRIFT")
    requested = tuple(sorted(_unique_nonempty(requested_capabilities, "REQUESTED_CAPABILITY")))
    if not requested:
        raise ModuleCompositionError("MODULE_COMPOSITION_REQUESTED_CAPABILITY_REQUIRED")
    if maximum_authority_level not in _AUTHORITY_RANK:
        raise ModuleCompositionError("MODULE_COMPOSITION_AUTHORITY_INVALID")
    normalized_external = _normalize_external_ports(external_ports)
    rows = registry["abis"]
    assert isinstance(rows, list)
    known_abi_ids = {str(row["abi_id"]) for row in rows if isinstance(row, Mapping)}
    if eligible_abi_ids is None:
        eligible = known_abi_ids
    else:
        eligible = _unique_nonempty(eligible_abi_ids, "ELIGIBLE_ABI")
        unknown = sorted(eligible - known_abi_ids)
        if unknown:
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_ELIGIBLE_ABI_UNKNOWN:{unknown[0]}"
            )
    resolver = _Resolver(
        [
            dict(row)
            for row in rows
            if isinstance(row, Mapping) and str(row["abi_id"]) in eligible
        ],
        maximum_authority_level=maximum_authority_level,
    )
    for capability in requested:
        resolver.resolve_root(capability)
    external_bindings = resolver.bind_external_ports(normalized_external)
    resolver.validate_locks(abi_locks)
    receipt = _composition_receipt(
        registry=registry,
        registry_sha256=_sha256(path.read_bytes()),
        requested_capabilities=requested,
        eligible_abi_ids=tuple(sorted(eligible)),
        maximum_authority_level=maximum_authority_level,
        selected=resolver.topological,
        capability_bindings=resolver.capability_bindings,
        dependency_edges=resolver.dependency_edges,
        external_bindings=external_bindings,
    )
    try:
        validate_document("module_composition_receipt", receipt, root=root)
    except ContractValidationError as exc:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_RECEIPT_INVALID:{exc}") from exc
    if output_path is not None:
        _write_receipt(Path(output_path), receipt)
    return receipt


class _Resolver:
    def __init__(self, abis: Sequence[dict[str, object]], *, maximum_authority_level: str):
        self.abis = tuple(abis)
        self.maximum_authority_level = maximum_authority_level
        self.selected_by_module: dict[str, dict[str, object]] = {}
        self.state: dict[str, str] = {}
        self.topological: list[dict[str, object]] = []
        self.capability_bindings: list[dict[str, str]] = []
        self.dependency_edges: list[dict[str, str]] = []

    def resolve_root(self, capability: str) -> None:
        provider = self._select_provider(capability=capability, dependency=None)
        self._record_binding("composition-root", capability, provider)

    def _select_provider(
        self,
        *,
        capability: str,
        dependency: Mapping[str, object] | None,
    ) -> dict[str, object]:
        module_id = str(dependency["module_id"]) if dependency is not None else None
        constraint = str(dependency["version_constraint"]) if dependency is not None else "*"
        candidates = [
            abi
            for abi in self.abis
            if capability in abi["provides"]
            and (module_id is None or str(abi["module_id"]) == module_id)
            and _constraint_allows(str(abi["abi_version"]), constraint)
        ]
        if not candidates:
            matching_capability = [abi for abi in self.abis if capability in abi["provides"]]
            if matching_capability and module_id is not None:
                raise ModuleCompositionError(
                    f"MODULE_COMPOSITION_ABI_CONSTRAINT_UNSATISFIED:{module_id}:{constraint}"
                )
            raise ModuleCompositionError(f"MODULE_COMPOSITION_MISSING_CAPABILITY:{capability}")
        selected_existing = [
            abi
            for abi in candidates
            if str(abi["module_id"]) in self.selected_by_module
            and self.selected_by_module[str(abi["module_id"])]["abi_id"] == abi["abi_id"]
        ]
        if selected_existing:
            chosen = selected_existing[0]
        else:
            if module_id is not None and module_id in self.selected_by_module:
                raise ModuleCompositionError(
                    f"MODULE_COMPOSITION_ABI_CONSTRAINT_CONFLICT:{module_id}:{constraint}"
                )
            highest = max(_parse_version(str(abi["abi_version"])) for abi in candidates)
            newest = [
                abi
                for abi in candidates
                if _parse_version(str(abi["abi_version"])) == highest
            ]
            chosen = sorted(newest, key=lambda abi: (str(abi["module_id"]), str(abi["abi_id"])))[0]
            chosen_module = str(chosen["module_id"])
            existing = self.selected_by_module.get(chosen_module)
            if existing is not None and existing["abi_id"] != chosen["abi_id"]:
                raise ModuleCompositionError(
                    f"MODULE_COMPOSITION_ABI_CONSTRAINT_CONFLICT:{chosen_module}"
                )
            self.selected_by_module[chosen_module] = chosen
        self._visit(chosen)
        return chosen

    def _visit(self, abi: dict[str, object]) -> None:
        module_id = str(abi["module_id"])
        state = self.state.get(module_id)
        if state == "visiting":
            raise ModuleCompositionError(f"MODULE_COMPOSITION_CYCLE:{module_id}")
        if state == "done":
            return
        if (
            _AUTHORITY_RANK[str(abi["authority_level"])]
            > _AUTHORITY_RANK[self.maximum_authority_level]
        ):
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_AUTHORITY_ESCALATION:{module_id}:{abi['authority_level']}"
            )
        self.state[module_id] = "visiting"
        dependencies = abi["dependencies"]
        assert isinstance(dependencies, list)
        for dependency in sorted(dependencies, key=lambda row: str(row["capability"])):
            assert isinstance(dependency, Mapping)
            capability = str(dependency["capability"])
            provider = self._select_provider(capability=capability, dependency=dependency)
            contract = self._bind_dependency(abi, dependency, provider)
            self._record_binding(module_id, capability, provider)
            self.dependency_edges.append(
                {
                    "consumer_module_id": module_id,
                    "required_capability": capability,
                    "provider_module_id": str(provider["module_id"]),
                    "consumer_input_port": str(dependency["input_port"]),
                    "provider_output_port": str(dependency["output_port"]),
                    "contract": contract,
                }
            )
        self.state[module_id] = "done"
        self.topological.append(abi)

    def _bind_dependency(
        self,
        consumer: Mapping[str, object],
        dependency: Mapping[str, object],
        provider: Mapping[str, object],
    ) -> str:
        consumer_inputs = _ports_by_name(consumer["typed_inputs"])
        provider_outputs = _ports_by_name(provider["typed_outputs"])
        input_name = str(dependency["input_port"])
        output_name = str(dependency["output_port"])
        if input_name not in consumer_inputs:
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_MISSING_PORT:{consumer['module_id']}.{input_name}"
            )
        if output_name not in provider_outputs:
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_MISSING_PORT:{provider['module_id']}.{output_name}"
            )
        input_contract = str(consumer_inputs[input_name]["contract"])
        output_contract = str(provider_outputs[output_name]["contract"])
        if input_contract != output_contract:
            raise ModuleCompositionError(
                "MODULE_COMPOSITION_PORT_CONTRACT_MISMATCH:"
                f"{consumer['module_id']}.{input_name}:{provider['module_id']}.{output_name}"
            )
        return input_contract

    def _record_binding(
        self,
        requester_id: str,
        capability: str,
        provider: Mapping[str, object],
    ) -> None:
        row = {
            "requester_id": requester_id,
            "capability": capability,
            "provider_module_id": str(provider["module_id"]),
            "provider_abi_id": str(provider["abi_id"]),
        }
        if row not in self.capability_bindings:
            self.capability_bindings.append(row)

    def bind_external_ports(self, external_ports: Mapping[str, str]) -> list[dict[str, str]]:
        required: dict[str, str] = {}
        for abi in self.topological:
            for port in abi["typed_inputs"]:
                assert isinstance(port, Mapping)
                if port["source"] != "external":
                    continue
                key = f"{abi['module_id']}.{port['name']}"
                required[key] = str(port["contract"])
        missing = sorted(set(required) - set(external_ports))
        if missing:
            raise ModuleCompositionError(f"MODULE_COMPOSITION_MISSING_PORT:{missing[0]}")
        extra = sorted(set(external_ports) - set(required))
        if extra:
            raise ModuleCompositionError(f"MODULE_COMPOSITION_UNUSED_EXTERNAL_PORT:{extra[0]}")
        bindings: list[dict[str, str]] = []
        for key in sorted(required):
            if external_ports[key] != required[key]:
                raise ModuleCompositionError(
                    f"MODULE_COMPOSITION_PORT_CONTRACT_MISMATCH:{key}"
                )
            module_id, input_port = key.rsplit(".", 1)
            bindings.append(
                {"module_id": module_id, "input_port": input_port, "contract": required[key]}
            )
        return bindings

    def validate_locks(self, locks: Mapping[str, Mapping[str, str]] | None) -> None:
        if locks is None:
            return
        if set(locks) != set(self.selected_by_module):
            raise ModuleCompositionError("MODULE_COMPOSITION_ABI_DRIFT:lock-set")
        for module_id, abi in self.selected_by_module.items():
            lock = locks[module_id]
            expected = {
                "abi_id": str(abi["abi_id"]),
                "abi_version": str(abi["abi_version"]),
                "abi_digest": str(abi["abi_digest"]),
            }
            if any(str(lock.get(key)) != value for key, value in expected.items()):
                raise ModuleCompositionError(f"MODULE_COMPOSITION_ABI_DRIFT:{module_id}")


def _validate_abi(abi: Mapping[str, object]) -> None:
    if abi.get("abi_digest") != module_abi_digest(abi):
        raise ModuleCompositionError(f"MODULE_COMPOSITION_ABI_DRIFT:{abi['abi_id']}")
    _parse_version(str(abi["abi_version"]))
    for field in (
        "provides",
        "requires",
        "positional_parameters",
        "keyword_parameters",
        "required_model_capabilities",
        "required_hooks",
        "allowed_imports",
        "forbidden_changed_paths",
    ):
        values = abi[field]
        assert isinstance(values, list)
        if len(values) != len({str(value) for value in values}):
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_ABI_DUPLICATE_FIELD:{abi['abi_id']}:{field}"
            )
    inputs = _ports_by_name(abi["typed_inputs"])
    outputs = _ports_by_name(abi["typed_outputs"])
    if len(inputs) != len(abi["typed_inputs"]) or len(outputs) != len(abi["typed_outputs"]):
        raise ModuleCompositionError(f"MODULE_COMPOSITION_PORT_DUPLICATE:{abi['abi_id']}")
    dependencies = abi["dependencies"]
    assert isinstance(dependencies, list)
    dependency_capabilities = [str(row["capability"]) for row in dependencies]
    if len(dependency_capabilities) != len(set(dependency_capabilities)):
        raise ModuleCompositionError(f"MODULE_COMPOSITION_DEPENDENCY_DUPLICATE:{abi['abi_id']}")
    if set(dependency_capabilities) != {str(value) for value in abi["requires"]}:
        raise ModuleCompositionError(
            f"MODULE_COMPOSITION_CAPABILITY_CLOSURE_INVALID:{abi['abi_id']}"
        )
    dependency_inputs: set[str] = set()
    for dependency in dependencies:
        assert isinstance(dependency, Mapping)
        input_name = str(dependency["input_port"])
        if input_name not in inputs or inputs[input_name]["source"] != "dependency":
            raise ModuleCompositionError(
                f"MODULE_COMPOSITION_DEPENDENCY_PORT_INVALID:{abi['abi_id']}:{input_name}"
            )
        dependency_inputs.add(input_name)
        _validate_constraint(str(dependency["version_constraint"]))
    declared_dependency_inputs = {
        name for name, port in inputs.items() if port["source"] == "dependency"
    }
    if dependency_inputs != declared_dependency_inputs:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_DEPENDENCY_PORT_INVALID:{abi['abi_id']}")
    admission_suite = abi["admission_suite"]
    assert isinstance(admission_suite, Mapping)
    if admission_suite["evaluator_binding"] != abi["evaluator"]:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_ADMISSION_BINDING_DRIFT:{abi['abi_id']}")
    if admission_suite["test_template_binding"] != abi["test_template"]:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_ADMISSION_BINDING_DRIFT:{abi['abi_id']}")
    portability = abi["portability"]
    assert isinstance(portability, Mapping)
    if str(abi["model_family"]) not in portability["model_family_scope"]:
        raise ModuleCompositionError(
            f"MODULE_COMPOSITION_PORTABILITY_SCOPE_INVALID:{abi['abi_id']}"
        )


def _composition_receipt(
    *,
    registry: Mapping[str, object],
    registry_sha256: str,
    requested_capabilities: Sequence[str],
    eligible_abi_ids: Sequence[str],
    maximum_authority_level: str,
    selected: Sequence[Mapping[str, object]],
    capability_bindings: Sequence[Mapping[str, str]],
    dependency_edges: Sequence[Mapping[str, str]],
    external_bindings: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    selected_rows = [
        {
            "module_id": abi["module_id"],
            "abi_id": abi["abi_id"],
            "abi_version": abi["abi_version"],
            "abi_digest": abi["abi_digest"],
            "provides": sorted(str(value) for value in abi["provides"]),
            "requires": sorted(str(value) for value in abi["requires"]),
            "authority_level": abi["authority_level"],
            "side_effect_class": abi["side_effect_class"],
            "admission_suite_id": abi["admission_suite"]["suite_id"],
            "cost_model": dict(abi["cost_model"]),
            "portable": abi["portability"]["portable"],
            "license_spdx_id": abi["license"]["spdx_id"],
        }
        for abi in selected
    ]
    authority = max(
        (str(abi["authority_level"]) for abi in selected),
        key=lambda level: _AUTHORITY_RANK[level],
    )
    costs = [abi["cost_model"] for abi in selected]
    composition_cost = {
        "cpu_seconds_upper_bound": sum(float(cost["cpu_seconds_upper_bound"]) for cost in costs),
        "gpu_hours_upper_bound": sum(float(cost["gpu_hours_upper_bound"]) for cost in costs),
        "peak_memory_bytes_upper_bound": max(
            int(cost["peak_memory_bytes_upper_bound"]) for cost in costs
        ),
    }
    sorted_bindings = sorted(
        (dict(row) for row in capability_bindings),
        key=lambda row: (row["requester_id"], row["capability"], row["provider_module_id"]),
    )
    sorted_edges = sorted(
        (dict(row) for row in dependency_edges),
        key=lambda row: (row["consumer_module_id"], row["required_capability"]),
    )
    identity = {
        "registry_digest": registry["registry_digest"],
        "requested_capabilities": list(requested_capabilities),
        "eligible_abi_ids": list(eligible_abi_ids),
        "maximum_authority_level": maximum_authority_level,
        "selected_modules": [
            {
                "module_id": row["module_id"],
                "abi_id": row["abi_id"],
                "abi_version": row["abi_version"],
                "abi_digest": row["abi_digest"],
            }
            for row in selected_rows
        ],
        "capability_bindings": sorted_bindings,
        "dependency_edges": sorted_edges,
        "external_bindings": list(external_bindings),
    }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-module-composition-receipt",
        "state": "admitted",
        "composition_id": f"composition-{_sha256(_canonical_json(identity))[:24]}",
        "receipt_digest": "",
        "registry_id": registry["registry_id"],
        "registry_digest": registry["registry_digest"],
        "registry_sha256": registry_sha256,
        "requested_capabilities": list(requested_capabilities),
        "eligible_abi_ids": list(eligible_abi_ids),
        "maximum_authority_level": maximum_authority_level,
        "effective_authority_level": authority,
        "selected_modules": selected_rows,
        "capability_bindings": sorted_bindings,
        "dependency_edges": sorted_edges,
        "external_bindings": list(external_bindings),
        "composition_cost": composition_cost,
        "side_effect_classes": sorted({str(abi["side_effect_class"]) for abi in selected}),
        "side_effects": {
            "generated_module_imported": False,
            "evaluator_selection_authority": False,
            "gpu_budget_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "This receipt admits only ABI compatibility and capability closure. "
            "Execution, evaluator selection, GPU allocation, scientific verdicts, and promotion "
            "remain Kernel-owned."
        ),
    }
    receipt["receipt_digest"] = module_composition_receipt_digest(receipt)
    return receipt


def _ports_by_name(value: object) -> dict[str, Mapping[str, object]]:
    assert isinstance(value, list)
    return {
        str(port["name"]): port
        for port in value
        if isinstance(port, Mapping)
    }


def _normalize_external_ports(external_ports: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_contract in external_ports.items():
        if not isinstance(raw_key, str) or not isinstance(raw_contract, str):
            raise ModuleCompositionError("MODULE_COMPOSITION_EXTERNAL_PORT_INVALID")
        key = raw_key
        contract = raw_contract
        if not key or "." not in key or not contract or key in normalized:
            raise ModuleCompositionError("MODULE_COMPOSITION_EXTERNAL_PORT_INVALID")
        normalized[key] = contract
    return normalized


def _unique_nonempty(values: Sequence[str], field: str) -> set[str]:
    if isinstance(values, (str, bytes)) or any(not isinstance(value, str) for value in values):
        raise ModuleCompositionError(f"MODULE_COMPOSITION_{field}_INVALID")
    normalized = list(values)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ModuleCompositionError(f"MODULE_COMPOSITION_{field}_INVALID")
    return set(normalized)


def _parse_version(value: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(value)
    if match is None:
        raise ModuleCompositionError(f"MODULE_COMPOSITION_ABI_VERSION_INVALID:{value}")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _validate_constraint(constraint: str) -> None:
    if constraint == "*":
        return
    tokens = [token.strip() for token in constraint.split(",")]
    if not tokens or any(not token or _CONSTRAINT.fullmatch(token) is None for token in tokens):
        raise ModuleCompositionError(
            f"MODULE_COMPOSITION_ABI_VERSION_CONSTRAINT_INVALID:{constraint}"
        )


def _constraint_allows(version: str, constraint: str) -> bool:
    _validate_constraint(constraint)
    if constraint == "*":
        return True
    parsed = _parse_version(version)
    for token in (part.strip() for part in constraint.split(",")):
        match = _CONSTRAINT.fullmatch(token)
        assert match is not None
        operator = match.group(1) or "=="
        expected = tuple(int(match.group(index)) for index in range(2, 5))
        if operator in {"=", "=="} and parsed != expected:
            return False
        if operator == ">=" and parsed < expected:
            return False
        if operator == "<=" and parsed > expected:
            return False
        if operator == ">" and parsed <= expected:
            return False
        if operator == "<" and parsed >= expected:
            return False
    return True


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    destination = path.expanduser().resolve()
    payload = _canonical_json(receipt) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return
        raise ModuleCompositionError("MODULE_COMPOSITION_OUTPUT_CONFLICT")
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if not destination.is_file() or destination.read_bytes() != payload:
            raise ModuleCompositionError("MODULE_COMPOSITION_OUTPUT_CONFLICT")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
