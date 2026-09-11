"""Bind method implementation checks and composition hypotheses to real files.

These are requirements for later target-side calibration. Declared tests do not
constitute execution evidence, faithful reproduction, or a positive effect.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document


class MethodRealizationError(ValueError):
    """A proposal omits or contradicts its implementation validation contract."""


def validate_realization(method: Mapping[str, object], *, root: Path | None = None) -> None:
    composition = method.get('composition')
    if composition is not None:
        try:
            validate_document('open_method_composition', composition, root=root)
        except ContractValidationError as exc:
            raise MethodRealizationError(f'METHOD_COMPOSITION_INVALID:{exc}') from exc
        if len(set(composition['component_method_ids'])) != 2:
            raise MethodRealizationError('METHOD_COMPOSITION_DISTINCT_COMPONENTS_REQUIRED')
        for field in ('complementarity', 'predicted_joint_effect', 'conflict_resolution'):
            if not composition[field].strip():
                raise MethodRealizationError('METHOD_COMPOSITION_EXPLANATION_EMPTY')
        if any(not value.strip() for value in composition['anti_conditions']):
            raise MethodRealizationError('METHOD_COMPOSITION_EXPLANATION_EMPTY')
        if method['target_mapping']['mapping_state'] != 'composition_candidate':
            raise MethodRealizationError('METHOD_COMPOSITION_MAPPING_MISMATCH')
    realization = method.get('implementation_validation')
    if realization is None:
        return  # Historical Method IR remains readable.
    try:
        validate_document('method_implementation_validation', realization, root=root)
    except ContractValidationError as exc:
        raise MethodRealizationError(f'METHOD_IMPLEMENTATION_VALIDATION_INVALID:{exc}') from exc
    role = method.get('study_role')
    required = {'no_future_leakage'}
    required |= (
        {'control_equivalence'}
        if role == 'baseline'
        else {'hook_execution', 'ablation_effect'}
    )
    if method['training']['mode'] == 'training':
        required |= {'optimizer_binding', 'parameter_update', 'train_infer_parity'}
    if realization['stateful']:
        required.add('state_lifecycle')
    kinds = [check['kind'] for check in realization['checks']]
    if len(kinds) != len(set(kinds)) or required - set(kinds):
        raise MethodRealizationError('METHOD_IMPLEMENTATION_CHECKS_INCOMPLETE')
    for check in realization['checks']:
        if not check['observable'].strip() or not check['failure_condition'].strip():
            raise MethodRealizationError('METHOD_IMPLEMENTATION_CHECK_EMPTY')


def validate_realization_files(
    method: Mapping[str, object], files: Sequence[Mapping[str, object]],
    tests: Sequence[Mapping[str, object]], *, root: Path | None = None,
) -> None:
    if method.get('implementation_validation') is None:
        raise MethodRealizationError('METHOD_IMPLEMENTATION_VALIDATION_REQUIRED')
    validate_realization(method, root=root)
    realization = method['implementation_validation']
    implementations = {row['relative_path'] for row in files if row['role'] == 'implementation' and str(row['content_utf8']).strip()}
    if not set(realization['implementation_files']).issubset(implementations):
        raise MethodRealizationError('METHOD_IMPLEMENTATION_FILES_UNBOUND')
    names = [row['name'] for row in tests]
    if len(names) != len(set(names)):
        raise MethodRealizationError('METHOD_IMPLEMENTATION_TEST_NAMES_DUPLICATE')
    if any(check['test_name'] not in names for check in realization['checks']):
        raise MethodRealizationError('METHOD_IMPLEMENTATION_TEST_UNBOUND')
    mapping = method['target_mapping']['mapping_state']
    novelty = method.get('mechanism_hypothesis', {}).get('novelty_status')
    if (mapping == 'composition_candidate' or novelty == 'composition') and method.get('composition') is None:
        raise MethodRealizationError('METHOD_COMPOSITION_CONTRACT_REQUIRED')
