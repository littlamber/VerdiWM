"""Compile an open-method A/B study into four isolated candidate bundles.

The shared experimental binding is kernel input, outside the LLM proposals.
This emits calibration work; it does not execute generated tests or allocate GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping, Sequence

from wmloop.control.open_method_pipeline import compile_open_method_proposal, _normalize_method_ir
from wmloop.storage import atomic_write, canonical_bytes, checked_path

ROLES = ('baseline', 'source_only', 'target_only', 'combined')


class OpenMethodStudyError(ValueError):
    """A composition lacks a complete, comparable, budgeted experiment."""


def compile_open_method_study(
    *, proposals: Mapping[str, Mapping[str, object]],
    base_revision: Mapping[str, object], experiment_binding: Mapping[str, str],
    target_portrait_binding: Mapping[str, str], seeds: Sequence[int],
    estimated_gpu_hours_per_seed: Mapping[str, float], budget_gpu_hours: float,
    output_root: Path, project_root: Path,
) -> dict:
    """Compile all arms atomically; no partial study receives eligibility."""
    if set(proposals) != set(ROLES) or set(estimated_gpu_hours_per_seed) != set(ROLES):
        raise OpenMethodStudyError('OPEN_STUDY_FOUR_ARMS_REQUIRED')
    fields = {'checkpoint_digest','train_split_digest','selection_split_digest','confirmation_split_digest','verifier_digest'}
    if set(experiment_binding) != fields:
        raise OpenMethodStudyError('OPEN_STUDY_EXPERIMENT_BINDING_REQUIRED')
    for value in experiment_binding.values():
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise OpenMethodStudyError('OPEN_STUDY_DIGEST_INVALID')
    if len({experiment_binding[k] for k in ('train_split_digest','selection_split_digest','confirmation_split_digest')}) != 3:
        raise OpenMethodStudyError('OPEN_STUDY_DISTINCT_SPLITS_REQUIRED')
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise OpenMethodStudyError('OPEN_STUDY_SEEDS_INVALID')
    costs = list(estimated_gpu_hours_per_seed.values())
    if any(type(v) not in (float,int) or not math.isfinite(v) or v <= 0 for v in [*costs,budget_gpu_hours]):
        raise OpenMethodStudyError('OPEN_STUDY_BUDGET_INVALID')
    estimate = sum(costs)*len(seeds)
    if estimate > budget_gpu_hours:
        raise OpenMethodStudyError('OPEN_STUDY_FOUR_ARM_BUDGET_EXCEEDED')
    root = Path(project_root).resolve()
    methods = {role:_normalize_method_ir(proposals[role]['method_ir'],root=root) for role in ROLES}
    expected = [methods['source_only']['method_id'], methods['target_only']['method_id']]
    composition = methods['combined'].get('composition')
    if expected[0] == expected[1] or not isinstance(composition, Mapping) or composition.get('component_method_ids') != expected:
        raise OpenMethodStudyError('OPEN_STUDY_COMPONENT_BINDING_MISMATCH')
    if len({method['method_id'] for method in methods.values()}) != len(ROLES):
        raise OpenMethodStudyError('OPEN_STUDY_DISTINCT_ARMS_REQUIRED')
    if any(methods[role].get('composition') is not None for role in ROLES[:3]):
        raise OpenMethodStudyError('OPEN_STUDY_NESTED_COMPOSITION_UNSUPPORTED')
    primary = methods['baseline']['falsification']['primary_metrics']
    protected = methods['baseline']['falsification']['protected_metrics']
    for method in methods.values():
        if method.get('target_portrait_binding') != dict(target_portrait_binding):
            raise OpenMethodStudyError('OPEN_STUDY_PORTRAIT_BINDING_MISMATCH')
        if (method['falsification']['primary_metrics'] != primary or method['falsification']['protected_metrics'] != protected):
            raise OpenMethodStudyError('OPEN_STUDY_METRIC_MISMATCH')
    destination = checked_path(output_root, code='OPEN_STUDY_PATH_INVALID', error=OpenMethodStudyError)
    if destination == root or root in destination.parents or destination in root.parents or destination.exists():
        raise OpenMethodStudyError('OPEN_STUDY_OUTPUT_INVALID')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.',dir=destination.parent))
    try:
        arms = []
        for role in ROLES:
            manifest = compile_open_method_proposal(
                proposal=proposals[role], base_revision=base_revision,
                expected_portrait_binding=target_portrait_binding,
                output_root=temporary/role, project_root=root,
            )
            if manifest['state'] != 'ready_for_calibration':
                raise OpenMethodStudyError(f'OPEN_STUDY_ARM_NOT_READY:{role}')
            arms.append({'role':role, 'method_id':manifest['method_id'],
                         'bundle_path':role, 'overlay_id':manifest['overlay_id'],
                         'estimated_gpu_hours_per_seed':estimated_gpu_hours_per_seed[role]})
        files = {path.relative_to(temporary).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(temporary.rglob('*')) if path.is_file()}
        study = {
            'schema_version':1,'artifact_type':'verdiwm-open-method-study',
            'state':'ready_for_calibration', 'arms':arms,
            'base_revision':dict(base_revision),'experiment_binding':dict(experiment_binding),
            'target_portrait_binding':dict(target_portrait_binding),
            'paired_seeds':list(seeds),'primary_metrics':primary,'protected_metrics':protected,
            'budget_gpu_hours':budget_gpu_hours,'estimated_total_gpu_hours':estimate,
            'file_sha256':files,
            'required_contrasts':['source_only-baseline','target_only-baseline','combined-baseline',
                                  'combined-source_only','combined-target_only','combined-source_only-target_only+baseline'],
            'decision_requirements':[
                'Execute target-side implementation checks before effect trials.',
                'Verify episode disjointness from actual manifests; distinct digests alone do not prove it.',
                'Use paired evaluation units and seeds; cluster uncertainty by independent episode.',
                'Compare combined against both components, then confirm independently with protected metrics.',
                'Report additive interaction separately from improvement over the best single method.',
                'Include training and inference costs; require matched-cost controls for efficiency claims.',
                'Retain null, harmful, invalid and failed outcomes; incomplete arms cannot establish synergy.',
            ],
            'authority':{'gpu_scheduling':False,'promotion':False},
            'claim_boundary':'Four compiled arms and declared checks only. Calibration, actual split validation, budget admission and frozen verifier settlement remain required.',
        }
        study['study_id']='open-study-'+hashlib.sha256(canonical_bytes(study)).hexdigest()[:24]
        atomic_write(temporary/'study.json',canonical_bytes(study)+b'\n')
        os.replace(temporary,destination)
        return study
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    if 'output_root' in request or 'project_root' in request:
        parser.error('request cannot override output or project root')
    study=compile_open_method_study(**request,output_root=args.output,project_root=Path(__file__).resolve().parents[2])
    print(json.dumps({'study_id':study['study_id'],'state':study['state'],'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
