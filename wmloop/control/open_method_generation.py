"""Run the configured LLM adapter and compile its open-method implementation.

Generated code is retained in an isolated bundle for target-side calibration.
This bridge runs the trusted provider adapter, never the generated candidate.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from wmloop.contracts import validate_document
from wmloop.control.open_method_pipeline import compile_open_method_proposal
from wmloop.execute.llm_task_adapter import run_llm_task, task_request_digest
from wmloop.storage import atomic_write, canonical_bytes, checked_path


class OpenMethodGenerationError(ValueError):
    """Generated code or provenance does not match its research request."""


def generate_open_method(
    *, request: Mapping[str, object], adapter: Mapping[str, object],
    base_revision: Mapping[str, object], output_root: Path, project_root: Path,
) -> dict[str, object]:
    """Execute one bounded generation attempt and preserve success or failure.

    A fresh output directory is required. Retries use a new directory, keeping
    failed attempts inspectable instead of silently overwriting their evidence.
    """
    request = deepcopy(dict(request))
    root = Path(project_root).resolve()
    validate_document('llm_research_task', request, root=root)
    if request['task_type'] != 'open_method_generation' or request['output_schema'] != 'open_method_proposal':
        raise OpenMethodGenerationError('OPEN_GENERATION_TASK_TYPE_INVALID')
    inputs = request['input']
    portrait_binding = inputs.get('target_portrait_binding')
    if not isinstance(portrait_binding, Mapping) or not portrait_binding:
        raise OpenMethodGenerationError('OPEN_GENERATION_PORTRAIT_BINDING_REQUIRED')
    components = inputs.get('component_methods', [])
    from wmloop.control.open_method_ir import validate_method_ir
    for component in components:
        validate_method_ir(component, root=root)
        if component.get('composition') is not None:
            raise OpenMethodGenerationError('OPEN_GENERATION_NESTED_COMPOSITION_UNSUPPORTED')
    if components and (len(components) != 2 or len({row['method_id'] for row in components}) != 2):
        raise OpenMethodGenerationError('OPEN_GENERATION_COMPONENTS_INVALID')
    supplied_evidence = list(inputs.get('source_evidence', []))
    for component in components:
        supplied_evidence.extend(component['source_evidence'])
    allowed_sources = {(row.get('source_id'), row.get('source_digest')) for row in supplied_evidence}
    destination = checked_path(output_root, code='OPEN_GENERATION_PATH_INVALID', error=OpenMethodGenerationError)
    if destination == root or root in destination.parents or destination in root.parents:
        raise OpenMethodGenerationError('OPEN_GENERATION_OUTPUT_INSIDE_SOURCE')
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    atomic_write(destination/'input-lock.json', canonical_bytes({
        'request_digest': task_request_digest(request), 'base_revision':dict(base_revision),
    }) + b'\n')
    result = {
        'schema_version':1, 'artifact_type':'verdiwm-open-method-generation',
        'state':'blocked', 'task_id':request['task_id'],
        'request_digest':task_request_digest(request), 'compilation':None, 'blockers':[],
        'authority':{'gpu_scheduling':False, 'promotion':False},
        'claim_boundary':'Generated and compiled candidate only; target checks and frozen-verifier effect trials remain required.',
    }
    try:
        task = run_llm_task(request=request, adapter=adapter, output_root=destination/'llm-task', project_root=root)
        result['llm_task'] = task
        if task['state'] != 'completed':
            result['blockers'] = task.get('blockers') or [{'code':'OPEN_GENERATION_LLM_BLOCKED'}]
        else:
            response_path = checked_path(Path(task['response_path']), code='OPEN_GENERATION_RESPONSE_PATH_INVALID', error=OpenMethodGenerationError)
            if response_path != destination/'llm-task'/'response.json':
                raise OpenMethodGenerationError('OPEN_GENERATION_RESPONSE_PATH_INVALID')
            payload = response_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != task['response_sha256']:
                raise OpenMethodGenerationError('OPEN_GENERATION_RESPONSE_DIGEST_MISMATCH')
            response = json.loads(payload)
            validate_document('llm_research_task_response', response, root=root)
            if response['task_id'] != request['task_id'] or response['task_type'] != request['task_type']:
                raise OpenMethodGenerationError('OPEN_GENERATION_RESPONSE_BINDING_MISMATCH')
            proposal = response['output']
            validate_document('open_method_proposal', proposal, root=root)
            method = proposal['method_ir']
            if any((row.get('source_id'), row.get('source_digest')) not in allowed_sources
                   for row in method.get('source_evidence', [])):
                raise OpenMethodGenerationError('OPEN_GENERATION_SOURCE_UNBOUND')
            composition = method.get('composition')
            if composition is not None:
                expected = [row['method_id'] for row in components]
                if len(expected) != 2 or composition.get('component_method_ids') != expected:
                    raise OpenMethodGenerationError('OPEN_GENERATION_COMPONENT_BINDING_MISMATCH')
            elif components and proposal['state'] == 'candidate_ready':
                raise OpenMethodGenerationError('OPEN_GENERATION_COMPOSITION_REQUIRED')
            compiled = compile_open_method_proposal(
                proposal=proposal, base_revision=base_revision, output_root=destination/'candidate',
                project_root=root, expected_portrait_binding=portrait_binding,
            )
            result['compilation'] = compiled
            result['state'] = compiled['state']
            result['blockers'] = compiled['blockers']
    except Exception as exc:
        # Detailed adapter/compiler receipts stay local; never record raw provider output as a blocker.
        result['blockers'] = [{'code':'OPEN_GENERATION_FAILED', 'error_type':type(exc).__name__}]
        atomic_write(destination/'manifest.json', canonical_bytes(result) + b'\n')
        raise
    atomic_write(destination/'manifest.json', canonical_bytes(result) + b'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--base-revision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate_open_method(
        request=json.loads(args.request.read_text()), adapter=json.loads(args.adapter.read_text()),
        base_revision=json.loads(args.base_revision.read_text()), output_root=args.output,
        project_root=Path(__file__).resolve().parents[2],
    )
    print(json.dumps({'state':result['state'], 'task_id':result['task_id'], 'output':str(args.output)}))


if __name__ == '__main__':
    main()
