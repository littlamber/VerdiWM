import json
from pathlib import Path
import sys

import pytest

from test_mechanism_hypothesis import proposal, explanation
from test_open_method_study import request as study_request, PORTRAIT
from wmloop.control.open_method_generation import generate_open_method, OpenMethodGenerationError
from wmloop.control.open_method_pipeline import build_open_method_request, _normalize_method_ir

ROOT = Path(__file__).resolve().parents[1]


def generation_case(tmp_path, *, composed=False):
    study = study_request()
    components = [_normalize_method_ir(study['proposals'][role]['method_ir'], root=ROOT)
                  for role in ('source_only','target_only')] if composed else []
    candidate = study['proposals']['combined'] if composed else proposal(explanation())
    candidate['method_ir']['target_portrait_binding'] = PORTRAIT
    task = build_open_method_request(
        source_evidence=candidate['method_ir']['source_evidence'],
        target_portrait={'portrait_id':PORTRAIT['portrait_id']}, probe_fingerprints=[],
        failure_context=['Identity drift'], component_methods=components,
        target_portrait_binding=PORTRAIT,
        target_probe_binding=None,
    )
    candidate_path = tmp_path/'provider-proposal.json'
    candidate_path.write_text(json.dumps(candidate))
    provider = tmp_path/'provider.py'
    provider.write_text('''import json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
proposal = json.loads(Path(sys.argv[3]).read_text())
response = dict(schema_version=1, artifact_type='verdiwm-llm-research-task-response',
                task_id=request['task_id'], task_type=request['task_type'],
                state='completed', output=proposal)
Path(sys.argv[2]).write_text(json.dumps(response))
''')
    adapter = dict(command=[sys.executable, str(provider), '{request_path}', '{response_path}', str(candidate_path)],
                   timeout_seconds=10, max_output_bytes=100000,
                   provider_alias='local-fixture', model_alias='fixture', credential_environment_keys=[])
    return dict(request=task, adapter=adapter, base_revision=study['base_revision'],
                output_root=tmp_path/'generation', project_root=ROOT), candidate_path


@pytest.mark.parametrize('composed', [False, True])
def test_provider_response_becomes_isolated_candidate_code(tmp_path, composed):
    args, _ = generation_case(tmp_path, composed=composed)
    result = generate_open_method(**args)
    assert result['state'] == 'ready_for_calibration'
    assert not any(result['authority'].values())
    output = args['output_root']
    assert (output/'candidate'/'candidate-workspace'/'apply.py').is_file()
    check_plan = json.loads((output/'candidate'/'implementation-check-plan.json').read_text())
    assert check_plan['state'] == 'declared_not_executed'
    assert json.loads((output/'manifest.json').read_text()) == result
    assert (output/'llm-task'/'response.json').is_file()
    with pytest.raises(FileExistsError):
        generate_open_method(**args)


@pytest.mark.parametrize('change,error', [('source','SOURCE_UNBOUND'), ('components','COMPONENT_BINDING_MISMATCH')])
def test_generation_preserves_failure_and_rejects_invented_provenance(tmp_path, change, error):
    args, path = generation_case(tmp_path, composed=True)
    candidate = json.loads(path.read_text())
    if change == 'source':
        candidate['method_ir']['source_evidence'][0]['source_digest'] = 'b'*64
    else:
        candidate['method_ir']['composition']['component_method_ids'].reverse()
    path.write_text(json.dumps(candidate))
    with pytest.raises(OpenMethodGenerationError, match=error):
        generate_open_method(**args)
    output = args['output_root']
    result = json.loads((output/'manifest.json').read_text())
    assert result['state'] == 'blocked'
    assert result['compilation'] is None
    assert (output/'llm-task'/'response.json').is_file()
    assert not (output/'candidate').exists()


def test_provider_failure_has_no_candidate_and_keeps_receipt(tmp_path):
    args, _ = generation_case(tmp_path)
    args['adapter']['command'] = [sys.executable, '-c', 'raise SystemExit(1)']
    result = generate_open_method(**args)
    assert result['state'] == 'blocked'
    assert result['blockers']
    assert not (args['output_root']/'candidate').exists()
    assert (args['output_root']/'llm-task'/'receipt.json').is_file()
