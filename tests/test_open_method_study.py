from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_mechanism_hypothesis import proposal, explanation
from wmloop.control.open_method_pipeline import (
    _normalize_method_ir, compile_open_method_proposal, build_open_method_request,
    OpenMethodPipelineError,
)
from wmloop.control.open_method_ir import validate_method_ir, method_ir_digest, OpenMethodIRError
from wmloop.control.open_method_study import compile_open_method_study, OpenMethodStudyError
from wmloop.control.method_realization import MethodRealizationError
from wmloop.contracts import validate_document

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = {'portrait_id':'model-portrait-'+'a'*24,'portrait_digest':'a'*64}


def request():
    proposals = {}
    for role in ('baseline','source_only','target_only','combined'):
        value = proposal(explanation(causal_assumption=f'The {role} mechanism predicts reduced held-out state drift'))
        value['method_ir']['target_portrait_binding'] = PORTRAIT
        proposals[role] = value
    ids = [_normalize_method_ir(proposals[r]['method_ir'],root=ROOT)['method_id'] for r in ('source_only','target_only')]
    combined = proposals['combined']['method_ir']
    combined['target_mapping']['mapping_state'] = 'composition_candidate'
    combined['composition'] = dict(component_method_ids=ids,
        complementarity='Persistent state and local correction address different error sources',
        predicted_joint_effect='Combined repair improves over both single-mechanism controls',
        conflict_resolution='Correct the new observation before writing persistent memory',
        anti_conditions=['Correction removes genuine action-induced changes'])
    return dict(proposals=proposals,base_revision={'revision':'test','source_digest':'f'*64},
                experiment_binding=dict(zip(['checkpoint_digest','train_split_digest','selection_split_digest','confirmation_split_digest','verifier_digest'],[c*64 for c in 'abcde'])),
                target_portrait_binding=PORTRAIT,seeds=[11,22],
                estimated_gpu_hours_per_seed={r:0.1 for r in proposals},budget_gpu_hours=1)


def test_four_arm_compilation_binds_components_cost_and_validation(tmp_path):
    result = compile_open_method_study(**request(),output_root=tmp_path/'study',project_root=ROOT)
    assert [a['role'] for a in result['arms']] == ['baseline','source_only','target_only','combined']
    assert result['estimated_total_gpu_hours'] == pytest.approx(0.8)
    assert not any(result['authority'].values())
    for arm in result['arms']:
        path=tmp_path/'study'/arm['role']/'implementation-check-plan.json'
        assert json.loads(path.read_text())['state']=='declared_not_executed'
    for relative, digest in result['file_sha256'].items():
        assert hashlib.sha256((tmp_path/'study'/relative).read_bytes()).hexdigest() == digest
    assert 'combined-source_only-target_only+baseline' in result['required_contrasts']


@pytest.mark.parametrize('change,error',[
    ('missing_arm','FOUR_ARMS'),('budget','BUDGET_EXCEEDED'),('component','COMPONENT_BINDING'),
    ('metrics','METRIC_MISMATCH'),('split','DISTINCT_SPLITS'),('seed','SEEDS_INVALID'),
    ('blocked','ARM_NOT_READY'),
    ('duplicate_arm','DISTINCT_ARMS'),
])
def test_invalid_or_partial_study_is_not_published(tmp_path,change,error):
    req=request()
    if change=='missing_arm': del req['proposals']['source_only']
    if change=='budget': req['budget_gpu_hours']=0.3
    if change=='component': req['proposals']['combined']['method_ir']['composition']['component_method_ids'][0]='method-ir-'+'0'*24
    if change=='metrics': req['proposals']['baseline']['method_ir']['falsification']['primary_metrics']=['different_metric']
    if change=='split': req['experiment_binding']['confirmation_split_digest']=req['experiment_binding']['selection_split_digest']
    if change=='seed': req['seeds']=[11,11]
    if change=='blocked': req['proposals']['combined'].update(state='blocked',files=[],tests=[],execution_contract=None)
    if change=='duplicate_arm': req['proposals']['baseline'] = deepcopy(req['proposals']['source_only'])
    with pytest.raises(OpenMethodStudyError,match=error):
        compile_open_method_study(**req,output_root=tmp_path/'study',project_root=ROOT)
    assert not (tmp_path/'study').exists()
    assert not list(tmp_path.glob('.study.*'))


def test_training_method_requires_parameter_and_optimizer_checks(tmp_path):
    value=proposal(explanation())
    value['method_ir']['training'].update(mode='training',trainable_scope=['memory'])
    value['execution_contract']['entrypoints']['train']=['python','train.py']
    with pytest.raises(MethodRealizationError,match='CHECKS_INCOMPLETE'):
        compile_open_method_proposal(proposal=value,base_revision={'revision':'test','source_digest':'f'*64},output_root=tmp_path/'training',project_root=ROOT)


@pytest.mark.parametrize('change,error', [('file','FILES_UNBOUND'),('test','TEST_UNBOUND'),('stateful','CHECKS_INCOMPLETE'),('composition','COMPOSITION_CONTRACT_REQUIRED')])
def test_implementation_declarations_must_bind_real_files_and_tests(tmp_path,change,error):
    value=proposal(explanation())
    checks=value['method_ir']['implementation_validation']
    if change=='file': checks['implementation_files']=['missing.py']
    if change=='test': checks['checks'][0]['test_name']='missing_test'
    if change=='stateful': checks['stateful']=True
    if change=='composition': value['method_ir']['target_mapping']['mapping_state']='composition_candidate'
    with pytest.raises(MethodRealizationError,match=error):
        compile_open_method_proposal(proposal=value,base_revision={'revision':'test','source_digest':'f'*64},output_root=tmp_path/'bad',project_root=ROOT)


def test_stateful_training_checks_are_preserved_without_execution(tmp_path):
    value = proposal(explanation())
    value['method_ir']['training'].update(mode='training', trainable_scope=['memory'])
    value['execution_contract']['entrypoints']['train'] = ['python', 'apply.py']
    validation = value['method_ir']['implementation_validation']
    validation['stateful'] = True
    for kind in ('optimizer_binding', 'parameter_update', 'train_infer_parity', 'state_lifecycle'):
        validation['checks'].append({
            'kind': kind, 'test_name': 'anchor_selection',
            'observable': 'Target method state and parameters are inspected',
            'failure_condition': 'Reject if the declared state or parameter check fails',
        })
    value['files'][1]['content_utf8'] = 'raise RuntimeError("must not execute during compilation")\n'
    result = compile_open_method_proposal(
        proposal=value, base_revision={'revision':'test','source_digest':'f'*64},
        output_root=tmp_path/'training', project_root=ROOT,
    )
    assert result['state'] == 'ready_for_calibration'
    saved = json.loads((tmp_path/'training'/'implementation-check-plan.json').read_text())
    assert saved['requirements'] == validation
    assert saved['state'] == 'declared_not_executed'


def test_optional_composition_is_validated_without_implementation_contract():
    value = request()['proposals']['combined']['method_ir']
    value.pop('implementation_validation')
    value['composition']['component_method_ids'] = ['invalid']
    value['method_id'] = 'method-ir-' + method_ir_digest(value)[:24]
    with pytest.raises(MethodRealizationError, match='COMPOSITION_INVALID'):
        validate_method_ir(value)


def test_generation_request_binds_full_inputs_and_component_definitions():
    req = request()
    components = [_normalize_method_ir(req['proposals'][role]['method_ir'], root=ROOT)
                  for role in ('source_only', 'target_only')]
    inputs = dict(source_evidence=components[0]['source_evidence'],
                  target_portrait={'portrait_id':PORTRAIT['portrait_id'], 'observables':['history']},
                  probe_fingerprints=[{'fingerprint_id':'probe-1', 'response':[0.2]}],
                  failure_context=['Identity drift'], component_methods=components)
    original = build_open_method_request(**inputs)
    validate_document('llm_research_task', original)
    assert original['input']['component_methods'] == components
    for field in ('portrait', 'probe', 'component'):
        changed = deepcopy(inputs)
        if field == 'portrait':
            changed['target_portrait']['observables'].append('latent_state')
        elif field == 'probe':
            changed['probe_fingerprints'][0]['response'] = [-0.2]
        else:
            changed['component_methods'][0]['falsification']['anti_conditions'].append('Memory pollution')
            changed['component_methods'][0] = _normalize_method_ir(changed['component_methods'][0], root=ROOT)
        assert build_open_method_request(**changed)['task_id'] != original['task_id']
    inputs['probe_fingerprints'][0]['response'][0] = 99
    inputs['component_methods'][0]['training']['objective'] = 'Mutated caller data'
    assert original['input']['probe_fingerprints'][0]['response'] == [0.2]
    assert original['input']['component_methods'][0]['training']['objective'] != 'Mutated caller data'


@pytest.mark.parametrize('change,error', [('tamper','DIGEST_MISMATCH'), ('duplicate','DISTINCT_COMPONENTS'), ('missing','TWO_COMPONENTS')])
def test_generation_rejects_unbound_or_invalid_components(change,error):
    req = request()
    components = [_normalize_method_ir(req['proposals'][role]['method_ir'], root=ROOT)
                  for role in ('source_only', 'target_only')]
    if change == 'tamper': components[0]['training']['objective'] = 'Tampered objective'
    if change == 'duplicate': components[1] = components[0]
    if change == 'missing': components.pop()
    with pytest.raises((OpenMethodPipelineError, OpenMethodIRError), match=error):
        build_open_method_request(source_evidence=[], target_portrait={},
                                  probe_fingerprints=[], failure_context=[], component_methods=components)


def test_study_identity_binds_files_and_seeds(tmp_path):
    req = request()
    def compile_at(name):
        return compile_open_method_study(**req, output_root=tmp_path/name, project_root=ROOT)
    original = compile_at('first')
    assert compile_at('same')['study_id'] == original['study_id']
    req['proposals']['source_only']['files'][0]['content_utf8'] += '\n# implementation revision\n'
    assert compile_at('different_code')['study_id'] != original['study_id']
    req['seeds'] = [33,44]
    assert compile_at('different_seeds')['study_id'] != original['study_id']
