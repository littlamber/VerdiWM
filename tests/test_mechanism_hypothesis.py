import json
from pathlib import Path

import pytest

from wmloop.control.mechanism_hypothesis import build_mechanism_hypothesis, MechanismHypothesisError
from wmloop.control.open_method_ir import build_method_ir, OpenMethodIRError
from wmloop.control.open_method_pipeline import compile_open_method_proposal, OpenMethodPipelineError

PROJECT = Path(__file__).resolve().parents[1]


def explanation(**updates):
    values = dict(
        target_failure='Unstable scene identity during long rollouts',
        causal_assumption='Eviction of early anchors discards stable identity evidence',
        target_touchpoint=['history_selector.select'],
        predicted_observation='Anchor retention reduces paired late-horizon identity error',
        falsification_test='Reject if the gain persists after anchor removal ablation',
        required_capabilities=['history_selection'], source_evidence=['paper-a'],
        anti_conditions=['No persistent scene identity'], novelty_status='target_adaptation',
    )
    return build_mechanism_hypothesis(**(values | updates))


def method(hypothesis=None):
    return build_method_ir(
        source_evidence=[{'source_id':'paper-a','source_digest':'a'*64,'locator':'section 3', 'claim':'Retaining early context can stabilize recurrent predictions'}],
        mechanism={'summary':'Retain one anchor while preserving recent context', 'transformations':[{'stage':'inference','operation':'select_anchor','inputs':['history'],'outputs':['context']}]},
        target_mapping={'required_observables':['history'],'required_capabilities':['history_selection'], 'missing_capabilities':[], 'mapping_state':'direct_candidate'},
        training={'mode':'inference','objective':'Reduce scene identity drift', 'trainable_scope':[], 'data_requirements':[], 'scale':{'sequence_length':0,'batch_size':0,'planned_steps':0,'estimated_trainable_parameters':0}},
        falsification={'prediction':'Late-horizon identity error decreases with anchor context', 'primary_metrics':['identity_error'],'protected_metrics':['action_fidelity'], 'ablations':['Remove anchor slot'], 'anti_conditions':['No persistent scene identity']},
        mechanism_hypothesis=hypothesis,
        implementation_validation={
            'stateful':False, 'implementation_files':['apply.py'],
            'checks':[{'kind':kind, 'test_name':'anchor_selection',
                       'observable':'Selected anchors change the returned context',
                       'failure_condition':'Reject if context selection ignores the intervention'}
                      for kind in ['hook_execution','no_future_leakage','ablation_effect']],
        },
    )


def proposal(hypothesis=None):
    return {
        'schema_version':1, 'artifact_type':'verdiwm-open-method-proposal','state':'candidate_ready',
        'method_ir':method(hypothesis),
        'execution_contract':{'entrypoints':{'calibrate':['python','apply.py'],'infer':['python','apply.py'],'train':None}, 'inputs':[{'name':'history','semantic_contract':'sequence','required':True}], 'outputs':[{'name':'context','semantic_contract':'sequence','required':True}]},
        'files':[{'relative_path':'apply.py','content_utf8':'def select(history):\n    return history[:1] + history[-3:]\n','role':'implementation'}, {'relative_path':'test_apply.py','content_utf8':'from apply import select\nassert select(list(range(6))) == [0, 3, 4, 5]\n','role':'test'}],
        'tests':[{'name':'anchor_selection', 'command':['python','test_apply.py'],'state':'declared'}],
        'interface_extensions':[], 'blockers':[],
        'authority':{'source_mutation':False,'evaluator_mutation':False,'active_metric_mutation':False,'gpu_scheduling':False,'promotion':False},
        'claim_boundary':'This fixture checks proposal compilation only, not a scientific effect.',
    }


def compile_proposal(value, path):
    return compile_open_method_proposal(proposal=value, base_revision={'revision':'test-revision','source_digest':'a'*64},output_root=path,project_root=PROJECT)


def test_mechanism_contract_is_required_for_new_candidates(tmp_path):
    with pytest.raises(OpenMethodPipelineError, match='MECHANISM_HYPOTHESIS_REQUIRED'):
        compile_proposal(proposal(), tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    value = proposal(explanation())
    result = compile_proposal(value, tmp_path/'valid')
    assert result['state'] == 'ready_for_calibration'
    assert result['authority']['promotion'] is False
    assert result['authority']['gpu_scheduling'] is False
    saved = json.loads((tmp_path/'valid'/'method-ir.json').read_text())
    assert saved['mechanism_hypothesis'] == explanation()
    changed = method(explanation(causal_assumption='Different mechanism predicts a different transfer boundary'))
    assert changed['method_id'] != saved['method_id']


def test_proposal_normalizes_raw_hypothesis_and_rejects_unbound_sources(tmp_path):
    value = proposal(explanation())
    raw = value['method_ir']['mechanism_hypothesis']
    for key in ['schema_version','artifact_type','hypothesis_id']:
        raw.pop(key)
    assert compile_proposal(value, tmp_path/'normalized')['state'] == 'ready_for_calibration'
    with pytest.raises(OpenMethodIRError, match='HYPOTHESIS_SOURCE_UNBOUND'):
        method(explanation(source_evidence=['invented-evidence']))
    with pytest.raises(OpenMethodIRError, match='CAPABILITY_MISMATCH'):
        method(explanation(required_capabilities=['hidden_state_access']))
    with pytest.raises(MechanismHypothesisError, match='LIST_REQUIRED'):
        explanation(target_touchpoint='history_selector')


def test_blocked_proposal_remains_diagnostic_only(tmp_path):
    value = proposal(explanation())
    value.update(state='blocked',files=[],tests=[],execution_contract=None,blockers=[{'code':'REAL_IMPLEMENTATION_REQUIRED'}])
    result = compile_proposal(value, tmp_path/'blocked')
    assert result['state'] == 'blocked'
    assert not any(result['authority'].values())
