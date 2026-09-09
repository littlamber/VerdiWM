"""CPU integration of research state with pipeline orchestration, not model effects."""
import json
from pathlib import Path

import pytest

from wmloop.execute import autonomous_pipeline as pipeline


def setup_pipeline(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    evaluator = tmp_path/'evaluator.json'
    template = tmp_path/'template.json'
    template.write_text(json.dumps({'objective':'Improve held-out rollout consistency', 'falsification_criterion':'Reject if protected action fidelity degrades'}))
    evaluator.write_text(json.dumps({'scheduler_template':str(template), 'metrics':['rollout_error']}))
    monkeypatch.setattr(pipeline, '_input_document', lambda *args, **kwargs: {'test_binding': 'fixed'})
    monkeypatch.setattr(pipeline, 'run_onboarding', lambda options: {'state':'ready_for_conformance_smoke'})
    monkeypatch.setattr(pipeline, 'run_conformance', lambda options: {'verdict':'PASS'})
    output = tmp_path/'output'
    def compile_queue(**kwargs):
        path = output/'compiled'/'queue'/'queue.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'selected':[{'candidate_id':'repair-a'}]}))
        return {'queue_path':str(path)}
    monkeypatch.setattr(pipeline, 'compile_and_plan', compile_queue)
    def execute(**kwargs):
        execution = {'candidate_states':{'repair-a':'completed'}}
        (output/'compiled'/'queue'/'execution.json').write_text(json.dumps(execution))
        return execution
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    options = pipeline.AutonomousPipelineOptions(repo_root=source, output_root=output, evaluator_contract=evaluator, budget_total_gpu_hours=1)
    return options, execute


def state(options):
    return json.loads((options.output_root/'research'/'research-state.json').read_text())


def test_pipeline_records_phases_and_resumes_terminal_without_rerunning(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    result = pipeline.run_autonomous_pipeline(options)
    assert result['state'] == 'settled'
    current = state(options)
    assert current['phase'] == 'settled'
    assert current['goal']['description'] == 'Improve held-out rollout consistency'
    assert current['candidate_actions'] == [{'candidate_id':'repair-a'}]
    assert current['budget']['gpu_hours_remaining'] is None
    phases = {json.loads(path.read_text())['phase'] for path in (options.output_root/'research'/'research-history').glob('*.json')}
    assert phases == {'observe','hypothesize','select','implement','verify','remember'}
    monkeypatch.setattr(pipeline, 'run_selected_queue', lambda **kwargs: pytest.fail('Settled pipeline reran execution'))
    assert pipeline.run_autonomous_pipeline(options) == result
    assert state(options) == current


def test_interruption_and_resume_preserve_prior_evidence(tmp_path, monkeypatch):
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        raise RuntimeError('executor interrupted')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='executor interrupted'):
        pipeline.run_autonomous_pipeline(options)
    previous = state(options)
    assert previous['phase'] == 'blocked'
    assert 'execution' in previous['last_transition']['reason']
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'settled'
    assert state(options)['revision'] > previous['revision']
    assert (options.output_root/'research'/'research-history'/f"{previous['state_id']}.json").is_file()


def test_onboarding_blocker_is_projected_without_experiment_execution(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, 'run_onboarding', lambda options: {'state':'blocked'})
    monkeypatch.setattr(pipeline, 'run_selected_queue', lambda **kwargs: pytest.fail('Blocked onboarding ran experiment'))
    assert pipeline.run_autonomous_pipeline(options)['blocked_stage'] == 'onboarding'
    assert state(options)['phase'] == 'blocked'
