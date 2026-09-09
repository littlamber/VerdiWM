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


def test_negative_outcomes_are_snapshotted_before_retry(tmp_path, monkeypatch):
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    def negative(**kwargs):
        result = {'candidate_states':{'repair-a':'blocked'}, 'results':{'repair-a:gate':{'verdict':'FAIL'}}}
        (options.output_root/'compiled'/'queue'/'execution.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(pipeline, 'run_selected_queue', negative)
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'blocked'
    previous = state(options)
    refs = [Path(ref) for ref in previous['evidence_refs'] if 'execution-snapshots' in ref]
    assert len(refs) == 1
    snapshot = refs[0].read_bytes()
    assert json.loads(snapshot)['results']['repair-a:gate']['verdict'] == 'FAIL'
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    pipeline.run_autonomous_pipeline(options)
    assert refs[0].read_bytes() == snapshot
    assert str(refs[0]) in state(options)['evidence_refs']


def test_partial_execution_snapshot_survives_an_exception(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        (options.output_root/'compiled'/'queue'/'execution.json').write_text(json.dumps({'results':{'repair-a:screen':{'verdict':'PASS'}}}))
        raise RuntimeError('interrupted after first stage')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='interrupted after first stage'):
        pipeline.run_autonomous_pipeline(options)
    refs = [Path(ref) for ref in state(options)['evidence_refs'] if 'execution-snapshots' in ref]
    assert len(refs) == 1
    assert json.loads(refs[0].read_text())['results']['repair-a:screen']['verdict'] == 'PASS'


def test_same_output_cannot_run_concurrently(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    entered, release = Event(), Event()
    def run(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return execute(**kwargs)
    monkeypatch.setattr(pipeline, 'run_selected_queue', run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(pipeline.run_autonomous_pipeline, options)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(pipeline.AutonomousPipelineError, match='ALREADY_RUNNING'):
                pipeline.run_autonomous_pipeline(options)
        finally:
            release.set()
        assert first.result(timeout=5)['state'] == 'settled'
    # Lock is released at terminal completion, with no manual cleanup.
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'settled'


def test_scheduler_template_change_rejected_even_when_goal_is_unchanged(tmp_path, monkeypatch):
    real_input_document = pipeline._input_document
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, '_input_document', real_input_document)
    monkeypatch.setattr(pipeline, 'compute_source_revision', lambda *args, **kwargs: 'fixed-revision')
    monkeypatch.setattr(pipeline, 'compute_source_tree_revision', lambda *args, **kwargs: 'fixed-tree')
    pipeline.run_autonomous_pipeline(options)
    path = tmp_path/'template.json'
    value = json.loads(path.read_text())
    value['candidates'] = [{'candidate_id':'changed-method'}]
    path.write_text(json.dumps(value))
    with pytest.raises(pipeline.AutonomousPipelineError, match='INPUT_MISMATCH'):
        pipeline.run_autonomous_pipeline(options)


def test_corrupt_partial_receipt_does_not_mask_executor_failure(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        (options.output_root/'compiled'/'queue'/'execution.json').write_bytes(b'{incomplete')
        raise RuntimeError('original executor failure')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='original executor failure'):
        pipeline.run_autonomous_pipeline(options)
    refs = [Path(ref) for ref in state(options)['evidence_refs'] if 'execution-snapshots' in ref]
    assert refs[0].read_bytes() == b'{incomplete'
    assert state(options)['phase'] == 'blocked'
