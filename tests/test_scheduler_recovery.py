"""Resume decisions must agree with uninterrupted execution under the same queue."""
import hashlib
import json

import pytest

from wmloop.execute import experiment_scheduler as scheduler


def prepare(tmp_path):
    stages = []
    for stage in ('screen', 'gate', 'confirm'):
        path = tmp_path / f'{stage}.json'
        path.write_text(json.dumps({'stage': stage}))
        stages.append({'stage':stage, 'plan_path':path.name, 'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    queue = {'artifact_type':'verdiwm-auto-experiment-queue', 'state':'ready', 'campaign_id':'test',
             'batch_sha256':'a'*64, 'selected':[{'candidate_id':'repair', 'stages':stages}]}
    path = tmp_path / 'queue.json'
    path.write_text(json.dumps(queue))
    return dict(queue_path=path, workspace_root=tmp_path, archive_db=tmp_path/'archive.db',
                cas_root=tmp_path/'cas', lock_root=tmp_path/'locks')


def test_failed_screen_continues_to_confirmation_after_interruption(tmp_path, monkeypatch):
    options = prepare(tmp_path)
    calls = []
    def interrupted(**kwargs):
        stage = kwargs['plan_path'].stem
        calls.append(stage)
        if stage == 'gate':
            raise RuntimeError('interrupted at gate')
        return {'verdict':'FAIL'}
    monkeypatch.setattr(scheduler, 'run_auto_experiment', interrupted)
    with pytest.raises(RuntimeError, match='interrupted at gate'):
        scheduler.run_selected_queue(**options)
    def resume(**kwargs):
        calls.append(kwargs['plan_path'].stem)
        return {'verdict':'PASS'}
    monkeypatch.setattr(scheduler, 'run_auto_experiment', resume)
    result = scheduler.run_selected_queue(**options)
    assert calls == ['screen','gate','gate','confirm']
    assert result['candidate_states'] == {'repair':'completed'}
    assert result['results']['repair:screen']['verdict'] == 'FAIL'
    assert scheduler.run_selected_queue(**options) == result
    assert calls == ['screen','gate','gate','confirm']


def test_failed_formal_gate_stays_blocked_on_resume(tmp_path, monkeypatch):
    options = prepare(tmp_path)
    calls = []
    def run(**kwargs):
        stage = kwargs['plan_path'].stem
        calls.append(stage)
        return {'verdict':'FAIL' if stage == 'gate' else 'PASS'}
    monkeypatch.setattr(scheduler, 'run_auto_experiment', run)
    result = scheduler.run_selected_queue(**options)
    assert result['candidate_states'] == {'repair':'blocked'}
    assert scheduler.run_selected_queue(**options) == result
    assert calls == ['screen','gate']


@pytest.mark.parametrize('mutation', ['queue', 'completed_plan', 'execution'])
def test_changed_inputs_cannot_reuse_completed_results(tmp_path, monkeypatch, mutation):
    options = prepare(tmp_path)
    monkeypatch.setattr(scheduler, 'run_auto_experiment', lambda **kwargs: {'verdict':'PASS'})
    scheduler.run_selected_queue(**options)
    if mutation == 'queue':
        path = tmp_path/'queue.json'
        value = json.loads(path.read_text())
        value['selected'][0]['candidate_id'] = 'different-repair'
        path.write_text(json.dumps(value))
    elif mutation == 'completed_plan':
        (tmp_path/'screen.json').write_text('{"changed":true}')
    else:
        path = tmp_path/'execution.json'
        value = json.loads(path.read_text())
        value['campaign_id'] = 'different-campaign'
        path.write_text(json.dumps(value))
    monkeypatch.setattr(scheduler, 'run_auto_experiment', lambda **kwargs: pytest.fail('Changed inputs ran experiments'))
    with pytest.raises(scheduler.ExperimentSchedulerError, match='MISMATCH'):
        scheduler.run_selected_queue(**options)
