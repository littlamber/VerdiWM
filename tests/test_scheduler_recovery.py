"""Resume decisions must agree with uninterrupted execution under the same queue."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Event

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
        if kwargs['plan_path'].stem == 'screen':
            assert kwargs['recover_only'] is True
            return {'verdict':'FAIL'}
        return {'verdict':'PASS'}
    monkeypatch.setattr(scheduler, 'run_auto_experiment', resume)
    result = scheduler.run_selected_queue(**options)
    assert calls == ['screen','gate','screen','gate','confirm']
    assert result['candidate_states'] == {'repair':'completed'}
    assert result['results']['repair:screen']['verdict'] == 'FAIL'
    assert scheduler.run_selected_queue(**options) == result
    assert calls == ['screen','gate','screen','gate','confirm','screen','gate','confirm']


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
    assert calls == ['screen','gate','screen','gate']


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


def replace_after_source_read(monkeypatch, source: Path, replacement: bytes) -> list[str]:
    """Replace the source after its first complete read through either file API."""

    original_bytes, original_text = Path.read_bytes, Path.read_text
    reads = []

    def replace(path: Path, method: str) -> None:
        if path == source:
            reads.append(method)
            if len(reads) == 1:
                temporary = source.with_suffix(".replacement")
                temporary.write_bytes(replacement)
                temporary.replace(source)

    def read_bytes(path: Path) -> bytes:
        payload = original_bytes(path)
        replace(path, "bytes")
        return payload

    def read_text(path: Path, *args, **kwargs) -> str:
        value = original_text(path, *args, **kwargs)
        replace(path, "text")
        return value

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    return reads


@pytest.mark.parametrize("replacement_kind", ["different_candidate", "invalid_json"])
def test_queue_replacement_cannot_mix_executed_selection_and_recorded_digest(tmp_path, monkeypatch, replacement_kind):
    options = prepare(tmp_path)
    source = options["queue_path"]
    original = source.read_bytes()
    changed = json.loads(original)
    changed["selected"][0]["candidate_id"] = "other-repair"
    replacement = json.dumps(changed).encode() if replacement_kind == "different_candidate" else b"{"
    calls = []

    def run(**kwargs):
        calls.append(kwargs["output_root"].parent.name)
        return {"verdict": "PASS"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", run)
    with monkeypatch.context() as patcher:
        reads = replace_after_source_read(patcher, source, replacement)
        first = scheduler.run_selected_queue(**options)
    assert reads == ["bytes"]
    assert source.read_bytes() == replacement
    assert first["queue_sha256"] == hashlib.sha256(original).hexdigest()
    assert first["candidate_states"] == {"repair": "completed"}
    assert set(first["results"]) == {"repair:screen", "repair:gate", "repair:confirm"}
    assert calls == ["repair"] * 3
    before = (tmp_path / "execution.json").read_bytes()
    expected = "EXECUTION_BINDING_MISMATCH" if replacement_kind == "different_candidate" else "QUEUE_INVALID"
    with pytest.raises(scheduler.ExperimentSchedulerError, match=expected):
        scheduler.run_selected_queue(**options)
    assert (tmp_path / "execution.json").read_bytes() == before
    assert calls == ["repair"] * 3
    assert not any(options[key].exists() for key in ("archive_db", "cas_root", "lock_root"))
    assert not (tmp_path / "budget.db").exists()


@pytest.mark.parametrize("invalid_payload", [b"{", b"[]", b"\xff"])
def test_invalid_queue_snapshot_cannot_be_repaired_by_replacing_the_source_mid_read(tmp_path, monkeypatch, invalid_payload):
    options = prepare(tmp_path)
    source = options["queue_path"]
    valid = source.read_bytes()
    source.write_bytes(invalid_payload)
    monkeypatch.setattr(scheduler, "run_auto_experiment", lambda **kwargs: pytest.fail("Invalid queue ran an experiment"))
    with monkeypatch.context() as patcher:
        replace_after_source_read(patcher, source, valid)
        with pytest.raises(scheduler.ExperimentSchedulerError, match="QUEUE_INVALID"):
            scheduler.run_selected_queue(**options)
    assert source.read_bytes() == valid
    assert not (tmp_path / "execution.json").exists()


def test_batch_replacement_cannot_bind_old_selection_to_new_batch_digest(tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[1]
    original = (workspace / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json").read_bytes()
    batch = json.loads(original)
    changed = json.loads(original)
    changed["candidates"][0]["candidate_id"] = "replacement-candidate"
    replacement = json.dumps(changed).encode()
    source = tmp_path / "batch.json"
    source.write_bytes(original)
    destination = tmp_path / "planned"
    with monkeypatch.context() as patcher:
        reads = replace_after_source_read(patcher, source, replacement)
        queue = scheduler.plan_candidate_batch(batch_path=source, output_root=destination, workspace_root=workspace)
    assert reads == ["bytes"]
    assert source.read_bytes() == replacement
    assert queue["selected"][0]["candidate_id"] == batch["candidates"][0]["candidate_id"]
    assert queue["batch_sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads((destination / "batch.lock.json").read_bytes())["batch_sha256"] == queue["batch_sha256"]
    before = (destination / "queue.json").read_bytes()
    with pytest.raises(scheduler.ExperimentSchedulerError, match="BATCH_MISMATCH"):
        scheduler.plan_candidate_batch(batch_path=source, output_root=destination, workspace_root=workspace)
    assert (destination / "queue.json").read_bytes() == before


def test_same_queue_rejects_an_overlapping_writer_without_changing_the_winner_projection(tmp_path, monkeypatch):
    options = prepare(tmp_path)
    started, release = Event(), Event()
    calls = []

    def run(**kwargs):
        calls.append((kwargs["plan_path"].stem, kwargs["recover_only"]))
        if len(calls) == 1:
            started.set()
            assert release.wait(10), "Winner was not released"
        return {"verdict": "PASS"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", run)
    with ThreadPoolExecutor(max_workers=1) as executor:
        winner = executor.submit(scheduler.run_selected_queue, **options)
        try:
            assert started.wait(10), "Winner did not reach execution"
            lock_inode = (tmp_path / ".scheduler.lock").stat().st_ino
            with pytest.raises(scheduler.ExperimentSchedulerError, match="^EXPERIMENT_SCHEDULER_ALREADY_RUNNING$"):
                scheduler.run_selected_queue(**options)
            assert calls == [("screen", False)]
            assert not (tmp_path / "execution.json").exists()
            assert (tmp_path / ".scheduler.lock").stat().st_ino == lock_inode
        finally:
            release.set()
        result = winner.result(timeout=10)
    assert json.loads((tmp_path / "execution.json").read_bytes()) == result
    assert result["candidate_states"] == {"repair": "completed"}
    assert all(row["state"] == "completed" for row in result["results"].values())
    assert scheduler.run_selected_queue(**options) == result
    assert calls == [(stage, recovery) for recovery in (False, True) for stage in ("screen", "gate", "confirm")]
    assert (tmp_path / ".scheduler.lock").stat().st_ino == lock_inode


def test_scheduler_lock_allows_an_independent_queue_to_complete(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("first", "second")]
    for root in roots:
        root.mkdir()
    first_options, second_options = (prepare(root) for root in roots)
    started, release = Event(), Event()

    def run(**kwargs):
        if kwargs["workspace_root"] == roots[0] and kwargs["plan_path"].stem == "screen":
            started.set()
            assert release.wait(10), "First queue was not released"
        return {"verdict": "PASS"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", run)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(scheduler.run_selected_queue, **first_options)
        try:
            assert started.wait(10), "First queue did not reach execution"
            second = scheduler.run_selected_queue(**second_options)
            assert second["candidate_states"] == {"repair": "completed"}
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=10)["candidate_states"] == {"repair": "completed"}
