"""Real CPU process trees exercise cancellation without launching a GPU runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import _run_subprocess
from wmloop.control.model_run import compile_model_run, model_run_digest
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.model_runner import ModelRunnerError, execute_model_run
from wmloop.experiments.training_scale import build_training_scale_plan
from wmloop.process_lifecycle import LocalProcessTree, ProcessLifecycleError, retire_process_group


_CPU_WORKER = '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
output = Path(manifest['output']['root'])
config = json.loads(Path(__file__).with_name('config.json').read_text())
mode = sys.argv[2]
def dump(name, value):
    temporary = output / ('.' + name)
    temporary.write_text(json.dumps(value))
    temporary.replace(output / name)
def background():
    if config.get('fast_background'):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        dump(mode + '-background.json', [child.pid])
if mode in {'child', 'grandchild'}:
    if config.get('ignore_term'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == 'child':
        grandchild = subprocess.Popen([sys.executable, __file__, str(manifest_path), 'grandchild'], start_new_session=True)
        dump('child-pids.json', [os.getpid(), grandchild.pid])
    time.sleep(30)
elif mode == 'train' and config['stage'] in {'evaluate', 'completed'}:
    Path(manifest['training']['checkpoint_receipt']).write_text('{}')
    background()
elif mode == 'evaluate' and config['stage'] == 'completed':
    Path(manifest['evaluation']['evidence_receipt']).write_text('{}')
    background()
else:
    child = subprocess.Popen([sys.executable, __file__, str(manifest_path), 'child'], start_new_session=True)
    while not (output / 'child-pids.json').exists():
        time.sleep(.01)
    pids = json.loads((output / 'child-pids.json').read_text())
    dump('stage-pids.json', [os.getpid(), *pids])
    time.sleep(.5 if config.get('leader_exit') else 30)
'''

_DISPATCH_CONTROLLER = '''
import json, sys
from pathlib import Path
from wmloop.control import campaign_dispatcher
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
from wmloop.execute import model_runner
from wmloop.execute.gpu_lease import GpuLeaseManager
root = Path(sys.argv[1])
snapshot = json.loads((root / 'snapshot.json').read_text())
model_runner.GpuLeaseManager = lambda: GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=lambda: snapshot)
if (root / 'pipeline-command.json').exists():
    campaign_dispatcher._build_command = lambda execution: json.loads((root / 'pipeline-command.json').read_text())
try:
    run_dispatcher(DispatcherOptions(root / 'state'))
except KeyboardInterrupt:
    (root / 'interrupted.json').write_text('{}')
'''


def _read_when_ready(path: Path, *, timeout: float = 8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.01)
    pytest.fail(f"process did not produce {path.name}")


def _running(pid: int) -> bool:
    try:
        return (Path('/proc') / str(pid) / 'stat').read_bytes().rsplit(b')', 1)[1].split()[0] not in {b'Z', b'X'}
    except (FileNotFoundError, ProcessLookupError):
        return False


def _assert_retired(pids):
    deadline = time.monotonic() + 3
    while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not any(_running(pid) for pid in pids)


def _cleanup(pids):
    for pid in pids:
        if _running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _inputs(root: Path, *, stage='train', ignore_term=False, leader_exit=False, fast_background=False):
    model, data = root / 'model', root / 'data'
    model.mkdir()
    data.mkdir()
    (model / 'cpu.py').write_text(_CPU_WORKER)
    (model / 'config.json').write_text(json.dumps({'stage': stage, 'ignore_term': ignore_term,
        'leader_exit': leader_exit, 'fast_background': fast_background}))
    evaluator = root / 'evaluator.json'
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}')
    for name in ('train', 'val'):
        (data / f'{name}.json').write_text(json.dumps({'records': [
            {'episode_id': f'{name}-{i}', 'sample_id': f'{name}-{i}:0:150', 'horizon_frames': 150}
            for i in range(20)
        ]}))
    scale = build_training_scale_plan(train_manifest=data / 'train.json', val_manifest=data / 'val.json', stage='probe')
    execution = {
        'kind': 'pipeline', 'repo_root': str(model), 'output_root': str(root / 'output'),
        'evaluator_contract': str(evaluator), 'runtime_python': sys.executable,
        'budget_total_gpu_hours': 1, 'asset_bindings': {'--checkpoint': str(model)},
        'model_run_adapter': {
            'profile_id': 'cpu-lifecycle', 'model_family': 'cpu', 'capability_level': 'L2',
            'runner': {'requires_training': True, 'protocol_version': 1, 'adapter_id': None,
                       'train': ['cpu.py', '{model_run_manifest}', 'train'],
                       'evaluate': ['cpu.py', '{model_run_manifest}', 'evaluate'],
                       'checkpoint_receipt': 'checkpoint.json', 'evaluation_receipt': 'evidence.json'},
        },
    }
    snapshot = {'gpus': [{'index': 0, 'uuid': 'cpu-fixture', 'name': 'CPU fixture',
                          'memory_used_mib': 0, 'utilization_gpu_percent': 0}], 'compute_apps': []}
    (root / 'snapshot.json').write_text(json.dumps(snapshot))
    manager = GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=lambda: snapshot)
    return model, data, scale, execution, manager


def _manifest(root: Path, **kwargs):
    model, data, scale, execution, manager = _inputs(root, **kwargs)
    manifest = compile_model_run(campaign_id='fixture', execution=execution, model=model, data=data,
        training_scale_plan=scale, resource_allocation={
            'artifact_type': 'verdiwm-gpu-resource-allocation', 'state': 'ready',
            'roles': [{'role': 'autonomous_candidate_evaluation', 'gpu_indices': [0]}],
        })
    output = root / 'output'
    output.mkdir()
    (output / 'model-run.json').write_text(json.dumps(manifest))
    return manifest, manager


@pytest.mark.parametrize('stop', ['store', 'sigint', 'sigterm'])
@pytest.mark.parametrize('stage', ['train', 'evaluate'])
def test_model_dispatch_cancel_retires_nested_sessions_and_releases_lease(tmp_path, stage, stop):
    model, data, scale, execution, manager = _inputs(tmp_path, stage=stage)
    store = CampaignStore(tmp_path / 'state', project_root=tmp_path)
    store.create({'campaign_id': 'fixture', 'goal': 'exercise cancellation', 'model': str(model),
                  'dataset': str(data), 'budget': 1, 'resource_request': {'gpu_indices': [0]},
                  'training_scale_plan': scale, 'execution': execution})
    store.confirm('fixture')
    controller = subprocess.Popen([sys.executable, '-c', _DISPATCH_CONTROLLER, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    pids = []
    try:
        pids = _read_when_ready(tmp_path / 'output/stage-pids.json')
        time.sleep(.25)  # Let the dispatcher observe both nested sessions.
        if stop == 'store':
            store.cancel('fixture')
        else:
            os.kill(controller.pid, signal.SIGINT if stop == 'sigint' else signal.SIGTERM)
        stdout, stderr = controller.communicate(timeout=10)
        assert controller.returncode == (143 if stop == 'sigterm' else 0), stderr.decode()
        _assert_retired(pids)
        assert store.get('fixture')['status'] == 'cancelled'
        terminal = json.loads((store.root / 'dispatch/cancelled/fixture.json').read_text())
        receipt = json.loads((tmp_path / 'output/execution-receipt.json').read_text())
        assert receipt == terminal['result']
        assert receipt['state'] == 'cancelled'
        assert receipt['stages'][-1]['stage'] == stage
        assert receipt['leased_gpus'][0]['uuid'] == 'cpu-fixture'
        assert receipt['elapsed_seconds'] > 0
        assert not (tmp_path / 'output/evidence.json').exists()
        assert not (store.root / 'dispatch/running/fixture.json').exists()
        with manager.acquire([0]):
            _assert_retired(pids)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        _cleanup(pids)


@pytest.mark.parametrize('stage', ['train', 'evaluate'])
def test_model_timeout_retires_uncooperative_nested_sessions(tmp_path, stage):
    manifest, manager = _manifest(tmp_path, stage=stage, ignore_term=True)
    manifest['execution_policy']['stage_timeout_seconds'] = 2.0
    manifest['model_run_id'] = 'model-run-' + model_run_digest(manifest)[:24]
    (tmp_path / 'output/model-run.json').write_text(json.dumps(manifest))
    pids = []
    try:
        with pytest.raises(ModelRunnerError, match=f'MODEL_RUN_{stage.upper()}_TIMEOUT'):
            execute_model_run(manifest, lease_manager=manager)
        pids = _read_when_ready(tmp_path / 'output/stage-pids.json')
        _assert_retired(pids)
        timeout = json.loads((tmp_path / 'output/execution-timeout-receipt.json').read_text())
        assert timeout['state'] == 'timed_out'
        assert timeout['stage'] == stage
        assert not (tmp_path / 'output/evidence.json').exists()
        with manager.acquire([0]):
            _assert_retired(pids)
    finally:
        _cleanup(pids)


def test_model_completion_keeps_train_and_evaluate_receipts(tmp_path):
    manifest, manager = _manifest(tmp_path, stage='completed')
    result = execute_model_run(manifest, lease_manager=manager)
    assert result['state'] == 'completed'
    assert [row['stage'] for row in result['stages']] == ['train', 'evaluate']
    assert all(row['returncode'] == 0 for row in result['stages'])
    assert Path(result['checkpoint_receipt']).is_file()
    assert Path(result['evidence_receipt']).is_file()
    assert json.loads((tmp_path / 'output/execution-receipt.json').read_text()) == result
    with manager.acquire([0]):
        pass


def test_model_completion_retires_children_of_fast_stage_leaders(tmp_path):
    manifest, manager = _manifest(tmp_path, stage='completed', fast_background=True)
    pids = []
    try:
        result = execute_model_run(manifest, lease_manager=manager)
        for stage in ('train', 'evaluate'):
            pids.extend(_read_when_ready(tmp_path / f'output/{stage}-background.json'))
        assert result['state'] == 'completed'
        assert [row['returncode'] for row in result['stages']] == [0, 0]
        _assert_retired(pids)
        with manager.acquire([0]):
            _assert_retired(pids)
    finally:
        _cleanup(pids)


@pytest.mark.parametrize('stop', ['store', 'sigint', 'sigterm'])
def test_pipeline_dispatch_cancel_retires_nested_sessions(tmp_path, stop):
    manifest, _ = _manifest(tmp_path)
    model, data = tmp_path / 'model', tmp_path / 'data'
    execution = {'kind': 'pipeline', 'repo_root': str(model), 'output_root': str(tmp_path / 'output'),
        'evaluator_contract': str(tmp_path / 'evaluator.json'), 'budget_total_gpu_hours': 1,
        'asset_bindings': {'--checkpoint': str(model)}}
    store = CampaignStore(tmp_path / 'state', project_root=tmp_path)
    store.create({'campaign_id': 'fixture', 'goal': 'exercise cancellation', 'model': str(model),
        'dataset': str(data), 'budget': 1, 'resource_request': {'gpu_indices': []}, 'execution': execution})
    store.confirm('fixture')
    (tmp_path / 'pipeline-command.json').write_text(json.dumps([
        sys.executable, str(model / 'cpu.py'), manifest['manifest_path'], 'train']))
    controller = subprocess.Popen([sys.executable, '-c', _DISPATCH_CONTROLLER, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    pids = []
    try:
        pids = _read_when_ready(tmp_path / 'output/stage-pids.json')
        time.sleep(.25)
        if stop == 'store':
            store.cancel('fixture')
        else:
            os.kill(controller.pid, signal.SIGINT if stop == 'sigint' else signal.SIGTERM)
        stdout, stderr = controller.communicate(timeout=10)
        assert controller.returncode == (143 if stop == 'sigterm' else 0), stderr.decode()
        _assert_retired(pids)
        assert store.get('fixture')['status'] == 'cancelled'
        terminal = json.loads((store.root / 'dispatch/cancelled/fixture.json').read_text())
        assert terminal['result']['cancelled'] is True
        assert not (store.root / 'dispatch/running/fixture.json').exists()
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        _cleanup(pids)


@pytest.mark.parametrize('before_train', [True, False])
def test_model_cancellation_before_stage_does_not_launch_it(tmp_path, before_train):
    manifest, manager = _manifest(tmp_path, stage='evaluate')
    checkpoint = tmp_path / 'output/checkpoint.json'
    result = execute_model_run(manifest, lease_manager=manager,
        cancel_requested=lambda: before_train or checkpoint.exists())
    assert result['state'] == 'cancelled'
    if before_train:
        assert result['stages'][0]['state'] == 'not_started'
        assert result['stages'][0]['returncode'] is None
    assert checkpoint.exists() is (not before_train)
    assert not (tmp_path / 'output/stage-pids.json').exists()
    assert not (tmp_path / 'output/evidence.json').exists()
    with manager.acquire([0]):
        pass


@pytest.mark.parametrize('leader_exit', [True, False])
def test_pipeline_cleanup_retires_nested_sessions_after_leader_exit(tmp_path, monkeypatch, leader_exit):
    manifest, _ = _manifest(tmp_path, ignore_term=True, leader_exit=leader_exit)
    monkeypatch.setattr('wmloop.control.campaign_dispatcher._build_command',
        lambda execution: [sys.executable, str(tmp_path / 'model/cpu.py'), manifest['manifest_path'], 'train'])
    pids = []
    marker = tmp_path / 'output/stage-pids.json'
    started = time.monotonic()
    try:
        result = _run_subprocess({}, cancel_requested=lambda: marker.exists() and time.monotonic() - started > .4,
            poll_seconds=.03, terminate_grace_seconds=.2) if not leader_exit else _run_subprocess({}, poll_seconds=.03)
        pids = _read_when_ready(marker)
        _assert_retired(pids)
        assert result['cancelled'] is (not leader_exit)
        assert result['returncode'] == (0 if leader_exit else -signal.SIGTERM)
    finally:
        _cleanup(pids)


def test_recycled_pid_is_never_signalled(monkeypatch):
    class RecycledProcess:
        pid = 99999999
    tree = LocalProcessTree(RecycledProcess())
    tree._identities[RecycledProcess.pid] = 'old-start'
    monkeypatch.setattr(tree, '_identity', lambda pid: ('new-start', 'S'))
    signalled = []
    monkeypatch.setattr(os, 'kill', lambda *args: signalled.append(args))
    monkeypatch.setattr(os, 'killpg', lambda *args: signalled.append(args))
    tree._signal(signal.SIGKILL)
    assert signalled == []


@pytest.mark.parametrize('exit_status', [0, 7, -signal.SIGTERM])
def test_waitable_fast_leader_retires_previously_unobserved_child(tmp_path, exit_status):
    release, marker = tmp_path / 'release', tmp_path / 'child.json'
    command = [sys.executable, '-c', '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
release, marker = map(Path, sys.argv[1:3])
while not release.exists():
    time.sleep(.01)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
marker.write_text(json.dumps([child.pid]))
status = int(sys.argv[3])
if status < 0:
    os.kill(os.getpid(), -status)
os._exit(status)
''', str(release), str(marker), str(exit_status)]
    process = subprocess.Popen(command, start_new_session=True)
    tree = LocalProcessTree(process)
    pids = []
    try:
        release.touch()
        pids = _read_when_ready(marker)
        deadline = time.monotonic() + 5
        while tree.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert tree.poll() == exit_status
        assert process.returncode is None
        assert (Path('/proc') / str(process.pid)).exists()
        retire_process_group(process, tree=tree)
        assert process.returncode == exit_status
        _assert_retired(pids)
    finally:
        retire_process_group(process, tree=tree)
        _cleanup(pids)


def test_pipeline_completion_retires_fast_same_group_child(tmp_path, monkeypatch):
    marker = tmp_path / 'child.json'
    command = [sys.executable, '-c', '''
import json, subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps([child.pid]))
''', str(marker)]
    monkeypatch.setattr('wmloop.control.campaign_dispatcher._build_command', lambda execution: command)
    pids = []
    try:
        result = _run_subprocess({}, poll_seconds=.3)
        pids = _read_when_ready(marker)
        assert result['returncode'] == 0
        assert result['cancelled'] is False
        _assert_retired(pids)
    finally:
        _cleanup(pids)


@pytest.mark.parametrize('engine', ['backend', 'runtime', 'onboarding', 'decoder'])
def test_shared_cpu_callers_retire_fast_leader_children_before_return(tmp_path, monkeypatch, engine):
    marker = tmp_path / 'child.json'
    payload = {'metrics': {'fixture': 1.0}, 'per_frame': []}
    output = json.dumps(payload)
    command = [sys.executable, '-c', '''
import json, subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps([child.pid]))
sys.stdout.write(sys.argv[2])
sys.stderr.write('fixture stderr')
''', str(marker), output]
    pids = []
    try:
        if engine == 'backend':
            from wmloop.execute.backends import LocalSubprocessBackend

            result = LocalSubprocessBackend().run(command=command, worktree=tmp_path,
                environment={}, timeout_seconds=5)
            assert result.exit_code == 0 and result.timed_out is False
            assert result.stdout == output.encode() and result.stderr == b'fixture stderr'
        elif engine == 'runtime':
            from wmloop.execute.open_method_runtime import run_local_command

            result = run_local_command(command=command, cwd=tmp_path,
                output_root=tmp_path / 'logs', timeout_seconds=5)
            assert result['state'] == 'passed' and result['returncode'] == 0
            assert (tmp_path / 'logs/stdout.log').read_text() == output
        elif engine == 'onboarding':
            from wmloop.control.onboarding_scan.runtime import _run_runtime_command

            result = _run_runtime_command(command, cwd=tmp_path, env={}, timeout_seconds=5)
            assert result.returncode == 0
            assert result.stdout == output and result.stderr == 'fixture stderr'
        else:
            from wmloop.evaluate import future_video_metrics

            original_popen = subprocess.Popen
            monkeypatch.setattr(future_video_metrics.subprocess, 'Popen',
                lambda _command, **kwargs: original_popen(command, **kwargs))
            assert future_video_metrics.recompute_future_video_metrics(prediction=b'a', ground_truth=b'b') == payload
        pids = _read_when_ready(marker)
        _assert_retired(pids)
    finally:
        _cleanup(pids)


def test_legacy_retirement_rejects_leader_reaped_before_identity_capture():
    process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    assert process.wait(timeout=5) == 0
    with pytest.raises(ProcessLifecycleError, match='LOCAL_PROCESS_LEADER_ALREADY_REAPED'):
        retire_process_group(process)


@pytest.mark.parametrize('name_scope', ['unrelated', 'leader', 'child'])
def test_non_utf8_process_names_do_not_interrupt_tree_retirement(tmp_path, name_scope):
    marker, child_ready = tmp_path / 'pids.json', tmp_path / 'child-ready'
    command = [sys.executable, '-c', '''
import ctypes, json, os, sys, time
from pathlib import Path
marker, child_ready, scope = sys.argv[1:]
if scope == 'leader':
    ctypes.CDLL(None).prctl(15, b'leader)\\xff', 0, 0, 0)
child = os.fork()
if child == 0:
    if scope == 'child':
        ctypes.CDLL(None).prctl(15, b'child)\\xff', 0, 0, 0)
    Path(child_ready).touch()
    time.sleep(30)
    os._exit(0)
while not Path(child_ready).exists():
    time.sleep(.01)
Path(marker).write_text(json.dumps([os.getpid(), child]))
time.sleep(30)
''', str(marker), str(child_ready), name_scope]
    unrelated = subprocess.Popen([sys.executable, '-c', '''
import ctypes, sys, time
if sys.argv[1] == 'unrelated':
    ctypes.CDLL(None).prctl(15, b'unrelated)\\xff', 0, 0, 0)
time.sleep(30)
''', name_scope], start_new_session=True)
    process = subprocess.Popen(command, start_new_session=True)
    pids = []
    try:
        pids = _read_when_ready(marker)
        named_pid = unrelated.pid if name_scope == 'unrelated' else pids[1 if name_scope == 'child' else 0]
        deadline = time.monotonic() + 5
        while b'\xff' not in (Path('/proc') / str(named_pid) / 'stat').read_bytes():
            assert time.monotonic() < deadline
            time.sleep(.01)
        tree = LocalProcessTree(process)
        retire_process_group(process, tree=tree)
        assert process.returncode == -signal.SIGKILL
        _assert_retired(pids)
        assert _running(unrelated.pid)
    finally:
        _cleanup(pids)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize('error_type', [FileNotFoundError, ProcessLookupError, PermissionError])
def test_uninspectable_unrelated_process_does_not_prevent_retirement(tmp_path, monkeypatch, error_type):
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    tree = LocalProcessTree(process)
    read_bytes = Path.read_bytes
    unrelated_stat = Path('/proc') / str(unrelated.pid) / 'stat'

    def read_with_proc_race(path):
        if path == unrelated_stat:
            raise error_type('unrelated process cannot be inspected')
        return read_bytes(path)

    try:
        with monkeypatch.context() as context:
            context.setattr(Path, 'read_bytes', read_with_proc_race)
            retire_process_group(process, tree=tree)
        assert process.returncode == -signal.SIGKILL
        assert _running(unrelated.pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_tracked_process_permission_failure_remains_visible(monkeypatch):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    tree = LocalProcessTree(process)
    read_bytes = Path.read_bytes
    leader_stat = Path('/proc') / str(process.pid) / 'stat'

    def read_with_permission_failure(path):
        if path == leader_stat:
            raise PermissionError('tracked process cannot be inspected')
        return read_bytes(path)

    try:
        with monkeypatch.context() as context:
            context.setattr(Path, 'read_bytes', read_with_permission_failure)
            with pytest.raises(PermissionError, match='tracked process cannot be inspected'):
                retire_process_group(process, tree=tree)
        assert process.poll() is None
    finally:
        retire_process_group(process, tree=tree)


def test_observe_skips_task_directory_disappearing_during_iteration(monkeypatch):
    class Process:
        pid = 42424242

    task_directory = Path('/proc') / str(Process.pid) / 'task'
    original_glob = Path.glob

    def glob_with_disappearing_process(path, pattern):
        if path == task_directory:
            class VanishingIterator:
                def __iter__(self):
                    return self

                def __next__(self):
                    raise ProcessLookupError('process exited during /proc enumeration')

            return VanishingIterator()
        return original_glob(path, pattern)

    monkeypatch.setattr(
        LocalProcessTree, '_identity', staticmethod(lambda _pid: ('start', 'S'))
    )
    monkeypatch.setattr(Path, 'glob', glob_with_disappearing_process)

    tree = LocalProcessTree(Process())

    assert tree._identities == {Process.pid: 'start'}
