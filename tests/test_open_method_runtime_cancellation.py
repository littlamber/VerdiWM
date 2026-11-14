"""Real CPU processes exercise cancellation without loading a model or GPU."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from wmloop.archive.store import ArchiveStore
from wmloop.execute.backends import LocalSubprocessBackend
from wmloop.execute.open_method_runtime import run_local_command
from wmloop.execute.process_lifecycle import LocalCommandSignals, ProcessLifecycleError


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group lifecycle")


_CHILD = """
import json, os, pathlib, signal, subprocess, sys, time
root, index, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(root / ('pids-' + index + '.json')).write_text(json.dumps([os.getpid(), descendant.pid]))
if mode != 'success':
    time.sleep(60)
"""


_CONTROLLER = """
import concurrent.futures, ctypes, json, os, pathlib, signal, sys, time
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.backends import LocalSubprocessBackend
from wmloop.execute.open_method_runtime import LocalCommandSignals, run_local_command
from wmloop.execute import open_method_runtime as runtime

root, mode, workers, engine = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
# Adopt and reap the killed grandchildren inside this disposable test process.
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
snapshot = {'gpus': [{'index': 0, 'uuid': 'cpu-test', 'name': 'CPU fixture',
    'memory_used_mib': 0, 'utilization_gpu_percent': 0}], 'compute_apps': []}
if mode == 'launch_signal':
    original_popen = runtime.subprocess.Popen
    def interrupt_during_launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while not (root / 'pids-0.json').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)
        return process
    runtime.subprocess.Popen = interrupt_during_launch

def observe_lease_release():
    states = {}
    for path in root.glob('pids-*.json'):
        for pid in json.loads(path.read_text()):
            stat = pathlib.Path('/proc') / str(pid) / 'stat'
            states[str(pid)] = stat.read_text().split(') ')[1][0] if stat.exists() else None
    (root / 'before-lease-release.json').write_text(json.dumps(states))

def execute(index, scope=None):
    command = [sys.executable, str(root / 'child.py'), str(root), str(index), mode]
    timeout = 0.5 if mode == 'timeout' else 30
    if engine == 'backend':
        result = LocalSubprocessBackend().run(command=command, worktree=root,
            environment=dict(os.environ), timeout_seconds=timeout, signal_scope=scope)
        return {'state': 'passed' if result.exit_code == 0 and not result.timed_out else 'failed',
            'error': 'TIMEOUT' if result.timed_out else None}
    return run_local_command(command=command, cwd=root, output_root=root / ('process-' + str(index)),
        timeout_seconds=timeout, signal_scope=scope)

def execute_auto():
    from wmloop.execute import auto_experiment as auto
    from wmloop.execute.gpu_sampling import GpuSamplingRecorder
    class ObservedLease:
        def __init__(self, lease):
            self.lease = lease
        def __enter__(self):
            return self.lease
        def __exit__(self, *args):
            observe_lease_release()
            self.lease.release()
    class Manager:
        def __init__(self, **kwargs):
            self.manager = GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=lambda: snapshot)
        def acquire(self, *args, **kwargs):
            return ObservedLease(self.manager.acquire(*args, **kwargs))
    # Only resource admission/auditing are CPU fixtures. The real transaction,
    # budget ledger, sampler, backend, and kernel signals execute unchanged.
    auto.GpuLeaseManager = Manager
    auto.verify_onboarding_admission = lambda *args, **kwargs: None
    auto.run_gpu_exclusivity_audit = lambda **kwargs: {'report_path': str(root / 'audit/report.json')}
    auto.verify_gpu_exclusivity_ready = lambda *args, **kwargs: {'state': 'ready'}
    auto.GpuSamplingRecorder = lambda **kwargs: GpuSamplingRecorder(**kwargs,
        sample_provider=lambda index: {'status': 'unavailable'})
    plan = {'schema_version': 1, 'artifact_type': 'verdiwm-auto-experiment-plan',
        'campaign_id': 'cancel-test', 'trial_id': 'cancel-test',
        'objective': 'Observe real cancellation lifecycle',
        'hypothesis': 'Cancellation retires every child process',
        'selection_reason': 'CPU process lifecycle regression test',
        'falsification_criterion': 'Any child remains running after lease release',
        'stage': 'screen', 'experiment_kind': 'diagnostic',
        'command': [sys.executable, str(root / 'child.py'), str(root), '0', mode],
        'working_directory': '.', 'allowed_gpu_indices': [0],
        'estimated_gpu_hours': 0.01, 'total_budget_gpu_hours': 0.1,
        'timeout_seconds': 30, 'gpu_wait_seconds': 0, 'sample_interval_seconds': 0.1,
        'result_path': 'result.json', 'artifacts': ['result.json'],
        'metric_gates': [{'metric': 'diagnostic', 'role': 'utility', 'operator': 'gte', 'threshold': 0}],
        'environment': {}, 'cleanup_policy': 'retain'}
    (root / 'plan.json').write_text(json.dumps(plan))
    auto.run_auto_experiment(plan_path=root / 'plan.json', output_root=root / 'auto',
        workspace_root=root, archive_db=root / 'archive.db', cas_root=root / 'cas',
        budget_db=root / 'budget.db', lock_root=root / 'locks')

try:
    if engine == 'auto':
        execute_auto()
    else:
        with GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=lambda: snapshot).acquire([0]):
            try:
                if workers == 1:
                    result = execute(0)
                    (root / 'result.json').write_text(json.dumps(result))
                else:
                    with LocalCommandSignals() as scope, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(execute, index, scope) for index in range(workers + 2)]
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
            finally:
                observe_lease_release()
finally:
    restored = all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    (root / 'handlers-restored.json').write_text(json.dumps(restored))
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
"""


def _wait_for(path: Path, controller: subprocess.Popen, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        assert controller.poll() is None, controller.communicate(timeout=1)[1]
        time.sleep(0.01)
    assert path.exists(), f"process never became ready: {path.name}"


def _running(pid: int) -> bool:
    path = Path("/proc") / str(pid) / "stat"
    try:
        return path.read_text().split(") ")[1][0] not in {"Z", "X"}
    except FileNotFoundError:
        return False


_PROCESS_CASES = [
    ("signal", signal.SIGTERM, 1),
    ("signal", signal.SIGINT, 1),
    ("signal", signal.SIGTERM, 2),
    ("signal", signal.SIGINT, 2),
    ("launch_signal", None, 1),
    ("timeout", None, 1),
    ("success", None, 1),
]


@pytest.mark.parametrize("engine, mode, signum, workers", [
    (engine, *case) for engine in ("runtime", "backend") for case in _PROCESS_CASES
] + [("auto", "signal", signum, 1) for signum in (signal.SIGTERM, signal.SIGINT)])
def test_process_groups_retire_before_gpu_lease_release(tmp_path, mode, signum, workers, engine):
    (tmp_path / "child.py").write_text(textwrap.dedent(_CHILD))
    script = tmp_path / "controller.py"
    script.write_text(textwrap.dedent(_CONTROLLER))
    controller = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), mode, str(workers), engine],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    descendants = []
    try:
        for index in range(workers):
            path = tmp_path / f"pids-{index}.json"
            _wait_for(path, controller)
            descendants.extend(json.loads(path.read_text()))
        if signum is not None:
            os.kill(controller.pid, signum)
        _, stderr = controller.communicate(timeout=10)
        expected = 128 + signal.SIGTERM if signum == signal.SIGTERM or mode == "launch_signal" else -signal.SIGINT if signum == signal.SIGINT else 0
        assert controller.returncode == expected, stderr.decode()
        assert all(not _running(pid) for pid in descendants)
        states = json.loads((tmp_path / "before-lease-release.json").read_text())
        assert all(state in (None, "Z", "X") for state in states.values()), states
        assert json.loads((tmp_path / "handlers-restored.json").read_text()) is True
        if signum is not None or mode == "launch_signal":
            assert not (tmp_path / "pids-2.json").exists(), "queued work started after cancellation"
            assert not (tmp_path / "process-0/process.json").exists()
        else:
            result = json.loads((tmp_path / "result.json").read_text())
            assert result["state"] == ("failed" if mode == "timeout" else "passed")
            assert result["error"] == ("TIMEOUT" if mode == "timeout" else None)
        if engine == "auto":
            assert not (tmp_path / "auto/manifest.json").exists()
            assert not list((tmp_path / "auto/running").glob("*.json"))
            assert ArchiveStore(tmp_path / "archive.db").visible_settled_trials() == ()
            with sqlite3.connect(tmp_path / "budget.db") as connection:
                rows = connection.execute("SELECT state FROM trials").fetchall()
            assert rows == [("admitted",)]
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.communicate(timeout=5)
        for path in tmp_path.glob("pids-*.json"):
            for pid in json.loads(path.read_text()):
                if _running(pid):
                    os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("failure", [False, True])
def test_original_signal_handlers_restored_on_return_or_launch_failure(tmp_path, failure):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def handler(_signum, _frame):
        raise AssertionError("unexpected signal")

    try:
        for sig in original:
            signal.signal(sig, handler)
        command = [str(tmp_path / "missing")] if failure else [sys.executable, "-c", "pass"]
        result = run_local_command(command=command, cwd=tmp_path,
            output_root=tmp_path / "process", timeout_seconds=5)
        assert result["state"] == ("failed" if failure else "passed")
        assert all(signal.getsignal(sig) is handler for sig in original)
    finally:
        for sig, previous in original.items():
            signal.signal(sig, previous)


@pytest.mark.parametrize("engine", ["runtime", "backend"])
def test_worker_without_main_thread_scope_fails_before_spawning(tmp_path, engine):
    with ThreadPoolExecutor(max_workers=1) as pool:
        if engine == "backend":
            future = pool.submit(LocalSubprocessBackend().run,
                command=[sys.executable, "-c", "pass"], worktree=tmp_path, environment={}, timeout_seconds=5)
        else:
            future = pool.submit(run_local_command, command=[sys.executable, "-c", "pass"],
                cwd=tmp_path, output_root=tmp_path / "process", timeout_seconds=5)
        with pytest.raises(ProcessLifecycleError, match="MAIN_THREAD_SIGNAL_SCOPE_REQUIRED"):
            future.result()
    assert not (tmp_path / "process").exists()


def test_inactive_scope_cannot_launch_a_process(tmp_path):
    with pytest.raises(ProcessLifecycleError, match="SIGNAL_SCOPE_INACTIVE"):
        run_local_command(command=[sys.executable, "-c", "pass"], cwd=tmp_path,
            output_root=tmp_path / "process", timeout_seconds=5, signal_scope=LocalCommandSignals())
    assert not (tmp_path / "process").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_backend_polling_preserves_stdout_and_stderr(tmp_path, timeout):
    command = [sys.executable, "-c",
        "import sys, time; sys.stdout.write('o'*100000); sys.stdout.flush(); "
        "sys.stderr.write('e'*100000); sys.stderr.flush(); time.sleep(" + ("30" if timeout else "0.25") + ")"]
    result = LocalSubprocessBackend().run(command=command, worktree=tmp_path,
        environment={}, timeout_seconds=0.5 if timeout else 5)
    assert result.stdout == b"o" * 100000
    assert result.stderr == b"e" * 100000
    assert result.timed_out is timeout
    assert result.exit_code == (-signal.SIGKILL if timeout else 0)


def test_backend_restores_signal_handlers_when_process_cannot_start(tmp_path):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with pytest.raises(FileNotFoundError):
        LocalSubprocessBackend().run(command=[str(tmp_path / "missing")],
            worktree=tmp_path, environment={}, timeout_seconds=5)
    assert all(signal.getsignal(sig) == handler for sig, handler in original.items())


_STUDY_WAIT_CONTROLLER = """
import json, pathlib, signal, sys
from wmloop.execute import open_method_study_runner as runner
from wmloop.execute.gpu_lease import GpuLeaseManager

root = pathlib.Path(sys.argv[1])
inputs = json.loads((root / 'study-inputs.json').read_text())
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
snapshot = {'gpus': [{'index': 0, 'uuid': 'cpu-study-test', 'name': 'CPU fixture',
    'memory_used_mib': 2048, 'utilization_gpu_percent': 100}], 'compute_apps': []}
def busy_snapshot():
    (root / 'waiting-for-gpu').touch()
    return snapshot

manager = GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=busy_snapshot)
# Validation and the real threaded study phase run unchanged. Calibration is
# already covered independently; no candidate or GPU command may start here.
runner._run_calibrations = lambda **kwargs: [
    {'role': role, 'state': 'passed'} for role in runner.ROLES]
def unexpected_operation(**kwargs):
    (root / 'unexpected-candidate').touch()
    raise AssertionError('candidate launched while GPU admission was blocked')
runner.run_candidate_operation = unexpected_operation
original_trial = runner._run_trial
trial_exceptions = []
def observed_trial(**kwargs):
    try:
        return original_trial(**kwargs)
    except BaseException as exc:
        trial_exceptions.append(type(exc).__name__)
        raise
runner._run_trial = observed_trial
outcome = {}
try:
    runner.execute_open_method_study(**{key: pathlib.Path(value) for key, value in inputs.items()},
        output_root=root / 'execution', runtime_python=pathlib.Path(sys.executable),
        gpu_indices=[0], max_parallel=1, gpu_wait_seconds=30, lease_manager=manager)
except BaseException as exc:
    outcome = {'exception_type': type(exc).__name__,
        'exit_code': exc.code if isinstance(exc, SystemExit) else 130}
    raise
finally:
    snapshot['gpus'][0]['memory_used_mib'] = 0
    snapshot['gpus'][0]['utilization_gpu_percent'] = 0
    with GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=lambda: snapshot).acquire([0]):
        outcome['lease_reacquired'] = True
    outcome['trial_exceptions'] = trial_exceptions
    outcome['handlers_restored'] = all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    (root / 'study-outcome.json').write_text(json.dumps(outcome))
"""


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_open_study_signal_cancels_gpu_wait_without_trial_or_settlement(tmp_path, signum):
    from test_open_method_study_runner import _runner_fixture

    fixture = _runner_fixture(tmp_path)
    inputs = {
        "study_root": fixture["study"], "checkpoint": fixture["checkpoint"],
        "train_split": fixture["train"], "selection_split": fixture["selection"],
        "confirmation_split": fixture["confirmation"], "verifier": fixture["verifier"],
    }
    (tmp_path / "study-inputs.json").write_text(json.dumps({key: str(value) for key, value in inputs.items()}))
    script = tmp_path / "study-controller.py"
    script.write_text(textwrap.dedent(_STUDY_WAIT_CONTROLLER))
    controller = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        _wait_for(tmp_path / "waiting-for-gpu", controller)
        started = time.monotonic()
        os.kill(controller.pid, signum)
        _, stderr = controller.communicate(timeout=5)
        assert time.monotonic() - started < 2, "cancellation waited for the 30-second lease deadline"
        assert controller.returncode == (143 if signum == signal.SIGTERM else -signal.SIGINT), stderr.decode()
        result = json.loads((tmp_path / "study-outcome.json").read_text())
        assert result["exception_type"] == ("SystemExit" if signum == signal.SIGTERM else "KeyboardInterrupt")
        assert result["exit_code"] == (143 if signum == signal.SIGTERM else 130)
        assert result["lease_reacquired"] is True
        assert result["handlers_restored"] is True
        assert result["trial_exceptions"]
        assert set(result["trial_exceptions"]) == {"LocalProcessCancelled"}
        assert not (tmp_path / "unexpected-candidate").exists()
        assert not list((tmp_path / "execution").rglob("trial.json"))
        assert not (tmp_path / "execution" / "execution.json").exists()
        assert not (tmp_path / "execution" / "evidence").exists()
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.communicate(timeout=5)
