"""Integrity checks and bounded local processes shared by open research stages.

Local worktree execution is not an OS sandbox. It uses a copy, a clean
environment, explicit GPU devices and process-group cleanup, and records that
filesystem/network isolation is not enforced. It never grants publication.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from wmloop.control.candidate_execution import candidate_execution_digest, validate_candidate_execution_contract
from wmloop.control.open_method_ir import validate_method_ir, validate_candidate_overlay, method_ir_digest
from wmloop.storage import atomic_write, canonical_bytes, checked_path


class OpenRuntimeError(ValueError):
    """A changed artifact or invalid execution context must not run."""


def resolve_runtime_python(value) -> Path:
    """Resolve the interpreter itself; virtualenv launchers are often symlinks."""
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise OpenRuntimeError("OPEN_RUNTIME_RUNTIME_INVALID") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise OpenRuntimeError("OPEN_RUNTIME_RUNTIME_INVALID")
    return resolved


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path):
    path = checked_path(path, code='OPEN_RUNTIME_PATH_INVALID', error=OpenRuntimeError)
    if not path.is_file():
        raise OpenRuntimeError('OPEN_RUNTIME_FILE_MISSING')
    return json.loads(path.read_bytes())


def write_json(path, value):
    atomic_write(path, canonical_bytes(value) + b'\n')


def file_digest(path):
    path = checked_path(path, code='OPEN_RUNTIME_PATH_INVALID', error=OpenRuntimeError)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root):
    root = checked_path(root, code='OPEN_RUNTIME_PATH_INVALID', error=OpenRuntimeError)
    paths = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise OpenRuntimeError('OPEN_RUNTIME_SYMLINK_FORBIDDEN')
    return {p.relative_to(root).as_posix(): file_digest(p) for p in paths if p.is_file()}


def load_compilation(root):
    root = checked_path(root, code='OPEN_RUNTIME_INPUT_INVALID', error=OpenRuntimeError)
    manifest = read_json(root/'manifest.json')
    method = read_json(root/'method-ir.json')
    overlay = read_json(root/'candidate-overlay.json')
    execution = read_json(root/'candidate-execution.json')
    plan = read_json(root/'implementation-check-plan.json')
    validate_method_ir(method)
    validate_candidate_overlay(overlay)
    validate_candidate_execution_contract(execution)
    if (manifest['state'] != 'ready_for_calibration'
        or manifest['method_id'] != method['method_id']
        or manifest['method_ir_digest'] != method_ir_digest(method)
        or manifest['overlay_id'] != overlay['overlay_id']
        or overlay['method_id'] != method['method_id']
        or overlay['method_ir_digest'] != method_ir_digest(method)
        or execution['method_id'] != method['method_id']
        or manifest['execution_id'] != execution['execution_id']
        or overlay['execution_contract_binding'] != {'execution_id':execution['execution_id'], 'contract_digest':candidate_execution_digest(execution)}
        or plan['method_id'] != method['method_id']
        or plan['requirements'] != method['implementation_validation']
        or plan['tests'] != overlay['tests']
        or plan['state'] != 'declared_not_executed'):
        raise OpenRuntimeError('OPEN_RUNTIME_BUNDLE_BINDING_MISMATCH')
    expected = {row['relative_path']:row['sha256'] for row in overlay['files']}
    if tree_hashes(root/'candidate-workspace') != expected:
        raise OpenRuntimeError('OPEN_RUNTIME_CODE_CHANGED')
    return method, execution, plan


def run_local_command(*, command, cwd, output_root, timeout_seconds, runtime_python=None,
                      gpu_devices=(), replacements=None):
    """Execute once, with durable logs and cleanup of descendants on timeout."""
    if type(timeout_seconds) not in (float,int) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise OpenRuntimeError('OPEN_RUNTIME_TIMEOUT_INVALID')
    if not isinstance(command, list) or not command or any(not isinstance(t,str) or not t or '\0' in t for t in command):
        raise OpenRuntimeError('OPEN_RUNTIME_COMMAND_INVALID')
    output = checked_path(output_root, code='OPEN_RUNTIME_OUTPUT_INVALID', error=OpenRuntimeError)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    argv = list(command)
    for token, value in (replacements or {}).items():
        argv = [arg.replace('{'+token+'}', str(value)) for arg in argv]
    if argv[0] in {'python', 'python3'}:
        argv[0] = str(runtime_python or sys.executable)
    for name in ('home', 'tmp'):
        (output/name).mkdir(mode=0o700)
    environment = {'HOME':str(output/'home'), 'TMPDIR':str(output/'tmp'),
                   'PATH':str(Path(runtime_python or sys.executable).parent)+':/usr/local/bin:/usr/bin:/bin',
                   'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1',
                   'CUDA_VISIBLE_DEVICES':','.join(str(v) for v in gpu_devices)}
    write_json(output/'started.json', {'command':argv, 'timeout_seconds':timeout_seconds})
    started = time.monotonic()
    process = None
    code = None
    error = None
    try:
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                error = 'TIMEOUT'
    except OSError:
        error = 'PROCESS_START_FAILED'
    finally:
        if process is not None:
            # Also retire descendants after a parent exits successfully.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    # A successful process must not be allowed to mutate the copied candidate
    # after exit through a detached child. The process group is always retired;
    # the receipt records the observed parent result.
    receipt = {'state':'passed' if code == 0 and error is None else 'failed',
               'returncode':code, 'error':error, 'duration_seconds':time.monotonic()-started,
               'command':argv, 'stdout_sha256':file_digest(output/'stdout.log'),
               'stderr_sha256':file_digest(output/'stderr.log'),
               'isolation':{'backend':'worktree_process','filesystem_isolation_enforced':False,
                            'network_isolation_enforced':False,'credentials_forwarded':False,
                            'gpu_devices':list(gpu_devices)},
               'authority':{'community_projection':False}}
    write_json(output/'process.json', receipt)
    return receipt


def run_candidate_operation(*, compilation_root, operation, input_document, output_root,
                            timeout_seconds, runtime_python=None, gpu_devices=(), command=None):
    if operation not in {'calibrate', 'train', 'infer'}:
        raise OpenRuntimeError('OPEN_RUNTIME_OPERATION_INVALID')
    _, execution, _ = load_compilation(compilation_root)
    output = checked_path(output_root, code='OPEN_RUNTIME_OUTPUT_INVALID', error=OpenRuntimeError)
    source = Path(compilation_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise OpenRuntimeError('OPEN_RUNTIME_OUTPUT_OVERLAP')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    workspace = output/'workspace'
    shutil.copytree(source/'candidate-workspace', workspace)
    write_json(output/'input.json', input_document)
    artifacts = output/'artifacts'
    artifacts.mkdir(mode=0o700)
    argv = command if command is not None else execution['entrypoints'][operation]
    if not isinstance(argv, list) or not argv:
        raise OpenRuntimeError('OPEN_RUNTIME_OPERATION_UNBOUND')
    receipt = run_local_command(command=argv, cwd=workspace, output_root=output/'process',
        timeout_seconds=timeout_seconds, runtime_python=runtime_python, gpu_devices=gpu_devices,
        replacements={'candidate_root':workspace, 'output_root':artifacts,
                      'input_manifest':output/'input.json','world_size':max(1,len(gpu_devices)),'rank':0})
    # Re-check the source compilation after execution. A candidate that changes
    # its original bundle is rejected even though it ran in a copy.
    load_compilation(compilation_root)
    return {**receipt, 'operation':operation,'execution_id':execution['execution_id'],
            'artifacts':str(artifacts),'input_digest':digest(input_document)}
