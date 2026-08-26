"""Tool-driven engineering agent for autonomous, model-agnostic execution.

The agent is deliberately backend-neutral: a text-only OpenAI-compatible model
can emit the same structured actions as a native tool-calling backend.  The
Kernel validates every action before it touches a worktree or starts a process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from .runtime import AIProvider


class EngineeringPolicyError(ValueError):
    """An engineering action crossed its declared authority boundary."""


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+"),
)
_BLOCKED_TOKENS = {
    "sudo", "su", "doas", "pkexec", "mount", "umount", "systemctl",
    "service", "shutdown", "reboot", "init", "passwd", "useradd", "userdel",
}
_SHELLS = {"sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"}
_NETWORK_WRITE_MARKERS = {
    "push", "upload", "put", "post", "patch", "delete", "scp", "rsync",
}
_ALLOWED_ACTIONS = {
    "inspect_files", "read_file", "git_diff", "apply_patch", "run_command",
    "run_tests", "create_worktree", "collect_artifacts", "done",
}


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        result = value
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]" if match.lastindex and match.lastindex >= 2 else "[REDACTED]", result)
        return result
    return value


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


@dataclass(frozen=True)
class EngineeringSandbox:
    """Filesystem and process authority for one autonomous run."""

    worktree: Path
    output_root: Path
    readable_roots: tuple[Path, ...] = ()
    max_command_seconds: float = 1800.0
    allow_network_read: bool = True
    allow_external_write: bool = False
    allowed_gpus: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.worktree.is_absolute() or not self.output_root.is_absolute():
            raise EngineeringPolicyError("engineering roots must be absolute")
        if self.max_command_seconds <= 0:
            raise EngineeringPolicyError("max_command_seconds must be positive")

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        return (self.worktree, self.output_root)

    def check_read(self, path: Path) -> Path:
        target = path.expanduser().resolve()
        if not _inside(target, (*self.writable_roots, *self.readable_roots)):
            raise EngineeringPolicyError(f"read outside declared roots: {path}")
        return target

    def check_write(self, path: Path) -> Path:
        target = path.expanduser().resolve()
        if not _inside(target, self.writable_roots):
            raise EngineeringPolicyError(f"write outside run roots: {path}")
        return target

    def validate_command(self, argv: Sequence[str], *, cwd: Path | None = None) -> list[str]:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise EngineeringPolicyError("command must be a non-empty argv list")
        lowered = [item.lower() for item in argv]
        executable = Path(argv[0]).name.lower()
        if executable in _BLOCKED_TOKENS or any(token in _BLOCKED_TOKENS for token in lowered):
            raise EngineeringPolicyError("privilege or host-management command blocked")
        # Keep the action protocol argv-based.  A shell ``-c`` payload can
        # hide redirects, network writes, or destructive commands from the
        # token checks below, so callers must invoke scripts directly.
        if executable in _SHELLS and any(token in {"-c", "-lc", "/c"} for token in lowered[1:]):
            raise EngineeringPolicyError("shell command strings are blocked; invoke a script with argv")
        if executable == "git" and any(value in {"reset", "clean", "checkout", "restore"} for value in lowered[1:]):
            raise EngineeringPolicyError("destructive git mutation blocked")
        if executable in {"rm", "rmdir", "del"} and any(value in {"-r", "-rf", "-fr", "--recursive", "--force", "/s", "/q"} for value in lowered[1:]):
            raise EngineeringPolicyError("recursive or force deletion blocked")
        if executable in {"curl", "wget", "http", "httpie"}:
            markers = set(lowered)
            if any(marker in markers for marker in ("-x", "--request", "--data", "--data-raw", "--upload-file", "-t")):
                if any(value in lowered for value in _NETWORK_WRITE_MARKERS):
                    raise EngineeringPolicyError("external network write blocked")
            if not self.allow_network_read:
                raise EngineeringPolicyError("network access disabled by policy")
        if executable == "git" and any(value in {"push", "send-email", "remote"} for value in lowered[1:]):
            raise EngineeringPolicyError("external git write blocked")
        if any(value in {"upload", "push", "put", "post", "delete", "patch"} for value in lowered):
            if executable in {"modelscope", "huggingface-cli", "hf", "aws", "az", "gcloud", "curl", "wget"}:
                raise EngineeringPolicyError("external upload/write blocked")
        if cwd is not None and not _inside(cwd, self.writable_roots):
            raise EngineeringPolicyError(f"command cwd outside worktree/output: {cwd}")
        return list(argv)


@dataclass
class EngineeringTools:
    sandbox: EngineeringSandbox
    audit_path: Path
    default_cwd: Path | None = None
    _events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.audit_path = self.sandbox.check_write(self.audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_cwd = (self.default_cwd or self.sandbox.worktree).resolve()
        if not _inside(self.default_cwd, self.sandbox.writable_roots):
            raise EngineeringPolicyError("default cwd outside sandbox")

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        name = str(action.get("action", ""))
        args = action.get("args", {})
        if name not in _ALLOWED_ACTIONS or not isinstance(args, Mapping):
            return self._record(name, {"state": "rejected", "reason": "invalid_action"})
        try:
            result = getattr(self, "_" + name)(dict(args))
        except (EngineeringPolicyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            result = {"state": "rejected", "reason": str(exc)}
        return self._record(name, result)

    def _inspect_files(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self.sandbox.check_read(Path(str(args.get("path", self.sandbox.worktree))))
        pattern = str(args.get("pattern", "*"))
        limit = max(1, min(int(args.get("limit", 200)), 2000))
        rows = []
        for path in sorted(root.glob(pattern))[:limit]:
            if path.is_symlink() or not _inside(path, (*self.sandbox.writable_roots, *self.sandbox.readable_roots)):
                continue
            rows.append({"path": str(path.relative_to(root)), "kind": "dir" if path.is_dir() else "file", "size": path.stat().st_size if path.is_file() else None})
        return {"state": "ok", "root": str(root), "files": rows}

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.sandbox.check_read(Path(str(args["path"])))
        limit = max(1, min(int(args.get("max_bytes", 120000)), 500000))
        data = path.read_bytes()[:limit]
        return {"state": "ok", "path": str(path), "truncated": path.stat().st_size > limit, "content": data.decode("utf-8", "replace")}

    def _git_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        cwd = self.sandbox.check_read(Path(str(args.get("cwd", self.default_cwd))))
        argv = self.sandbox.validate_command(["git", "diff", "--no-ext-diff"], cwd=cwd)
        completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=self.sandbox.max_command_seconds, check=False)
        return {"state": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "stdout": completed.stdout[-120000:], "stderr": completed.stderr[-20000:]}

    def _apply_patch(self, args: dict[str, Any]) -> dict[str, Any]:
        diff = str(args.get("diff", ""))
        if not diff.strip():
            raise EngineeringPolicyError("empty patch")
        patch_path = self.sandbox.output_root / ".engineering-candidate.patch"
        patch_path = self.sandbox.check_write(patch_path)
        patch_path.write_text(diff, encoding="utf-8")
        cwd = self.sandbox.check_write(Path(str(args.get("cwd", self.default_cwd))))
        check = subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=cwd, capture_output=True, text=True, timeout=self.sandbox.max_command_seconds, check=False)
        if check.returncode:
            patch_path.unlink(missing_ok=True)
            return {"state": "failed", "reason": "patch_does_not_apply", "stderr": check.stderr[-20000:]}
        applied = subprocess.run(["git", "apply", str(patch_path)], cwd=cwd, capture_output=True, text=True, timeout=self.sandbox.max_command_seconds, check=False)
        patch_path.unlink(missing_ok=True)
        return {"state": "ok" if applied.returncode == 0 else "failed", "returncode": applied.returncode, "patch_digest": _digest(diff), "stderr": applied.stderr[-20000:]}

    def _run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        raw = args.get("argv", args.get("command"))
        if isinstance(raw, str):
            argv = shlex.split(raw)
        elif isinstance(raw, Sequence):
            argv = [str(value) for value in raw]
        else:
            raise EngineeringPolicyError("run_command requires argv")
        cwd = self.sandbox.check_write(Path(str(args.get("cwd", self.default_cwd))))
        argv = self.sandbox.validate_command(argv, cwd=cwd)
        timeout = min(float(args.get("timeout_seconds", self.sandbox.max_command_seconds)), self.sandbox.max_command_seconds)
        env = os.environ.copy()
        requested = args.get("gpus")
        if requested is not None:
            values = tuple(int(value) for value in requested) if isinstance(requested, Sequence) and not isinstance(requested, str) else (int(requested),)
            if self.sandbox.allowed_gpus and any(value not in self.sandbox.allowed_gpus for value in values):
                raise EngineeringPolicyError("requested GPU is outside allocation")
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, values))
        started = time.time()
        completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return {"state": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "argv": argv, "cwd": str(cwd), "stdout": completed.stdout[-120000:], "stderr": completed.stderr[-30000:], "duration_seconds": round(time.time() - started, 3)}

    def _run_tests(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("argv", ["python", "-m", "pytest", "-q"])
        return self._run_command({"argv": command, "cwd": args.get("cwd", self.default_cwd), "timeout_seconds": args.get("timeout_seconds", self.sandbox.max_command_seconds)})

    def _create_worktree(self, args: dict[str, Any]) -> dict[str, Any]:
        repository = self.sandbox.check_read(Path(str(args["repository"])))
        destination = self.sandbox.check_write(Path(str(args["destination"])))
        if destination.exists():
            raise EngineeringPolicyError("refusing to overwrite existing worktree")
        destination.parent.mkdir(parents=True, exist_ok=True)
        revision = str(args.get("revision", "HEAD"))
        subprocess.run(["git", "worktree", "add", "--detach", str(destination), revision], cwd=repository, capture_output=True, text=True, timeout=self.sandbox.max_command_seconds, check=True)
        return {"state": "ok", "worktree": str(destination), "revision": revision}

    def _collect_artifacts(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self.sandbox.check_read(Path(str(args.get("root", self.sandbox.worktree))))
        pattern = str(args.get("pattern", "**/*"))
        limit = max(1, min(int(args.get("limit", 200)), 2000))
        artifacts = []
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.is_symlink() or not _inside(path, self.sandbox.writable_roots):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": "sha256:" + digest})
            if len(artifacts) >= limit:
                break
        return {"state": "ok", "artifacts": artifacts}

    def _record(self, action: str, result: dict[str, Any]) -> dict[str, Any]:
        event = {"action": action, "result": _redact(result), "recorded_at": time.time()}
        self._events.append(event)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return result

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)


@dataclass
class EngineeringAgent:
    """Drive a bounded tool loop using any AIProvider backend."""

    ai: AIProvider
    tools: EngineeringTools
    max_steps: int = 32
    role: str = "engineering_agent"
    audit_path: Path | None = None

    def __post_init__(self) -> None:
        if self.audit_path is None:
            self.audit_path = self.tools.audit_path.with_name("ai-engineering.jsonl")
        self.audit_path = self.tools.sandbox.check_write(Path(self.audit_path))
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, *, objective: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        transcript: list[dict[str, Any]] = []
        instruction = self._prompt(objective, context or {}, transcript)
        for step in range(self.max_steps):
            raw = self.ai.complete(role=self.role, prompt=instruction)
            self._audit_ai(step=step, prompt=instruction, response=raw)
            action = self._parse_action(raw)
            if action is None:
                result = {"state": "abstain", "reason": "invalid_engineering_action", "step": step}
                transcript.append({"assistant": _redact(raw), "result": result})
                instruction = self._prompt(objective, context or {}, transcript)
                continue
            if action.get("action") == "done":
                result = dict(action.get("args", {}))
                result.setdefault("state", "completed")
                return {"state": result.get("state", "completed"), "result": _redact(result), "steps": step + 1, "events": list(self.tools.events)}
            tool_result = self.tools.execute(action)
            transcript.append({"assistant_action": _redact(action), "tool_result": _redact(tool_result)})
            instruction = self._prompt(objective, context or {}, transcript)
        return {"state": "abstain", "reason": "engineering_step_budget_exhausted", "steps": self.max_steps, "events": list(self.tools.events)}

    def _audit_ai(self, *, step: int, prompt: str, response: str) -> None:
        # Keep only digests and bounded redacted excerpts; prompts may contain
        # user paths, while provider credentials must never enter the ledger.
        event = {
            "step": step,
            "role": self.role,
            "prompt_digest": _digest(prompt),
            "response_digest": _digest(response),
            "response_excerpt": _redact(response)[-4000:],
            "recorded_at": time.time(),
        }
        assert self.audit_path is not None
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _prompt(self, objective: str, context: Mapping[str, Any], transcript: Sequence[Mapping[str, Any]]) -> str:
        return json.dumps({
            "objective": objective,
            "context": _redact(dict(context)),
            "allowed_actions": sorted(_ALLOWED_ACTIONS),
            "action_schema": {"action": "one allowed action", "args": "object; for done include state/reason"},
            "rules": [
                "Return exactly one JSON action and no markdown.",
                "Use argv arrays for commands; do not request shell=True.",
                "Work only inside the declared isolated worktree/output roots.",
                "Run tests after patches and inspect failures before retrying.",
                "Stop with done state=abstain if the task cannot be safely materialized.",
            ],
            "transcript": list(transcript)[-12:],
        }, sort_keys=True)

    @staticmethod
    def _parse_action(raw: str) -> dict[str, Any] | None:
        try:
            value = json.loads(raw.strip())
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or not isinstance(value.get("action"), str):
            return None
        return value


@dataclass
class AutonomousStageRunner:
    """Wrap an adapter stage runner with AI repair and deterministic retry.

    The adapter remains responsible for model-specific execution.  This bridge
    only decides when to ask the engineering agent for a bounded repair and
    records the receipt that lets the campaign resume from the failed stage.
    """

    stage_runner: Any
    agent_factory: Any
    max_repairs_per_stage: int = 2

    def __call__(self, idea: dict[str, Any], stage: str, context: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.stage_runner(idea, stage, context)
        except Exception as exc:
            result = {"state": "runtime_failed", "reason": f"{type(exc).__name__}: {exc}"}
        repairs = int(context.get("engineering_repairs", 0))
        if result.get("state") not in {"runtime_failed", "blocked", "requires_code_patch"} and not result.get("requires_engineering_repair"):
            return result
        if repairs >= self.max_repairs_per_stage:
            return {**result, "engineering": {"state": "abstain", "reason": "repair_budget_exhausted", "attempts": repairs}}
        agent = self.agent_factory(idea, stage, context)
        receipt = agent.run(
            objective=str(idea.get("objective", idea.get("title", f"repair {stage} for {idea.get('idea_id', 'idea')}"))),
            context={"stage": stage, "failure": _redact(result), "idea_id": idea.get("idea_id")},
        )
        if receipt.get("state") not in {"completed", "approved", "settled"}:
            return {**result, "engineering": receipt}
        retry_context = {**context, "engineering_repairs": repairs + 1, "engineering_receipt": receipt}
        retried = self.stage_runner(idea, stage, retry_context)
        return {**retried, "engineering": {"state": "repaired_and_retried", "attempt": repairs + 1, "receipt": receipt}}
