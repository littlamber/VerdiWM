"""Per-round fail-closed invariants for formal proposal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.freeze import verify_evaluator_freeze
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION, ProposalContext


class RoundStartGuardError(RuntimeError):
    """A required round-start invariant is missing or stale."""


@dataclass(frozen=True)
class RoundStartVerification:
    repo_root: Path
    library_version: str
    registry_digest: str
    registry_freeze_path: Path
    registry_freeze_sha256: str
    evaluator_freeze_path: Path
    evaluator_freeze_sha256: str
    primitive_count: int

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-round-start-verification",
            "state": "ready",
            "repo_root": str(self.repo_root),
            "library_version": self.library_version,
            "registry_digest": self.registry_digest,
            "registry_freeze": {
                "path": str(self.registry_freeze_path),
                "sha256": self.registry_freeze_sha256,
            },
            "evaluator_freeze": {
                "path": str(self.evaluator_freeze_path),
                "sha256": self.evaluator_freeze_sha256,
            },
            "primitive_count": self.primitive_count,
        }


def round_start_guard(
    repo_root: Path,
    *,
    registry_freeze_path: Path | None = None,
    evaluator_freeze_path: Path | None = None,
) -> Callable[[ProposalContext], dict[str, object]]:
    """Build a context-aware guard for :class:`wmloop.orchestrator.ResearchLoop`."""

    def _guard(context: ProposalContext) -> dict[str, object]:
        return verify_round_start_invariants(
            repo_root=repo_root,
            context=context,
            registry_freeze_path=registry_freeze_path,
            evaluator_freeze_path=evaluator_freeze_path,
        ).to_document()

    return _guard


def verify_round_start_invariants(
    *,
    repo_root: Path,
    context: ProposalContext,
    registry_freeze_path: Path | None = None,
    evaluator_freeze_path: Path | None = None,
) -> RoundStartVerification:
    """Verify registry and evaluator freeze identities before a round side effect."""

    repo = Path(repo_root).resolve(strict=True)
    registry_path = Path(registry_freeze_path or repo / "configs" / "registry_frozen.sha256").resolve(strict=True)
    evaluator_path = Path(evaluator_freeze_path or repo / "configs" / "eval_frozen.sha256").resolve(strict=True)
    registry_bytes, registry_freeze = _load_json_mapping(registry_path, "ROUND_START_REGISTRY_FREEZE_INVALID")
    evaluator_bytes, evaluator_freeze = _load_json_mapping(evaluator_path, "ROUND_START_EVALUATOR_FREEZE_INVALID")
    _verify_registry_freeze(repo=repo, context=context, freeze=registry_freeze)
    try:
        verify_evaluator_freeze(repo / "vendor" / "ACWM-Phys", evaluator_freeze)
    except Exception as exc:
        raise RoundStartGuardError(f"ROUND_START_EVALUATOR_FREEZE_INVALID:{exc}") from exc
    return RoundStartVerification(
        repo_root=repo,
        library_version=context.library_version,
        registry_digest=context.registry.digest(),
        registry_freeze_path=registry_path,
        registry_freeze_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        evaluator_freeze_path=evaluator_path,
        evaluator_freeze_sha256=hashlib.sha256(evaluator_bytes).hexdigest(),
        primitive_count=len(context.registry.names()),
    )


def _verify_registry_freeze(*, repo: Path, context: ProposalContext, freeze: Mapping[str, Any]) -> None:
    if (
        freeze.get("schema_version") != 1
        or freeze.get("artifact_type") != "wmloop-primitive-registry-freeze"
        or freeze.get("algorithm") != "sha256"
    ):
        raise RoundStartGuardError("ROUND_START_REGISTRY_FREEZE_INVALID")
    expected_digest = _digest_field(freeze.get("registry_digest"))
    expected_library = freeze.get("library_version")
    if expected_library != context.library_version:
        raise RoundStartGuardError(f"ROUND_START_LIBRARY_VERSION_MISMATCH:{context.library_version}:{expected_library}")
    repo_registry = PrimitiveRegistry.from_root(repo)
    context_digest = context.registry.digest()
    repo_digest = repo_registry.digest()
    if context_digest != expected_digest:
        raise RoundStartGuardError(f"ROUND_START_CONTEXT_REGISTRY_DIGEST_MISMATCH:{context_digest}:{expected_digest}")
    if repo_digest != expected_digest:
        raise RoundStartGuardError(f"ROUND_START_REPO_REGISTRY_DIGEST_MISMATCH:{repo_digest}:{expected_digest}")
    expected_names = freeze.get("primitive_names")
    if not isinstance(expected_names, list) or any(not isinstance(item, str) or not item for item in expected_names):
        raise RoundStartGuardError("ROUND_START_REGISTRY_FREEZE_INVALID")
    expected_names_tuple = tuple(expected_names)
    if tuple(sorted(expected_names_tuple)) != expected_names_tuple:
        raise RoundStartGuardError("ROUND_START_REGISTRY_FREEZE_INVALID")
    if context.registry.names() != expected_names_tuple or repo_registry.names() != expected_names_tuple:
        raise RoundStartGuardError("ROUND_START_REGISTRY_NAMES_MISMATCH")
    expected_count = freeze.get("primitive_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count != len(expected_names_tuple):
        raise RoundStartGuardError("ROUND_START_REGISTRY_COUNT_MISMATCH")


def _digest_field(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RoundStartGuardError("ROUND_START_REGISTRY_FREEZE_INVALID")
    return value


def _load_json_mapping(path: Path, error_code: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RoundStartGuardError(error_code) from exc
    if not isinstance(document, Mapping):
        raise RoundStartGuardError(error_code)
    return payload, document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="verify registry and evaluator freeze for round start")
    check.add_argument("--repo-root", type=Path, default=Path("."))
    check.add_argument("--registry-freeze", type=Path)
    check.add_argument("--evaluator-freeze", type=Path)
    check.add_argument("--library-version", default=CURRENT_LIBRARY_VERSION)
    args = parser.parse_args(argv)
    if args.command == "check":
        repo = Path(args.repo_root).resolve(strict=True)
        context = ProposalContext(
            failure_report={},
            goal_spec={},
            archive_statistics={},
            registry=PrimitiveRegistry.from_root(repo),
            library_version=args.library_version,
        )
        verification = verify_round_start_invariants(
            repo_root=repo,
            context=context,
            registry_freeze_path=args.registry_freeze,
            evaluator_freeze_path=args.evaluator_freeze,
        )
        print(json.dumps(verification.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise RoundStartGuardError("ROUND_START_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
