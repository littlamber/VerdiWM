"""Audit the versioned research-stack integrations.

The stack manifest records which ideas from external research systems are
adopted and where they terminate inside VerdiWM.  It is intentionally an
audit surface, not a plugin loader: external agents never receive execution
or verdict authority merely by being listed in the manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ResearchStackError(ValueError):
    """A research-stack manifest is malformed or unsafe."""


_AUTHORITIES = {"advisory", "proposal_only", "ranking_only"}
_MODES = {"adopted", "reference"}


@dataclass(frozen=True)
class StackComponent:
    """One external pattern mapped to a local, authority-bounded surface."""

    component_id: str
    inspiration: str
    source_url: str
    license: str
    mode: str
    local_surface: str
    authority: str
    rationale: str


def load_research_stack(
    manifest_path: Path, *, repo_root: Path | None = None
) -> tuple[StackComponent, ...]:
    """Load and validate a research-stack manifest without importing plugins."""

    path = Path(manifest_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ResearchStackError("RESEARCH_STACK_MANIFEST_INVALID")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStackError("RESEARCH_STACK_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise ResearchStackError("RESEARCH_STACK_MANIFEST_INVALID")
    if payload.get("artifact_type") != "verdiwm-research-stack-manifest":
        raise ResearchStackError("RESEARCH_STACK_ARTIFACT_TYPE_INVALID")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ResearchStackError("RESEARCH_STACK_COMPONENTS_EMPTY")
    root = Path(repo_root or path.parents[2]).expanduser().resolve()
    seen: set[str] = set()
    parsed: list[StackComponent] = []
    for raw in components:
        if not isinstance(raw, Mapping):
            raise ResearchStackError("RESEARCH_STACK_COMPONENT_INVALID")
        values = {
            key: _required_string(raw, key)
            for key in (
                "component_id",
                "inspiration",
                "source_url",
                "license",
                "mode",
                "local_surface",
                "authority",
                "rationale",
            )
        }
        component_id = values["component_id"]
        if component_id in seen:
            raise ResearchStackError("RESEARCH_STACK_COMPONENT_DUPLICATE")
        seen.add(component_id)
        if values["mode"] not in _MODES:
            raise ResearchStackError("RESEARCH_STACK_MODE_INVALID")
        if values["authority"] not in _AUTHORITIES:
            raise ResearchStackError("RESEARCH_STACK_AUTHORITY_INVALID")
        surface = root / values["local_surface"]
        if surface.is_symlink() or not surface.exists():
            raise ResearchStackError(
                f"RESEARCH_STACK_LOCAL_SURFACE_MISSING:{values['local_surface']}"
            )
        parsed.append(StackComponent(**values))
    return tuple(parsed)


def audit_research_stack(
    manifest_path: Path, *, repo_root: Path | None = None
) -> dict[str, object]:
    """Return a stable audit report suitable for CI and release receipts."""

    components = load_research_stack(manifest_path, repo_root=repo_root)
    adopted = [item for item in components if item.mode == "adopted"]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-research-stack-audit",
        "state": "ready",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "component_count": len(components),
        "adopted_component_count": len(adopted),
        "components": [
            {
                "component_id": item.component_id,
                "inspiration": item.inspiration,
                "license": item.license,
                "mode": item.mode,
                "local_surface": item.local_surface,
                "authority": item.authority,
            }
            for item in components
        ],
        "claim_boundary": (
            "This audit records bounded integration surfaces. It does not claim that "
            "an external project or imported strategy improves model quality."
        ),
    }


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ResearchStackError(f"RESEARCH_STACK_FIELD_INVALID:{key}")
    return item.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/research/research_stack_v1.json"),
    )
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_research_stack(args.manifest, repo_root=args.repo_root)
    except ResearchStackError as exc:
        print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
