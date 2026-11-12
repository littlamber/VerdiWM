"""Read-only onboarding scan components: capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from wmloop.control.onboarding_scan.files import _relative
from wmloop.control.onboarding_scan.types import _CAPABILITY_KEYWORDS


def _classify_capabilities(
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    files: Sequence[Path],
    *,
    repo: Path,
) -> list[dict[str, object]]:
    evidence_text: dict[str, str] = {}
    for entrypoint in entrypoints:
        evidence_text[str(entrypoint["path"])] = " ".join(
            [
                str(entrypoint["path"]),
                *[str(item) for item in entrypoint.get("functions", [])],
                *[str(item) for item in entrypoint.get("cli_flags", [])],
            ]
        ).lower()
    for asset in assets:
        evidence_text[str(asset["path"])] = str(asset["path"]).lower()
    for path in files:
        relative = _relative(path, repo).lower()
        if relative not in evidence_text:
            evidence_text[relative] = relative
    rows: list[dict[str, object]] = []
    for capability, keywords in _CAPABILITY_KEYWORDS.items():
        evidence = sorted(
            path
            for path, text in evidence_text.items()
            if any(keyword in text for keyword in keywords)
        )
        state = "discovered" if evidence else "not_discovered"
        rows.append(
            {
                "capability": capability,
                "state": state,
                "confidence": "medium" if evidence else "none",
                "evidence": evidence[:20],
            }
        )
    return rows
