"""Lint experiment artifacts against docs/ARTIFACT_CONVENTION.md.

The lint is advisory and read-only: it never mutates artifacts and it has no
authority over verification. Producers use it to find artifacts whose naming
or content would make the evidence graph unreadable; the workbench uses it to
keep non-compliant artifacts out of the default graph projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

_MAX_SCAN_FILES = 20000

_IDENTITY_FIELDS = (
    "campaign_id",
    "trial_id",
    "record_id",
    "candidate_id",
    "experiment_id",
    "relation_id",
    "probe_id",
    "proposal_id",
    "goal_id",
)
_EVIDENCE_FIELDS = (
    "evidence_refs",
    "receipt_ref",
    "verdict_ref",
    "failure_context_ref",
    "settlement_ref",
)
# Fields that mark a JSON payload as an experiment artifact even when it
# forgot to declare artifact_type. Detection must at least cover everything
# the evidence graph projects, otherwise the least conventional artifacts
# escape the lint entirely.
_MARKER_FIELDS = _IDENTITY_FIELDS + _EVIDENCE_FIELDS + (
    "model_ref",
    "target_backbone",
    "mechanism_id",
    "settlement_state",
    "verification_state",
)
_BARE_HASH = re.compile(r"\b[0-9a-f]{32,64}\b")
_BARE_MODEL_HEX = re.compile(r"[0-9a-f]{8,64}")
_BAD_SUFFIX = re.compile(r"[-_](final|new|copy|old|tmp|temp)\b", re.IGNORECASE)
_PORTABLE_REF = re.compile(r"^(cas://|urn:|sha256:)")


class ArtifactLintError(ValueError):
    """The lint invocation or root is invalid."""


def lint_payload(payload: Mapping[str, Any], *, source: str) -> list[dict[str, Any]]:
    """Return convention issues for one artifact payload."""

    issues: list[dict[str, Any]] = []

    def add(code: str, severity: str, detail: str) -> None:
        issues.append(
            {
                "code": code,
                "severity": severity,
                "detail": detail,
                "source": source,
                "artifact_type": payload.get("artifact_type"),
            }
        )

    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type:
        add("MISSING_ARTIFACT_TYPE", "error", "payload has no artifact_type")
    elif not artifact_type.startswith(("verdiwm-", "document")):
        add(
            "ARTIFACT_TYPE_NONCONVENTIONAL",
            "warning",
            f"artifact_type {artifact_type!r} does not follow verdiwm-<domain>-<noun>",
        )

    for field in _IDENTITY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            continue
        if _BARE_HASH.search(value) and not re.search(r"-[0-9a-f]{8,24}$", value):
            add(
                "IDENTITY_EMBEDS_BARE_HASH",
                "error",
                f"{field} embeds a raw hash; use a semantic slug with a trailing digest",
            )
        if _BAD_SUFFIX.search(value):
            add(
                "IDENTITY_BAD_SUFFIX",
                "error",
                f"{field}={value!r} uses a -final/-new/-copy style suffix",
            )

    model_ref = payload.get("model_ref")
    if isinstance(model_ref, str) and model_ref and (
        _PORTABLE_REF.match(model_ref) or _BARE_MODEL_HEX.fullmatch(model_ref)
    ):
        if not any(
            isinstance(payload.get(field), str) and payload.get(field)
            for field in ("model_family", "model_name", "target_backbone")
        ):
            add(
                "MODEL_REF_WITHOUT_NAME",
                "error",
                "model_ref is a bare CAS/sha256/hex reference with no model_family/model_name",
            )

    for field in _EVIDENCE_FIELDS:
        values = payload.get(field)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value and not _PORTABLE_REF.match(value):
                if re.match(r"^([A-Za-z]:[\\/]|/|\.{1,2}[\\/])", value) or value.endswith(
                    (".json", ".jsonl", ".db")
                ):
                    add(
                        "LOCAL_PATH_IN_EVIDENCE",
                        "error",
                        f"{field} contains a local path {value!r}; evidence refs must be portable",
                    )

    if not any(
        isinstance(payload.get(field), str) and payload.get(field)
        for field in ("claim_boundary", "summary", "notes")
    ):
        add(
            "MISSING_CLAIM_BOUNDARY",
            "warning",
            "no claim_boundary/summary; readers cannot tell what this artifact does not prove",
        )

    if not any(
        payload.get(field) for field in ("state", "status", "settlement_state", "verification_state")
    ):
        add("MISSING_STATE", "warning", "no state/status field from a closed enumeration")

    return issues


def lint_root(root: Path) -> dict[str, Any]:
    """Lint every JSON/JSONL artifact under root."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir() or base.is_symlink():
        raise ArtifactLintError("ARTIFACT_LINT_ROOT_INVALID")
    issues: list[dict[str, Any]] = []
    checked = 0
    for path in sorted(base.rglob("*.json")) + sorted(base.rglob("*.jsonl")):
        if checked >= _MAX_SCAN_FILES:
            break
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if (
            '"verdiwm-' not in text
            and '"artifact_type"' not in text
            and not any(f'"{marker}"' in text for marker in _MARKER_FIELDS)
        ):
            continue
        payloads: list[Any]
        if path.suffix == ".jsonl":
            try:
                payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError:
                continue
        else:
            try:
                payloads = [json.loads(text)]
            except json.JSONDecodeError:
                continue
        for ordinal, payload in enumerate(payloads):
            if not isinstance(payload, Mapping):
                continue
            if "artifact_type" not in payload and not any(
                marker in payload for marker in _MARKER_FIELDS
            ):
                continue
            checked += 1
            issues.extend(lint_payload(payload, source=f"{path}:{ordinal}"))
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = len(issues) - error_count
    return {
        "artifact_type": "verdiwm-artifact-lint-report",
        "schema_version": 1,
        "input_root": str(base),
        "checked": checked,
        "error_count": error_count,
        "warning_count": warning_count,
        "code_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "issues": issues,
    }


def make_compliance_filter(root: Path) -> Callable[[Mapping[str, Any]], bool]:
    """Return a predicate rejecting payloads with error-severity lint issues."""

    report = lint_root(root)
    bad_sources = {
        str(issue["source"])
        for issue in report["issues"]
        if issue["severity"] == "error"
    }
    # Pre-compute per-payload verdicts so identity collisions across files are safe.
    bad_payloads: set[str] = set()
    for issue in report["issues"]:
        if issue["severity"] != "error":
            continue
        bad_payloads.add(str(issue["source"]))

    def _key(source: str, ordinal: int) -> str:
        return f"{source}:{ordinal}"

    def include(payload: Mapping[str, Any], source: str, ordinal: int) -> bool:
        return _key(source, ordinal) not in bad_payloads

    include.report = report  # type: ignore[attr-defined]
    include.bad_sources = bad_sources  # type: ignore[attr-defined]
    return include


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)
    report = lint_root(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 1 if report["error_count"] else 0
    print(
        f"checked={report['checked']} errors={report['error_count']} "
        f"warnings={report['warning_count']}"
    )
    for code, count in report["code_counts"].items():
        print(f"  {code}: {count}")
    return 1 if report["error_count"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
