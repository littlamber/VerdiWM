"""Preserve a locally validated source snapshot and distributions for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from scripts.export.build_github_staging import audit_release_tree
from wmloop.storage import checked_path, sha256_file


def public_test_summary(source: Path) -> bytes:
    """Publish numeric totals only; arbitrary pytest text is private by default."""
    suites = ET.parse(source).getroot().findall("testsuite")
    if not suites:
        raise ValueError("PYTEST_SUITES_MISSING")
    totals = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    if any(value < 0 for value in totals.values()):
        raise ValueError("PYTEST_COUNTS_INVALID")
    root = ET.Element("testsuites")
    ET.SubElement(root, "testsuite", name="public-cpu-summary", **{key: str(value) for key, value in totals.items()})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def build_delivery(build_dir: Path, output_dir: Path) -> None:
    destination = checked_path(output_dir, code="RELEASE_OUTPUT_SYMLINK")
    if destination.exists():
        raise ValueError("RELEASE_OUTPUT_EXISTS")
    archive_report = json.loads((build_dir / "archive_report.json").read_text())
    source_report = json.loads((build_dir / "source_report.json").read_text())
    installs = {mode: json.loads((build_dir / f"install_{mode}.json").read_text()) for mode in ("locked", "resolved")}
    suites = ET.parse(build_dir / "pytest.xml").getroot().findall("testsuite")
    totals = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    if not suites or totals["tests"] == 0 or totals["failures"] or totals["errors"] or archive_report.get("status") != "passed":
        raise ValueError("RELEASE_VALIDATION_INCOMPLETE")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        source = temporary / "repository"
        shutil.copytree(build_dir / "source", source, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        findings = audit_release_tree(source)
        if findings:
            raise ValueError("DELIVERY_SOURCE_AUDIT_FAILED:" + ";".join(findings))
        source_hashes = {p.relative_to(source).as_posix(): sha256_file(p) for p in sorted(source.rglob("*")) if p.is_file()}
        if source_hashes != source_report["source_hashes"]:
            raise ValueError("SOURCE_CHANGED_DURING_VALIDATION")
        distributions = list(build_dir.glob("*.whl")) + list(build_dir.glob("*.tar.gz"))
        if len(distributions) != 2:
            raise ValueError("RELEASE_DISTRIBUTIONS_INCOMPLETE")
        wheel = next(path for path in distributions if path.suffix == ".whl")
        for mode, receipt in installs.items():
            if receipt.get("status") != "passed" or receipt.get("dependency_mode") != mode or receipt.get("wheel_sha256") != sha256_file(wheel):
                raise ValueError("RELEASE_INSTALL_RECEIPT_INVALID")
        for path in distributions:
            shutil.copy2(path, temporary / path.name)
        (temporary / "pytest.xml").write_bytes(public_test_summary(build_dir / "pytest.xml"))
        report = {
            "status": "passed",
            "claim_boundary": "Local source, CPU and offline packaging validation; no GPU effect claim, GitHub push or approval of development Git history.",
            "history_policy": "snapshot_has_no_git_history; existing_public_history_not_audited_or_authorized_for_rewrite",
            "tests": totals,
            "test_report_policy": "Public numeric summary only; raw pytest XML, host metadata, test output and local paths remain private.",
            "archives": archive_report,
            "installations": installs,
            "source_files": len(source_hashes),
            "distributions": {p.name: sha256_file(p) for p in distributions},
            "source_hashes": source_hashes,
        }
        (temporary / "release_report.json").write_text(json.dumps(report, indent=2) + "\n")
        # Archive contents were audited before delivery; audit all remaining
        # public attachments as well, without treating binary archives as text.
        from scripts.export.build_github_staging import audit_text

        for name in ("pytest.xml", "release_report.json"):
            if audit_text((temporary / name).read_text(), name):
                raise ValueError("PUBLIC_ATTACHMENT_AUDIT_FAILED")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_delivery(args.build_dir, args.output_dir)


if __name__ == "__main__":
    main()
