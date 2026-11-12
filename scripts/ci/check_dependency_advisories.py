"""Query OSV for the locked PyPI versions; unavailable audits never pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import tomli

from wmloop.storage import atomic_write, checked_path


def audit_dependencies(lock: Path) -> dict[str, object]:
    document = tomli.loads(lock.read_text())
    packages = sorted({(p["name"], p["version"]) for p in document["package"] if "registry" in p.get("source", {})})
    provenance = {"lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "checked_at": datetime.now(timezone.utc).isoformat()}
    queries = [{"package": {"name": name, "ecosystem": "PyPI"}, "version": version} for name, version in packages]
    request = Request("https://api.osv.dev/v1/querybatch", data=json.dumps({"queries": queries}).encode(), headers={"Content-Type": "application/json", "User-Agent": "VerdiWM-dependency-audit"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        results = payload["results"]
        if not isinstance(results, list) or len(results) != len(packages):
            raise ValueError("OSV_RESULT_COUNT_MISMATCH")
        findings = []
        for (name, version), result in zip(packages, results):
            if result.get("next_page_token"):
                raise ValueError("OSV_PAGINATED_RESULT_INCOMPLETE")
            for vulnerability in result.get("vulns", []):
                findings.append({"package": name, "version": version, "advisory": vulnerability["id"]})
        return {**provenance, "status": "blocked" if findings else "passed", "scope": "all locked PyPI packages, including optional test groups", "package_count": len(packages), "findings": findings}
    except Exception as exc:
        return {**provenance, "status": "unavailable", "package_count": len(packages), "error_type": type(exc).__name__, "claim_boundary": "No completed advisory lookup; this is not a clean vulnerability result."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dependencies(args.lock)
    target = checked_path(args.report, code="DEPENDENCY_REPORT_SYMLINK")
    atomic_write(target, (json.dumps(report, indent=2) + "\n").encode())
    logging.getLogger(__name__).info("%s", json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
