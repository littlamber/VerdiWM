"""Audit source and wheel archives and verify the sdist rebuilds the same wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile
import zipfile

from scripts.export.build_github_staging import MAX_TEXT_BYTES, audit_release_tree

REQUIRED_WHEEL = {
    "wmloop/__init__.py",
    "wmloop/control/local_http.py",
    "wmloop/experiments/formal_visualization.py",
    "wmloop/web/workbench.js",
    "wmloop/cli.py",
    "wmloop/control/workbench.py",
    "wmloop/control/research_state.py",
    "wmloop/control/mechanism_hypothesis.py",
    "wmloop/control/method_realization.py",
    "wmloop/control/open_method_study.py",
    "wmloop/control/open_method_generation.py",
    "wmloop/execute/open_method_runtime.py",
    "wmloop/execute/open_method_study_runner.py",
    "wmloop/control/method_calibration.py",
    "configs/schemas/research_state.schema.json",
    "configs/schemas/mechanism_hypothesis.schema.json",
    "configs/schemas/method_implementation_validation.schema.json",
    "configs/schemas/open_method_composition.schema.json",
    "configs/schemas/open_method_evaluation.schema.json",
    "configs/schemas/open_method_study_execution.schema.json",
    "configs/schemas/open_method_verifier.schema.json",
    "wmloop/control/onboarding.py",
    "wmloop/execute/autonomous_pipeline.py",
    "wmloop/execute/experiment_scheduler.py",
    "wmloop/experiments/community_bundle.py",
    "wmloop/experiments/community_export.py",
    "configs/schemas/goal_spec.schema.json",
    "configs/schemas/adapter_profile.schema.json",
    "configs/schemas/model_batch_request.schema.json",
    "configs/schemas/model_batch_plan.schema.json",
    "configs/schemas/model_batch_execution.schema.json",
    "configs/schemas/model_batch_status.schema.json",
    "configs/schemas/community_bundle.schema.json",
    "configs/schemas/community_bundle_signature.schema.json",
    "configs/schemas/community_export.schema.json",
    "configs/schemas/settled_evidence.schema.json",
    "wmloop/archive/community_projection.py",
    "wmloop/control/campaign_repository.py",
    "configs/retrieval/mechanism_tag_ontology_v1.json",
}


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("ARCHIVE_PATH_INVALID")
    return Path(*path.parts)


def audit_archives(wheel: Path, sdist: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="verdiwm-archive-") as directory:
        root = Path(directory)
        wheel_root = root / "wheel"
        wheel_root.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            missing = sorted(REQUIRED_WHEEL - names)
            if missing:
                raise ValueError("WHEEL_FILES_MISSING:" + ";".join(missing))
            if len(names) != len(archive.infolist()):
                raise ValueError("WHEEL_DUPLICATE_MEMBERS")
            expected = {}
            for info in archive.infolist():
                relative = _relative(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16) or info.file_size > MAX_TEXT_BYTES:
                    raise ValueError("WHEEL_UNSAFE_MEMBER")
                if info.is_dir():
                    continue
                data = archive.read(info)
                expected[info.filename] = hashlib.sha256(data).hexdigest()
                target = wheel_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        findings = audit_release_tree(wheel_root)
        if findings:
            raise ValueError("WHEEL_AUDIT_FAILED:" + ";".join(findings))
        source_root = root / "sdist"
        source_root.mkdir()
        with tarfile.open(sdist, "r:gz") as archive:
            seen = set()
            for member in archive:
                relative = _relative(member.name)
                if member.name in seen or (not member.isdir() and not member.isfile()) or member.size > MAX_TEXT_BYTES:
                    raise ValueError("SDIST_UNSAFE_MEMBER")
                seen.add(member.name)
                target = source_root / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as handle:
                        target.write_bytes(handle.read())
        children = list(source_root.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("SDIST_ROOT_INVALID")
        source = children[0]
        findings = audit_release_tree(source)
        if findings:
            raise ValueError("SDIST_AUDIT_FAILED:" + ";".join(findings))
        for required in ("AGENTS.md", "THIRD_PARTY_NOTICES.md", "uv.lock", "tests/test_release_security.py", "scripts/ci/check_control_plane.sh"):
            if not (source / required).is_file():
                raise ValueError("SDIST_FILE_MISSING:" + required)
        rebuilt = root / "rebuilt"
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(rebuilt), str(source)], check=True)
        wheels = list(rebuilt.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("SDIST_WHEEL_MISSING")
        with zipfile.ZipFile(wheels[0]) as archive:
            actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}
        if actual != expected:
            raise ValueError("SDIST_WHEEL_CONTENT_MISMATCH")
        return {"status": "passed", "wheel_files": len(expected), "sdist_files": len(seen), "sdist_rebuild_matches": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_archives(args.wheel, args.sdist)
    from wmloop.storage import atomic_write, checked_path
    target = checked_path(args.report, code="ARCHIVE_REPORT_SYMLINK")
    atomic_write(target, (json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
