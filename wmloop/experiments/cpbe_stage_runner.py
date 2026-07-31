"""Run reproducible CPBE static and offline admission stages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CPBEStageRunnerError(RuntimeError):
    """A CPBE stage could not produce trustworthy admission evidence."""


def publish_static_offline_receipts(
    *,
    plan_path: Path,
    repo_root: Path,
    output_root: Path,
    python_executable: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Recheck materialized probes and publish hash-bound stage receipts."""

    root = Path(repo_root).resolve(strict=True)
    plan_file = Path(plan_path).resolve(strict=True)
    plan = _load_object(plan_file)
    if plan.get("artifact_type") != "verdiwm-cpbe-plan" or plan.get("state") != "ready":
        raise CPBEStageRunnerError("CPBE_STAGE_PLAN_NOT_READY")
    selected = _mapping_sequence(plan.get("selected_work_orders"), "CPBE_STAGE_WORK_ORDERS_INVALID")
    if not selected:
        raise CPBEStageRunnerError("CPBE_STAGE_WORK_ORDERS_EMPTY")
    interpreter = Path(python_executable or sys.executable).absolute()
    if not interpreter.is_file():
        raise CPBEStageRunnerError("CPBE_STAGE_PYTHON_MISSING")

    evidence_files: dict[str, bytes] = {}
    receipts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for selected_order in selected:
        probe_id = _text(selected_order, "probe_id")
        work_order_path = plan_file.parent / "work-orders" / f"{probe_id}.json"
        work_order_bytes = work_order_path.read_bytes()
        work_order = _load_object(work_order_path)
        if work_order != selected_order:
            raise CPBEStageRunnerError(f"CPBE_STAGE_WORK_ORDER_PLAN_MISMATCH:{probe_id}")

        descriptor_path = root / "configs" / "probes" / "staging" / f"{probe_id}.json"
        module_path = root / "wmloop" / "diagnose" / "probes" / f"{probe_id}.py"
        test_path = root / "tests" / f"test_{probe_id}.py"
        descriptor = _load_object(descriptor_path)
        _validate_descriptor(descriptor=descriptor, work_order=work_order, probe_id=probe_id)
        module_name, callable_name = _descriptor_entrypoint(descriptor)
        if module_name != f"wmloop.diagnose.probes.{probe_id}":
            raise CPBEStageRunnerError(f"CPBE_STAGE_MODULE_PATH_MISMATCH:{probe_id}")
        if not module_path.is_file() or not test_path.is_file():
            raise CPBEStageRunnerError(f"CPBE_STAGE_IMPLEMENTATION_MISSING:{probe_id}")
        module_spec = importlib.util.spec_from_file_location(f"_verdiwm_stage_{probe_id}", module_path)
        if module_spec is None or module_spec.loader is None:
            raise CPBEStageRunnerError(f"CPBE_STAGE_MODULE_LOAD_FAILED:{probe_id}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if not callable(getattr(module, callable_name, None)):
            raise CPBEStageRunnerError(f"CPBE_STAGE_CALLABLE_MISSING:{probe_id}")

        completed = subprocess.run(
            [str(interpreter), "-m", "pytest", "-q", str(test_path.relative_to(root))],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_payload = completed.stdout.encode("utf-8")
        if completed.returncode != 0:
            raise CPBEStageRunnerError(f"CPBE_STAGE_OFFLINE_TEST_FAILED:{probe_id}\n{completed.stdout}")

        prefix = f"evidence/{probe_id}"
        payloads = {
            f"{prefix}/work-order.json": work_order_bytes,
            f"{prefix}/descriptor.json": descriptor_path.read_bytes(),
            f"{prefix}/probe.py": module_path.read_bytes(),
            f"{prefix}/test_probe.py": test_path.read_bytes(),
            f"{prefix}/pytest.log": log_payload,
        }
        evidence_files.update(payloads)
        static_paths = [
            f"{prefix}/work-order.json",
            f"{prefix}/descriptor.json",
            f"{prefix}/probe.py",
        ]
        offline_paths = [f"{prefix}/test_probe.py", f"{prefix}/pytest.log"]
        receipts.extend(
            [
                _receipt(probe_id=probe_id, stage="static", paths=static_paths, payloads=payloads),
                _receipt(probe_id=probe_id, stage="offline", paths=offline_paths, payloads=payloads),
            ]
        )
        summaries.append(
            {
                "probe_id": probe_id,
                "module": module_name,
                "callable": callable_name,
                "test_return_code": completed.returncode,
                "work_order_sha256": hashlib.sha256(work_order_bytes).hexdigest(),
                "descriptor_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
                "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            }
        )

    receipt_bytes = b"".join(canonical_json(receipt) for receipt in receipts)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-static-offline-stage-report",
        "state": "ready",
        "experiment_id": plan["experiment_id"],
        "probe_count": len(summaries),
        "receipt_count": len(receipts),
        "probes": summaries,
        "next_stage": "canary",
        "claim_boundary": (
            "Static and offline admission only. Runtime, locality, collision separation, "
            "selector regret, accepted coverage, repair benefit, and transfer remain unmeasured."
        ),
    }
    return write_bundle(
        output_root=output_root,
        files={
            **evidence_files,
            "cpbe-stage-receipts.jsonl": receipt_bytes,
            "stage-report.json": canonical_json(report),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cpbe-static-offline-stage-manifest",
            "state": "ready",
            "experiment_id": plan["experiment_id"],
            "probe_count": len(summaries),
            "receipt_count": len(receipts),
            "receipt_path": "cpbe-stage-receipts.jsonl",
            "report_path": "stage-report.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_descriptor(
    *, descriptor: Mapping[str, Any], work_order: Mapping[str, Any], probe_id: str
) -> None:
    if (
        descriptor.get("probe_id") != probe_id
        or descriptor.get("role") != "diagnostic"
        or descriptor.get("verdict_exposure_allowed") is not False
        or descriptor.get("program") != work_order.get("program")
    ):
        raise CPBEStageRunnerError(f"CPBE_STAGE_DESCRIPTOR_MISMATCH:{probe_id}")
    admission = descriptor.get("admission") or descriptor.get("admission_state")
    if not isinstance(admission, Mapping):
        raise CPBEStageRunnerError(f"CPBE_STAGE_DESCRIPTOR_ADMISSION_MISSING:{probe_id}")


def _descriptor_entrypoint(descriptor: Mapping[str, Any]) -> tuple[str, str]:
    implementation = descriptor.get("implementation")
    if isinstance(implementation, Mapping):
        module = implementation.get("module")
        callable_name = implementation.get("measurement_callable")
    else:
        module = descriptor.get("module")
        callable_name = descriptor.get("callable")
    if not isinstance(module, str) or not module or not isinstance(callable_name, str) or not callable_name:
        raise CPBEStageRunnerError("CPBE_STAGE_DESCRIPTOR_ENTRYPOINT_INVALID")
    return module, callable_name


def _receipt(
    *, probe_id: str, stage: str, paths: Sequence[str], payloads: Mapping[str, bytes]
) -> dict[str, object]:
    artifacts = [
        {
            "path": path,
            "sha256": hashlib.sha256(payloads[path]).hexdigest(),
            "size_bytes": len(payloads[path]),
        }
        for path in paths
    ]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-stage-receipt",
        "probe_id": probe_id,
        "stage": stage,
        "passed": True,
        "metrics": {},
        "evidence_refs": list(paths),
        "evidence_artifacts": artifacts,
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CPBEStageRunnerError(f"CPBE_STAGE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise CPBEStageRunnerError(f"CPBE_STAGE_JSON_INVALID:{path}")
    return value


def _mapping_sequence(value: object, error: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CPBEStageRunnerError(error)
    return list(value)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise CPBEStageRunnerError(f"CPBE_STAGE_TEXT_INVALID:{key}")
    return item


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = publish_static_offline_receipts(
        plan_path=args.plan,
        repo_root=args.repo_root,
        output_root=args.output_root,
        python_executable=args.python,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
