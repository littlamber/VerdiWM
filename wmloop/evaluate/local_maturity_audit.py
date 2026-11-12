"""First-contact, CPU-only maturity audit for the Verdi control plane."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wmloop.archive.store import ContentAddressedStore
from wmloop.geometry.community_knowledge import build_knowledge_lifecycle_record
from wmloop.storage import atomic_write, checked_path, sha256_file


EVALUATOR_TIMEOUT_SECONDS = 60


class LocalMaturityAuditError(ValueError):
    """The local maturity audit could not produce trustworthy evidence."""


def run_local_maturity_audit(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Run an end-to-end evaluator and receipt audit without model import or GPU use."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    destination = checked_path(output_root, code="LOCAL_AUDIT_OUTPUT_SYMLINK", error=LocalMaturityAuditError)
    if destination.exists():
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_EXISTS")
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_MUST_BE_OUTSIDE_REPO")
    script = root / "scripts" / "evaluate_psnr_ssim_smoke.py"
    contract = root / "configs" / "evaluators" / "wan22_droid_psnr_ssim_smoke_v1.json"
    if not script.is_file() or script.is_symlink() or not contract.is_file() or contract.is_symlink():
        raise LocalMaturityAuditError("LOCAL_AUDIT_ASSETS_MISSING")

    source_digest_before = _tree_digest(root)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Reserve the final location because receipts bind absolute input paths.
    # If another caller already owns it, mkdir fails outside cleanup ownership.
    destination.mkdir(mode=0o700)
    try:
        workspace = destination
        inputs_root = workspace / "inputs"
        evidence_root = workspace / "evidence"
        inputs_root.mkdir(mode=0o700)
        evidence_root.mkdir(mode=0o700)
        ground_truth = inputs_root / "ground-truth"
        perfect_prediction = inputs_root / "perfect"
        noisy_prediction = inputs_root / "noisy"
        malformed_prediction = inputs_root / "malformed"
        for directory in (ground_truth, perfect_prediction, noisy_prediction, malformed_prediction):
            directory.mkdir()
        frames = [
            np.linspace(0.05, 0.95, 48, dtype=np.float32).reshape(6, 8),
            np.flip(np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(6, 8), axis=1),
        ]
        for index, frame in enumerate(frames):
            np.save(ground_truth / f"frame-{index:03d}.npy", frame)
            np.save(perfect_prediction / f"frame-{index:03d}.npy", frame)
            np.save(noisy_prediction / f"frame-{index:03d}.npy", np.clip(frame + 0.025, 0, 1))
        np.save(malformed_prediction / "frame-000.npy", frames[0][:-1])
        np.save(malformed_prediction / "frame-001.npy", frames[1])

        perfect_output = evidence_root / "perfect-output"
        perfect = _invoke(script, contract, perfect_prediction, ground_truth, perfect_output)
        perfect_receipt = perfect_output / "psnr-ssim-receipt.json"
        verified = _invoke_verify(script, contract, perfect_receipt)
        noisy_output = evidence_root / "noisy-output"
        noisy = _invoke(script, contract, noisy_prediction, ground_truth, noisy_output, baseline=perfect_receipt)
        malformed_output = evidence_root / "malformed-output"
        malformed = _invoke(script, contract, malformed_prediction, ground_truth, malformed_output, expect_failure=True)
        repeated_output = evidence_root / "repeated-output"
        repeated = _invoke(script, contract, perfect_prediction, ground_truth, repeated_output)

        cas_root = workspace / "cas"
        cas = ContentAddressedStore(cas_root)
        receipt_bytes = perfect_receipt.read_bytes()
        receipt_ref = cas.put_bytes(receipt_bytes, media_type="application/json").uri
        cas_roundtrip = cas.read_bytes(receipt_ref) == receipt_bytes
        knowledge = build_knowledge_lifecycle_record(
            action="deprecation",
            subject_kind="psnr_ssim_smoke_receipt",
            subject_id=str(perfect.get("receipt_id")),
            reason="Local maturity audit deposited a verified paired receipt for replay and community projection.",
            authority_ref=receipt_ref,
            evidence_refs=(receipt_ref,),
            root=root,
        )
        knowledge_bytes = json.dumps(knowledge, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        knowledge_ref = cas.put_bytes(knowledge_bytes, media_type="application/json").uri
        knowledge_roundtrip = cas.read_bytes(knowledge_ref) == knowledge_bytes
        source_digest_after = _tree_digest(root)

        gates = [
            {"id": "cli_entrypoint", "state": "pass", "detail": "Evaluator executed through the Python CLI entrypoint."},
            {"id": "perfect_metrics", "state": "pass" if perfect.get("metrics", {}).get("psnr_is_infinite") and float(perfect.get("metrics", {}).get("ssim", 0)) > 0.999999 else "blocked", "detail": "Identical paired frames produce infinite PSNR and SSIM≈1."},
            {"id": "degradation_verdict", "state": "pass" if noisy.get("verdict") in {"HARMFUL", "NULL"} and float(noisy.get("metrics", {}).get("ssim", 1)) < 1 else "blocked", "detail": f"Noisy prediction verdict={noisy.get('verdict')}."},
            {"id": "receipt_verification", "state": "pass" if verified.get("state") == "verified" else "blocked", "detail": "Receipt identity, schema, and input digests verify."},
            {"id": "fail_closed_input", "state": "pass" if malformed.get("state") == "blocked" else "blocked", "detail": "Shape mismatch is rejected before a receipt is published."},
            {"id": "cas_knowledge_roundtrip", "state": "pass" if cas_roundtrip and knowledge_roundtrip and receipt_ref.startswith("cas://sha256/") else "blocked", "detail": "Settled receipt is accepted by the community knowledge contract and round-trips through CAS."},
            {"id": "source_immutability", "state": "pass" if source_digest_before == source_digest_after else "blocked", "detail": "Evaluator run does not modify the Verdi source tree."},
            {"id": "deterministic_metrics", "state": "pass" if perfect.get("metrics") == repeated.get("metrics") else "blocked", "detail": "Repeated evaluation returns identical aggregate metrics."},
        ]
        report = {
            "schema_version": 1,
            "artifact_type": "verdiwm-local-maturity-audit",
            "audit_id": "local-maturity-" + hashlib.sha256((str(root) + str(script)).encode()).hexdigest()[:24],
            "state": "ready" if all(gate["state"] == "pass" for gate in gates) else "partial",
            "operational_state": "ready" if all(gate["state"] == "pass" for gate in gates[:5]) else "blocked",
            "evaluator_state": "ready" if all(gate["state"] == "pass" for gate in gates[1:5]) else "blocked",
            "receipt_state": "ready" if gates[3]["state"] == "pass" else "blocked",
            "knowledge_state": "ready" if gates[5]["state"] == "pass" else "blocked",
            "passed_gate_count": sum(gate["state"] == "pass" for gate in gates),
            "blocked_gate_count": sum(gate["state"] == "blocked" for gate in gates),
            "gates": gates,
            "artifacts": {
                "evaluator_contract": str(contract),
                "evaluator_contract_sha256": _sha256(contract),
                "receipt_ref": receipt_ref,
                "knowledge_ref": knowledge_ref,
                "knowledge_id": knowledge["lifecycle_id"],
                "receipt_id": perfect.get("receipt_id"),
                "noisy_verdict": noisy.get("verdict"),
            },
            "side_effects": {"gpu_execution_started": False, "source_modified": source_digest_before != source_digest_after, "model_import_executed": False},
            "claim_boundary": "CPU control-plane and paired PSNR/SSIM receipt closure only; this does not establish real Wan/DROID execution, IRG transfer prediction, or scientific improvement.",
            "next_work": ["Run an explicitly confirmed real-model campaign with a frozen target-side evaluator before making scientific claims."],
        }
        _write_report(destination, report)
        return {**report, "manifest_path": str(destination / "manifest.json")}
    except BaseException:
        # Ctrl-C must also leave this owned output available for a clean retry.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _invoke(script: Path, contract: Path, prediction: Path, target: Path, output: Path, *, baseline: Path | None = None, expect_failure: bool = False) -> dict[str, Any]:
    command = [sys.executable, str(script), "--predicted-dir", str(prediction), "--ground-truth-dir", str(target), "--output-root", str(output), "--contract", str(contract)]
    if baseline is not None:
        command.extend(["--baseline-receipt", str(baseline)])
    completed = _run_evaluator(command, script, output.parent)
    if expect_failure:
        if completed.returncode == 0:
            raise LocalMaturityAuditError("LOCAL_AUDIT_EXPECTED_FAILURE_MISSING")
        return _response_object(completed.stderr.strip() or completed.stdout.strip(), "LOCAL_AUDIT_FAILURE_UNPARSEABLE")
    if completed.returncode != 0:
        raise LocalMaturityAuditError(f"LOCAL_AUDIT_EVALUATOR_FAILED:{completed.stderr.strip()}")
    return _response_object(completed.stdout.strip())


def _invoke_verify(script: Path, contract: Path, receipt: Path) -> dict[str, Any]:
    completed = _run_evaluator([sys.executable, str(script), "--verify-receipt", str(receipt), "--contract", str(contract)], script, receipt.parent.parent)
    if completed.returncode != 0:
        raise LocalMaturityAuditError(f"LOCAL_AUDIT_RECEIPT_VERIFY_FAILED:{completed.stderr.strip() or completed.stdout.strip()}")
    return _response_object(completed.stdout.strip())


def _response_object(raw: str, error_code: str = "LOCAL_AUDIT_OUTPUT_UNPARSEABLE") -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalMaturityAuditError(error_code) from exc
    if not isinstance(value, dict):
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_OBJECT_REQUIRED")
    return value


def _run_evaluator(command: list[str], script: Path, workspace: Path) -> subprocess.CompletedProcess[str]:
    """Bound the bundled CPU evaluator and reap it on timeout or cancellation."""
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(workspace / ".pycache")})
    try:
        return subprocess.run(command, cwd=script.parents[1], env=environment, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, check=False, timeout=EVALUATOR_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise LocalMaturityAuditError("LOCAL_AUDIT_EVALUATOR_TIMEOUT") from exc


def _tree_digest(root: Path) -> str:
    """Hash source/configuration without traversing generated runtime trees."""
    digest = hashlib.sha256()
    transient = {".git", ".venv", "__pycache__", ".pytest_cache", ".hypothesis", ".ruff_cache", "node_modules"}
    generated_roots = {"artifacts", "results", "runs", "outputs", "logs", "cache", "build", "dist", "htmlcov", ".verdiwm", ".audit-state"}
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        directories[:] = sorted(name for name in directories if name not in transient and not (parent == root and name in generated_roots) and not (parent / name).is_symlink())
        for name in sorted(files):
            path = parent / name
            if path.is_symlink() or path.suffix in {".pyc", ".pyo", ".log"} or name == ".coverage":
                continue
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + sha256_file(path).encode())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _write_report(destination: Path, report: dict[str, object]) -> None:
    """Publish report files atomically and the completion manifest last."""
    atomic_write(destination / "local-maturity-audit.json", (json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode())
    lines = ["# VerdiWM Local Maturity Audit", "", f"- State: `{report['state']}`", f"- Operational state: `{report['operational_state']}`", f"- Evaluator state: `{report['evaluator_state']}`", f"- Receipt state: `{report['receipt_state']}`", f"- Knowledge state: `{report['knowledge_state']}`", "", "| Gate | State | Detail |", "| --- | --- | --- |"]
    lines.extend(f"| `{gate['id']}` | `{gate['state']}` | {gate['detail']} |" for gate in report["gates"])
    lines.extend(["", "## Claim boundary", "", str(report["claim_boundary"]), ""])
    atomic_write(destination / "local-maturity-audit.md", "\n".join(lines).encode())
    manifest = {"schema_version": 1, "artifact_type": "verdiwm-local-maturity-audit-manifest", "state": report["state"], "audit_id": report["audit_id"], "report_path": str(destination / "local-maturity-audit.json")}
    atomic_write(destination / "manifest.json", (json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode())
