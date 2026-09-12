"""First-contact, CPU-only maturity audit for the Verdi control plane."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from wmloop.archive.store import ContentAddressedStore
from wmloop.geometry.community_knowledge import build_knowledge_lifecycle_record


class LocalMaturityAuditError(ValueError):
    """The local maturity audit could not produce trustworthy evidence."""


def run_local_maturity_audit(*, repo_root: Path, output_root: Path) -> dict[str, object]:
    """Run an end-to-end evaluator and receipt audit without model import or GPU use."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_EXISTS")
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_MUST_BE_OUTSIDE_REPO")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    script = root / "scripts" / "evaluate_psnr_ssim_smoke.py"
    contract = root / "configs" / "evaluators" / "wan22_droid_psnr_ssim_smoke_v1.json"
    if not script.is_file() or script.is_symlink() or not contract.is_file() or contract.is_symlink():
        raise LocalMaturityAuditError("LOCAL_AUDIT_ASSETS_MISSING")

    source_digest_before = _tree_digest(root)
    with tempfile.TemporaryDirectory(prefix="verdi-local-audit-", dir=str(destination.parent)) as temporary:
        workspace = Path(temporary)
        ground_truth = workspace / "ground-truth"
        perfect_prediction = workspace / "perfect"
        noisy_prediction = workspace / "noisy"
        malformed_prediction = workspace / "malformed"
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

        perfect_output = workspace / "perfect-output"
        perfect = _invoke(script, contract, perfect_prediction, ground_truth, perfect_output)
        perfect_receipt = perfect_output / "psnr-ssim-receipt.json"
        verified = _invoke_verify(script, contract, perfect_receipt)
        noisy_output = workspace / "noisy-output"
        noisy = _invoke(script, contract, noisy_prediction, ground_truth, noisy_output, baseline=perfect_receipt)
        malformed_output = workspace / "malformed-output"
        malformed = _invoke(script, contract, malformed_prediction, ground_truth, malformed_output, expect_failure=True)
        repeated_output = workspace / "repeated-output"
        repeated = _invoke(script, contract, perfect_prediction, ground_truth, repeated_output)

        cas_root = destination.parent / f".{destination.name}.cas"
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
        shutil.rmtree(cas_root, ignore_errors=True)

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


def _invoke(script: Path, contract: Path, prediction: Path, target: Path, output: Path, *, baseline: Path | None = None, expect_failure: bool = False) -> dict[str, Any]:
    command = [sys.executable, str(script), "--predicted-dir", str(prediction), "--ground-truth-dir", str(target), "--output-root", str(output), "--contract", str(contract)]
    if baseline is not None:
        command.extend(["--baseline-receipt", str(baseline)])
    child_environment = dict(__import__("os").environ)
    child_environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(output.parent / ".pycache")})
    completed = subprocess.run(command, cwd=str(script.parents[1]), env=child_environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    if expect_failure:
        if completed.returncode == 0:
            raise LocalMaturityAuditError("LOCAL_AUDIT_EXPECTED_FAILURE_MISSING")
        try:
            return json.loads(completed.stderr.strip() or completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise LocalMaturityAuditError("LOCAL_AUDIT_FAILURE_UNPARSEABLE") from exc
    if completed.returncode != 0:
        raise LocalMaturityAuditError(f"LOCAL_AUDIT_EVALUATOR_FAILED:{completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise LocalMaturityAuditError("LOCAL_AUDIT_OUTPUT_UNPARSEABLE") from exc


def _invoke_verify(script: Path, contract: Path, receipt: Path) -> dict[str, Any]:
    child_environment = dict(__import__("os").environ)
    child_environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(receipt.parent.parent / ".pycache")})
    completed = subprocess.run([sys.executable, str(script), "--verify-receipt", str(receipt), "--contract", str(contract)], cwd=str(script.parents[1]), env=child_environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise LocalMaturityAuditError(f"LOCAL_AUDIT_RECEIPT_VERIFY_FAILED:{completed.stderr.strip() or completed.stdout.strip()}")
    return json.loads(completed.stdout.strip())


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + _sha256(path).encode())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(destination: Path, report: dict[str, object]) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        (temporary / "local-maturity-audit.json").write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        lines = ["# VerdiWM Local Maturity Audit", "", f"- State: `{report['state']}`", f"- Operational state: `{report['operational_state']}`", f"- Evaluator state: `{report['evaluator_state']}`", f"- Receipt state: `{report['receipt_state']}`", f"- Knowledge state: `{report['knowledge_state']}`", "", "| Gate | State | Detail |", "| --- | --- | --- |"]
        lines.extend(f"| `{gate['id']}` | `{gate['state']}` | {gate['detail']} |" for gate in report["gates"])
        lines.extend(["", "## Claim boundary", "", str(report["claim_boundary"]), ""])
        (temporary / "local-maturity-audit.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {"schema_version": 1, "artifact_type": "verdiwm-local-maturity-audit-manifest", "state": report["state"], "audit_id": report["audit_id"], "report_path": str(destination / "local-maturity-audit.json")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        import os

        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
