"""Verification of conformance admissions used by external model plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding import (
    OnboardingError,
    compute_asset_fingerprint,
    compute_source_revision,
    compute_source_tree_revision,
)


class OnboardingAdmissionError(ValueError):
    """A conformance admission is missing, stale, or invalid."""


def verify_onboarding_admission(
    admission: object, *, expected_repo_root: Path
) -> dict[str, object]:
    """Verify a hash-bound PASS receipt and the current source revision."""

    if not isinstance(admission, Mapping):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_REQUIRED")
    allowed = {"receipt_path", "receipt_sha256", "onboarding_report_sha256"}
    if set(admission) != allowed:
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_FIELDS_INVALID")
    receipt_path = Path(str(admission.get("receipt_path", ""))).expanduser().resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_RECEIPT_MISSING")
    payload = receipt_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != admission.get("receipt_sha256"):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_RECEIPT_HASH_MISMATCH")
    try:
        receipt = json.loads(payload)
        if not isinstance(receipt, dict):
            raise ValueError("receipt object required")
        validate_document("model_conformance_receipt", receipt)
    except (json.JSONDecodeError, ValueError, ContractValidationError) as exc:
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_RECEIPT_INVALID") from exc
    if (
        receipt.get("verdict") != "PASS"
        or receipt.get("optimization_launch_allowed") is not True
    ):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_NOT_PASSING")
    if receipt.get("onboarding_report_sha256") != admission.get(
        "onboarding_report_sha256"
    ):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_REPORT_HASH_MISMATCH")
    repo = Path(str(receipt.get("repo_root", ""))).resolve()
    if repo != Path(expected_repo_root).resolve():
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_REPOSITORY_MISMATCH")
    if compute_source_revision(repo) != receipt.get("source_revision"):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_SOURCE_DRIFT")
    if compute_source_tree_revision(repo) != receipt.get("source_tree_revision"):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_SOURCE_TREE_DRIFT")
    verify_receipt_asset_bindings(receipt)
    return receipt


def verify_receipt_asset_bindings(receipt: Mapping[str, object]) -> None:
    """Reject a PASS receipt whose evaluator assets no longer match."""

    bindings = receipt.get("asset_bindings")
    if not isinstance(bindings, list):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_ASSET_BINDINGS_INVALID")
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise OnboardingAdmissionError(
                "ONBOARDING_ADMISSION_ASSET_BINDINGS_INVALID"
            )
        parameter = str(binding.get("parameter", ""))
        path_value = binding.get("resolved_path")
        expected = binding.get("observed_fingerprint")
        if (
            binding.get("state") != "bound"
            or not parameter
            or not isinstance(path_value, str)
            or not path_value
            or not isinstance(expected, str)
            or not expected
        ):
            raise OnboardingAdmissionError(
                f"ONBOARDING_ADMISSION_ASSET_BINDING_INVALID:{parameter}"
            )
        try:
            observed = compute_asset_fingerprint(Path(path_value))
        except OnboardingError as exc:
            raise OnboardingAdmissionError(
                f"ONBOARDING_ADMISSION_ASSET_MISSING:{parameter}"
            ) from exc
        if observed != expected:
            raise OnboardingAdmissionError(
                f"ONBOARDING_ADMISSION_ASSET_DRIFT:{parameter}"
            )


def admission_from_manifest(conformance_root: Path) -> dict[str, str]:
    """Build an admission reference from a verified conformance directory."""

    root = Path(conformance_root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_MANIFEST_INVALID") from exc
    if not isinstance(manifest, dict):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_MANIFEST_INVALID")
    receipt_path = root / "conformance-receipt.json"
    admission = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": str(manifest.get("receipt_sha256", "")),
        "onboarding_report_sha256": "",
    }
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_RECEIPT_INVALID") from exc
    if not isinstance(receipt, dict):
        raise OnboardingAdmissionError("ONBOARDING_ADMISSION_RECEIPT_INVALID")
    admission["onboarding_report_sha256"] = str(
        receipt.get("onboarding_report_sha256", "")
    )
    return admission
