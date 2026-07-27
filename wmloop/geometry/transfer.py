"""Capability-preserving transfer certificates with explicit abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math

from wmloop.geometry.types import CompileReceipt, GeometryValidationError


@dataclass(frozen=True)
class TransferCertificate:
    descriptor_name: str
    status: str
    compile_pass: bool
    support_overlap: float
    effective_sample_size: float
    alignment_error: float
    sign_agreement: float
    calibrated_lower_bound: float
    goal_threshold: float
    terms: dict[str, bool]
    abstention_reasons: tuple[str, ...]

    @property
    def licensed(self) -> bool:
        return self.status == "licensed"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-transfer-certificate",
            "descriptor_name": self.descriptor_name,
            "status": self.status,
            "compile_pass": self.compile_pass,
            "support_overlap": self.support_overlap,
            "effective_sample_size": self.effective_sample_size,
            "alignment_error": self.alignment_error,
            "sign_agreement": self.sign_agreement,
            "calibrated_lower_bound": self.calibrated_lower_bound,
            "goal_threshold": self.goal_threshold,
            "terms": dict(self.terms),
            "abstention_reasons": list(self.abstention_reasons),
        }


def evaluate_transfer_certificate(
    *,
    receipt: CompileReceipt,
    support_overlap: float,
    effective_sample_size: float,
    alignment_error: float,
    sign_agreement: float,
    calibrated_lower_bound: float,
    minimum_overlap: float,
    minimum_effective_sample_size: float,
    maximum_alignment_error: float,
    minimum_sign_agreement: float,
    goal_threshold: float,
) -> TransferCertificate:
    """Evaluate the six certificate terms from Eq. 6.

    Failed certificates return ``status=abstain``.  They do not throw unless
    the certificate inputs themselves are malformed, which lets the online
    loop cold-start safely on unsupported targets.
    """

    values = {
        "support_overlap": support_overlap,
        "effective_sample_size": effective_sample_size,
        "alignment_error": alignment_error,
        "sign_agreement": sign_agreement,
        "calibrated_lower_bound": calibrated_lower_bound,
        "minimum_overlap": minimum_overlap,
        "minimum_effective_sample_size": minimum_effective_sample_size,
        "maximum_alignment_error": maximum_alignment_error,
        "minimum_sign_agreement": minimum_sign_agreement,
        "goal_threshold": goal_threshold,
    }
    numeric = {name: _finite(value) for name, value in values.items()}
    if not 0.0 <= numeric["support_overlap"] <= 1.0 or not 0.0 <= numeric["minimum_overlap"] <= 1.0:
        raise GeometryValidationError("TRANSFER_OVERLAP_INVALID")
    if not 0.0 <= numeric["sign_agreement"] <= 1.0 or not 0.0 <= numeric["minimum_sign_agreement"] <= 1.0:
        raise GeometryValidationError("TRANSFER_SIGN_AGREEMENT_INVALID")
    if numeric["effective_sample_size"] < 0.0 or numeric["minimum_effective_sample_size"] < 0.0:
        raise GeometryValidationError("TRANSFER_EFFECTIVE_SAMPLE_SIZE_INVALID")
    if numeric["alignment_error"] < 0.0 or numeric["maximum_alignment_error"] < 0.0:
        raise GeometryValidationError("TRANSFER_ALIGNMENT_INVALID")

    terms = {
        "compile": receipt.compiled,
        "overlap": numeric["support_overlap"] >= numeric["minimum_overlap"],
        "effective_sample_size": numeric["effective_sample_size"] >= numeric["minimum_effective_sample_size"],
        "alignment": numeric["alignment_error"] <= numeric["maximum_alignment_error"],
        "sign_agreement": numeric["sign_agreement"] >= numeric["minimum_sign_agreement"],
        "effect_lower_bound": numeric["calibrated_lower_bound"] > numeric["goal_threshold"],
    }
    reasons = tuple(name for name, passed in terms.items() if not passed)
    return TransferCertificate(
        descriptor_name=receipt.descriptor_name,
        status="licensed" if not reasons else "abstain",
        compile_pass=receipt.compiled,
        support_overlap=numeric["support_overlap"],
        effective_sample_size=numeric["effective_sample_size"],
        alignment_error=numeric["alignment_error"],
        sign_agreement=numeric["sign_agreement"],
        calibrated_lower_bound=numeric["calibrated_lower_bound"],
        goal_threshold=numeric["goal_threshold"],
        terms=terms,
        abstention_reasons=reasons,
    )


def _finite(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError("TRANSFER_VALUE_INVALID") from exc
    if not math.isfinite(number):
        raise GeometryValidationError("TRANSFER_VALUE_INVALID")
    return number
