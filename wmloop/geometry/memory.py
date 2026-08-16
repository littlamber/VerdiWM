"""Context-local Intervention-Effect Memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wmloop.geometry.types import GeometryValidationError


_STATUSES = {"confirmed", "null", "rejected", "interaction"}
_TRANSFER_STATES = {"local_only", "licensed_prior", "abstained"}


@dataclass(frozen=True)
class EffectContext:
    campaign_id: str
    backbone_family: str
    capability_class: str
    goal_schema: str
    outcome_schema: str
    chart_id: str
    data_regime: str
    horizons: tuple[int, ...]

    def __post_init__(self) -> None:
        fields = (
            self.campaign_id,
            self.backbone_family,
            self.capability_class,
            self.goal_schema,
            self.outcome_schema,
            self.chart_id,
            self.data_regime,
        )
        if any(not value for value in fields):
            raise GeometryValidationError("EFFECT_CONTEXT_IDENTITY_INVALID")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise GeometryValidationError("EFFECT_CONTEXT_HORIZONS_INVALID")


@dataclass(frozen=True)
class EffectRecord:
    record_id: str
    primitive: str
    context: EffectContext
    status: str
    mean_effect: float
    standard_error: float
    lower_bound: float
    goal_threshold: float
    validity_gates: Mapping[str, bool]
    replication_count: int
    evidence_refs: tuple[str, ...]
    interaction_with: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.record_id or not self.primitive or self.status not in _STATUSES:
            raise GeometryValidationError("EFFECT_RECORD_IDENTITY_INVALID")
        for value in (self.mean_effect, self.standard_error, self.lower_bound, self.goal_threshold):
            if not math.isfinite(float(value)):
                raise GeometryValidationError("EFFECT_RECORD_VALUE_INVALID")
        if self.standard_error < 0.0 or self.replication_count < 1:
            raise GeometryValidationError("EFFECT_RECORD_UNCERTAINTY_INVALID")
        if not self.validity_gates or not self.evidence_refs:
            raise GeometryValidationError("EFFECT_RECORD_EVIDENCE_MISSING")
        if self.status == "confirmed":
            if self.lower_bound <= self.goal_threshold:
                raise GeometryValidationError("CONFIRMED_EFFECT_LOWER_BOUND_NOT_POSITIVE")
            if not all(self.validity_gates.values()):
                raise GeometryValidationError("CONFIRMED_EFFECT_GATE_FAILED")
            if self.replication_count < 2:
                raise GeometryValidationError("CONFIRMED_EFFECT_REPLICATION_MISSING")
        if self.status == "interaction" and not self.interaction_with:
            raise GeometryValidationError("INTERACTION_EFFECT_PAIR_MISSING")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = 1
        payload["artifact_type"] = "verdiwm-intervention-effect-record"
        return payload


class EffectMemory:
    """Deduplicated collection retaining positive, null, and harmful effects."""

    def __init__(self, records: Iterable[EffectRecord] = ()) -> None:
        self._records: dict[str, EffectRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: EffectRecord) -> None:
        if record.record_id in self._records:
            raise GeometryValidationError(f"EFFECT_RECORD_DUPLICATE:{record.record_id}")
        self._records[record.record_id] = record

    def records(self) -> tuple[EffectRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def query(
        self,
        *,
        primitive: str | None = None,
        backbone_family: str | None = None,
        capability_class: str | None = None,
        chart_id: str | None = None,
        status: str | None = None,
    ) -> tuple[EffectRecord, ...]:
        if status is not None and status not in _STATUSES:
            raise GeometryValidationError("EFFECT_QUERY_STATUS_INVALID")
        return tuple(
            record
            for record in self.records()
            if (primitive is None or record.primitive == primitive)
            and (backbone_family is None or record.context.backbone_family == backbone_family)
            and (capability_class is None or record.context.capability_class == capability_class)
            and (chart_id is None or record.context.chart_id == chart_id)
            and (status is None or record.status == status)
        )

    def write_jsonl(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(
            json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for record in self.records()
        )
        destination.write_text(text, encoding="utf-8")
        return destination

    def transferable_experiences(
        self,
        *,
        certificates: Mapping[str, object] | None = None,
        applicability: Mapping[str, object] | None = None,
        anti_conditions: Sequence[str] = (),
    ) -> tuple[dict[str, object], ...]:
        """Compile bounded transfer projections without changing memory state."""

        supplied = certificates or {}
        return tuple(
            build_transferable_experience(
                record,
                transfer_certificate=supplied.get(record.record_id),
                applicability=applicability,
                anti_conditions=anti_conditions,
            )
            for record in self.records()
        )

    def write_transferable_jsonl(
        self,
        path: Path,
        *,
        certificates: Mapping[str, object] | None = None,
        applicability: Mapping[str, object] | None = None,
        anti_conditions: Sequence[str] = (),
    ) -> Path:
        """Persist deterministic transfer projections as a derived view."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.transferable_experiences(
            certificates=certificates,
            applicability=applicability,
            anti_conditions=anti_conditions,
        )
        text = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
        destination.write_text(text, encoding="utf-8")
        return destination


def build_transferable_experience(
    effect: EffectRecord,
    *,
    transfer_certificate: object | None = None,
    applicability: Mapping[str, object] | None = None,
    anti_conditions: Sequence[str] = (),
) -> dict[str, object]:
    """Compile one effect into a migration-aware, non-authoritative record.

    A positive effect is never enough to license reuse. The existing transfer
    certificate must be present, licensed, and have every declared term pass.
    Null/rejected effects remain useful negative or boundary evidence, but are
    never upgraded to positive transfer priors.
    """

    if not isinstance(effect, EffectRecord):
        raise GeometryValidationError("TRANSFER_EXPERIENCE_EFFECT_INVALID")
    if any(not isinstance(value, str) or not value for value in anti_conditions):
        raise GeometryValidationError("TRANSFER_EXPERIENCE_ANTI_CONDITIONS_INVALID")
    cert = _certificate_payload(transfer_certificate)
    if cert is not None:
        status = cert.get("status")
        terms = cert.get("terms")
        if (
            cert.get("artifact_type") != "verdiwm-transfer-certificate"
            or status not in {"licensed", "abstain"}
            or not isinstance(terms, Mapping)
            or not terms
        ):
            raise GeometryValidationError("TRANSFER_EXPERIENCE_CERTIFICATE_INVALID")
        if any(not isinstance(value, bool) for value in terms.values()):
            raise GeometryValidationError("TRANSFER_EXPERIENCE_CERTIFICATE_TERMS_INVALID")
    licensed = (
        effect.status == "confirmed"
        and cert is not None
        and cert.get("status") == "licensed"
        and all(bool(value) for value in cert.get("terms", {}).values())
    )
    transfer_state = "licensed_prior" if licensed else (
        "abstained" if cert is not None and cert.get("status") == "abstain" else "local_only"
    )
    context = asdict(effect.context)
    default_applicability = {
        key: context[key]
        for key in (
            "backbone_family",
            "capability_class",
            "goal_schema",
            "outcome_schema",
            "data_regime",
            "chart_id",
        )
    }
    applicability_payload = dict(applicability or default_applicability)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-transferable-experience",
        "state": "ready",
        "primitive": effect.primitive,
        "source_effect_record_id": effect.record_id,
        "source_context": context,
        "effect": {
            "status": effect.status,
            "mean_effect": effect.mean_effect,
            "standard_error": effect.standard_error,
            "lower_bound": effect.lower_bound,
            "goal_threshold": effect.goal_threshold,
            "replication_count": effect.replication_count,
        },
        "applicability": applicability_payload,
        "anti_conditions": list(anti_conditions),
        "transfer_state": transfer_state,
        "transfer_authority": "licensed_prior" if licensed else "ranking_only",
        "evidence_refs": list(effect.evidence_refs),
        "transfer_certificate": cert,
        "claim_boundary": (
            "This is a derived transfer projection. A licensed prior permits a "
            "bounded reuse experiment; it does not replace target-side validation."
        ),
    }
    body["experience_id"] = "experience-" + hashlib.sha256(
        _canonical_json(body)
    ).hexdigest()[:24]
    return body


def _certificate_payload(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise GeometryValidationError("TRANSFER_EXPERIENCE_CERTIFICATE_INVALID")
    return dict(value)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError("TRANSFER_EXPERIENCE_PAYLOAD_INVALID") from exc
