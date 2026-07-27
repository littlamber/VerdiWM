"""Context-local Intervention-Effect Memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from wmloop.geometry.types import GeometryValidationError


_STATUSES = {"confirmed", "null", "rejected", "interaction"}


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
