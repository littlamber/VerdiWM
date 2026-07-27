"""InD/OoD gap aggregation with a deterministic worst-condition witness."""

from __future__ import annotations

import math
from collections.abc import Mapping


def build_ood_profile(*, ind_auc: float, ood_auc_by_condition: Mapping[str, float]) -> dict[str, float | str]:
    if not math.isfinite(ind_auc) or not ood_auc_by_condition:
        raise ValueError("OOD_PROBE_INPUT_INVALID")
    normalized = {str(condition): float(value) for condition, value in ood_auc_by_condition.items()}
    if any(not condition or not math.isfinite(value) for condition, value in normalized.items()):
        raise ValueError("OOD_PROBE_INPUT_INVALID")
    condition, ood_auc = min(normalized.items(), key=lambda item: (item[1], item[0]))
    return {"ind_auc": ind_auc, "ood_auc": ood_auc, "gap": ind_auc - ood_auc, "worst_ood_condition": condition}
