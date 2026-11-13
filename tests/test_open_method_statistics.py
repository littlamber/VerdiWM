import copy
import json

import pytest

from test_open_method_study_runner import _runner_fixture
from wmloop.verify.open_method_statistics import paired_statistics


def _inputs(tmp_path):
    fixture = _runner_fixture(tmp_path)
    study = json.loads((fixture["study"] / "study.json").read_text())
    verifier = json.loads(fixture["verifier"].read_text())
    methods = {arm["role"]: arm["method_id"] for arm in study["arms"]}
    observations = []
    for role in methods:
        for seed in study["confirmation_seeds"]:
            effect = {"baseline": [0, 0], "source_only": [1, 3],
                      "target_only": [2, 4], "combined": [4, 8]}[role]
            observations.append({
                "schema_version": 1, "artifact_type": "verdiwm-open-method-evaluation",
                "study_id": study["study_id"], "method_id": methods[role], "role": role,
                "phase": "confirmation", "seed": seed, "verifier_id": verifier["verifier_id"],
                "validity_gates": {"data_bound": True},
                "unit_metrics": [{"unit_id": unit, "metrics": {"identity_error": -value,
                                  "action_fidelity": 1.0}} for unit, value in zip(("a", "b"), effect)],
                "claim_boundary": "Paired observation, not a settled effect claim.",
            })
    return {"study": study, "verifier": verifier, "observations": observations,
            "phase": "confirmation", "expected_units": ["a", "b"]}


def test_statistics_uses_paired_episodes_and_small_sample_interval(tmp_path):
    result = paired_statistics(**_inputs(tmp_path))
    effects = result["metrics"]["identity_error"]["effects_vs_baseline"]
    assert effects["source_only"]["mean"] == pytest.approx(2)
    assert effects["source_only"]["standard_error"] == pytest.approx(1)
    assert effects["source_only"]["lower_bound"] == pytest.approx(2 - 12.7064)
    assert result["metrics"]["identity_error"]["interaction"]["mean"] == pytest.approx(1)
    assert result["metrics"]["action_fidelity"]["effects_vs_baseline"]["combined"]["mean"] == 0


@pytest.mark.parametrize("mutation, reason", [
    ("duplicate", "IDENTITY_INVALID"), ("missing", "INCOMPLETE"),
    ("identity", "IDENTITY_INVALID"), ("gate", "GATE_FAILED"),
    ("unit", "UNIT_MISMATCH"), ("nan", "NUMERIC_INVALID"),
    ("huge_integer", "NUMERIC_INVALID"),
])
def test_statistics_rejects_unpaired_or_invalid_observations(tmp_path, mutation, reason):
    inputs = _inputs(tmp_path)
    docs = inputs["observations"]
    if mutation == "duplicate":
        docs.append(copy.deepcopy(docs[0]))
    elif mutation == "missing":
        docs.pop()
    elif mutation == "identity":
        docs[0]["method_id"] = "method-ir-" + "0" * 24
    elif mutation == "gate":
        docs[0]["validity_gates"]["data_bound"] = False
    elif mutation == "unit":
        docs[0]["unit_metrics"][0]["unit_id"] = "other"
    elif mutation == "nan":
        docs[0]["unit_metrics"][0]["metrics"]["identity_error"] = float("nan")
    else:
        docs[0]["unit_metrics"][0]["metrics"]["identity_error"] = 10**400
    with pytest.raises(ValueError, match=reason):
        paired_statistics(**inputs)
