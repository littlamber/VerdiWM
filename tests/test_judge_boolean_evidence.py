from dataclasses import replace

import pytest

from wmloop.verify.judge import VerificationEvidence, judge


@pytest.mark.parametrize("field", [
    "readonly_evaluator_verified",
    "accept_split_verified",
    "extended_horizon_verified",
    "diff_audit_passed",
    "evidence_complete",
    "action_following_gate_enabled",
])
@pytest.mark.parametrize("invalid", ["false", 1, None])
def test_formal_verdict_rejects_non_boolean_gate_evidence(field, invalid):
    evidence = VerificationEvidence(
        proposal_id="proposal", readonly_evaluator_verified=True,
        accept_split_verified=True, extended_horizon_verified=True,
        diff_audit_passed=True, evidence_complete=True,
        accept_metric_deltas={"quality": 1.0}, replication_deltas=[1.0, 1.0, 1.0],
        action_following_observed=1.0, action_following_threshold=0.5,
    )
    assert judge(evidence).verdict.value == "ACCEPT"
    with pytest.raises(ValueError, match="VERIFICATION_EVIDENCE_INVALID"):
        judge(replace(evidence, **{field: invalid}))
