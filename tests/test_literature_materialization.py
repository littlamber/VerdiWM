import json
import hashlib

import pytest

from wmloop.contracts import validate_document
from wmloop.control.method_candidate_compiler import _materialization_receipt_blockers
import subprocess
from pathlib import Path

from wmloop.execute.literature_materialization import (
    run_literature_method_materialization,
    LiteratureMaterializationError,
)


def _git_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "seed.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=test", "-c", "user.email=test@invalid", "commit", "-qm", "seed"],
        check=True,
    )
    return source


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    orders = staging / "work-orders"
    orders.mkdir(parents=True)
    method = {
        "artifact_type": "wmloop-literature-method-candidate",
        "candidate_id": "method-demo",
        "source": {"arxiv_id": "2401.1", "title": "Demo", "source_url": "https://arxiv.org/abs/2401.1"},
        "target_failure_signatures": ["long_horizon_drift"],
        "required_hook": "H3",
        "estimated_gpu_hours": 0.1,
    }
    order_path = orders / "method-demo.json"
    order_path.write_text(json.dumps({"artifact_type": "wmloop-primitive-materialization-work-order", "literature_method": method}), encoding="utf-8")
    (staging / "manifest.json").write_text(json.dumps({"artifact_type": "wmloop-literature-method-staging-manifest", "work_order_paths": {"method-demo": str(order_path)}}), encoding="utf-8")
    prototype = {
        "candidate_id": "base",
        "candidate_kind": "literature_method",
        "parameters": {},
        "hypothesis": "base hypothesis is sufficiently specific",
        "selection_reason": "base selection reason is sufficiently specific",
        "falsification_criterion": "base falsification criterion is sufficiently specific",
        "stages": [],
    }
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"candidates": [prototype]}), encoding="utf-8")
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(json.dumps({"scheduler_template": str(template)}), encoding="utf-8")
    return staging / "manifest.json", evaluator


def test_literature_work_order_smoke_does_not_enter_science_catalog(tmp_path: Path):
    staging, evaluator = _inputs(tmp_path)
    manifest = run_literature_method_materialization(
        method_staging_manifest=staging,
        output_root=tmp_path / "materialization",
        source_root=_git_source(tmp_path),
        project_root=Path(__file__).resolve().parents[1],
        evaluator_contract=evaluator,
        interface_smoke_only=True,
    )
    assert manifest["ready_count"] == 0
    assert manifest["blocked_count"] == 1
    assert manifest["interface_smoke_only"] is True
    assert manifest["records"][0]["state"] == "blocked"
    catalog = json.loads(Path(manifest["candidate_catalog_path"]).read_text())
    validate_document("method_candidate_catalog", catalog)
    assert catalog["candidates"] == []
    receipt_path = tmp_path / "materialization" / "candidates" / "method-demo" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert all(row["passed"] for row in receipt["command_receipts"])
    assert receipt["blockers"] == [{"code": "SURROGATE_IMPLEMENTATION_NOT_FORMAL_CANDIDATE"}]
    assert receipt["side_effects"]["candidate_compilation_authority"] is False
    # Reopening an explicit smoke as a formal method must never upgrade it.
    with pytest.raises(LiteratureMaterializationError, match="MODE_MISMATCH"):
        run_literature_method_materialization(
            method_staging_manifest=staging, output_root=tmp_path / "materialization",
            source_root=tmp_path / "source", project_root=Path(__file__).resolve().parents[1],
            evaluator_contract=evaluator,
        )


def test_literature_work_order_is_blocked_without_real_implementation(tmp_path: Path):
    staging, evaluator = _inputs(tmp_path)
    manifest = run_literature_method_materialization(
        method_staging_manifest=staging,
        output_root=tmp_path / "materialization-blocked",
        source_root=_git_source(tmp_path),
        project_root=Path(__file__).resolve().parents[1],
        evaluator_contract=evaluator,
    )
    assert manifest["state"] == "blocked"
    assert manifest["ready_count"] == 0
    assert manifest["blocked_count"] == 1
    assert manifest["records"][0]["error"] == "REAL_METHOD_IMPLEMENTATION_REQUIRED"


@pytest.mark.parametrize("policy,surrogate,expected", [
    (None, False, "POLICY_REQUIRES_REVALIDATION"),
    ("real_method_required_v2", True, "SURROGATE_IMPLEMENTATION"),
    ("real_method_required_v2", False, None),
])
def test_compiler_rejects_legacy_and_surrogate_receipts(tmp_path, policy, surrogate, expected):
    receipt = {"artifact_type":"verdiwm-automatic-materialization-receipt", "state":"ready_for_candidate_compilation", "candidate_id":"demo", "side_effects":{"candidate_compilation_authority":True}, "surrogate":surrogate}
    if policy is not None:
        receipt["admission_policy"] = policy
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    blockers = _materialization_receipt_blockers({"candidate_id":"demo", "materialization_receipt_path":str(path), "materialization_receipt_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    if expected is None:
        assert blockers == []
    else:
        assert expected in blockers[0]["code"]


def test_materialization_cache_is_bound_to_work_order_contents(tmp_path):
    staging, evaluator = _inputs(tmp_path)
    options = dict(method_staging_manifest=staging, output_root=tmp_path / "materialization",
                   source_root=_git_source(tmp_path), project_root=Path(__file__).resolve().parents[1],
                   evaluator_contract=evaluator)
    first = run_literature_method_materialization(**options)
    assert run_literature_method_materialization(**options) == first
    order = staging.parent / "work-orders" / "method-demo.json"
    value = json.loads(order.read_text())
    value["literature_method"]["target_failure_signatures"] = ["different_failure"]
    order.write_text(json.dumps(value))
    with pytest.raises(LiteratureMaterializationError, match="INPUT_MISMATCH"):
        run_literature_method_materialization(**options)
