import json
import subprocess
from pathlib import Path

from wmloop.execute.literature_materialization import (
    run_literature_method_materialization,
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


def test_literature_work_order_is_materialized_into_admitted_catalog(tmp_path: Path):
    staging, evaluator = _inputs(tmp_path)
    manifest = run_literature_method_materialization(
        method_staging_manifest=staging,
        output_root=tmp_path / "materialization",
        source_root=_git_source(tmp_path),
        project_root=Path(__file__).resolve().parents[1],
        evaluator_contract=evaluator,
    )
    assert manifest["ready_count"] == 1
    catalog = json.loads(Path(str(manifest["candidate_catalog_path"])).read_text())
    assert catalog["candidates"][0]["candidate_id"] == "method-demo"
    receipt = Path(catalog["candidates"][0]["materialization_receipt_path"])
    assert json.loads(receipt.read_text())["state"] == "ready_for_candidate_compilation"
