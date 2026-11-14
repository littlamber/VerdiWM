"""CPU regression checks for extracted services and portable resource paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wmloop.cli import _parser, _resolve_run_inputs
from wmloop.control.campaign_api import CampaignAPIError
from wmloop.control.commands import run as campaign_command
from wmloop.control.commands.io import repository_root
from wmloop.execute.autonomous_pipeline import AutonomousPipelineOptions
from wmloop.execute.pipeline.retrieval import build_irg_context, build_literature_query
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.training_smoke.io import _canonical_json_bytes, _write_bytes_atomic
from wmloop.experiments.engineering import ExperimentEngineeringError
from wmloop.retrieve.discovery.extraction import (
    EvidenceOnlyMechanismExtractor,
    extractor_identity,
)
from wmloop.retrieve.query_planning import build_literature_query_plan
from wmloop.storage import canonical_json_bytes


class CampaignRecorder:
    def __init__(self, *, repair_required: bool = False) -> None:
        self.payloads = []
        self.events = []
        self.repair_required = repair_required

    def create(self, payload):
        self.payloads.append(dict(payload))
        self.events.append("create")
        if self.repair_required and len(self.payloads) == 1:
            raise CampaignAPIError("fixture missing adapter")
        return {"campaign_id": "fixture"}

    def confirm(self, campaign_id):
        self.events.append("confirm")
        return {"campaign_id": campaign_id, "status": "queued"}

    def get(self, campaign_id):
        self.events.append("get")
        return {"campaign_id": campaign_id, "status": "completed"}


def _args(tmp_path, *extra):
    return _parser().parse_args(
        [
            "run",
            "--goal",
            "改善轨迹",
            "--model",
            str(tmp_path / "model"),
            "--data",
            str(tmp_path / "data"),
            "--state-root",
            str(tmp_path / "state"),
            *extra,
        ]
    )


def _run(args, store, **overrides):
    def dispatch(store, **kwargs):
        store.events.append("dispatch")
        return {"state": "completed"}

    dependencies = {
        "resolve_inputs": _resolve_run_inputs,
        "store_factory": lambda path: store,
        "dispatch": dispatch,
        "configured_llm_adapter": lambda path: {},
        "load_training_scale_plan": lambda *args, **kwargs: {},
        "is_repairable_error": lambda error: True,
    }
    dependencies.update(overrides)
    return campaign_command.run(args, **dependencies)


@pytest.mark.parametrize("queue_only", [True, False])
def test_run_preserves_confirm_dispatch_order(
    tmp_path, monkeypatch, capsys, queue_only
):
    monkeypatch.chdir(tmp_path)
    store = CampaignRecorder()
    args = _args(tmp_path, *(["--queue-only"] if queue_only else []))
    assert _run(args, store) == 0
    assert store.events == (
        ["create", "confirm"]
        if queue_only
        else ["create", "confirm", "dispatch", "get"]
    )
    response = json.loads(capsys.readouterr().out)
    assert (
        response["status"] == "queued"
        if queue_only
        else response["campaign"]["status"] == "completed"
    )
    assert store.payloads[0]["goal"] == "改善轨迹"


def test_run_uses_installed_schema_root_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    observed = []

    def load_plan(path, *, root):
        observed.append(root)
        assert (root / "configs/schemas/training_scale_plan.schema.json").is_file()
        return {"state": "ready"}

    args = _args(
        tmp_path, "--queue-only", "--training-scale-plan", str(tmp_path / "plan.json")
    )
    assert _run(args, CampaignRecorder(), load_training_scale_plan=load_plan) == 0
    assert observed == [repository_root()]


def test_run_auto_repair_keeps_receipt_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    observed = []

    def repair(**kwargs):
        observed.append(kwargs)
        kwargs["output_root"].mkdir(parents=True)
        (kwargs["output_root"] / "manifest.json").write_text('{"state":"ready"}')
        return {
            "state": "ready",
            "adapter_profile_path": "fixture-profile",
            "input_digest": "fixture",
            "assurance_level": "fixture",
        }

    monkeypatch.setattr(campaign_command, "run_adapter_repair", repair)
    store = CampaignRecorder(repair_required=True)
    args = _args(
        tmp_path,
        "--queue-only",
        "--repair-base-profile",
        str(tmp_path / "profile.json"),
    )
    assert _run(args, store) == 0
    assert store.events == ["create", "create", "confirm"]
    assert observed[0]["project_root"] == repository_root()
    binding = store.payloads[1]["adapter_repair"]
    from wmloop.storage import sha256_file

    assert binding["manifest_sha256"] == sha256_file(Path(binding["manifest_path"]))


def test_blocked_engineering_does_not_create_campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        campaign_command,
        "lint_experiment_manifest",
        lambda *args, **kwargs: {"state": "blocked", "blockers": ["fixture"]},
    )
    store = CampaignRecorder()
    with pytest.raises(
        ExperimentEngineeringError, match="EXPERIMENT_ENGINEERING_BLOCKED"
    ):
        _run(
            _args(tmp_path, "--engineering-manifest", str(tmp_path / "manifest.json")),
            store,
        )
    assert store.events == []


def test_lock_environment_is_read_when_instances_are_created(tmp_path, monkeypatch):
    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("VERDIWM_LOCK_ROOT", str(root))
        options = AutonomousPipelineOptions(
            repo_root=tmp_path,
            output_root=tmp_path / "out",
            evaluator_contract=tmp_path / "evaluator",
        )
        assert options.lock_root == root
        assert GpuLeaseManager()._lock_root == root


def test_utf8_receipt_bytes_survive_storage_consolidation(tmp_path):
    payload = {"观察": "轨迹", "gain": 0.5}
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert _canonical_json_bytes(payload) == expected
    assert canonical_json_bytes(payload) != expected
    path = tmp_path / "nested" / "receipt.json"
    _write_bytes_atomic(path, expected)
    assert path.read_bytes() == expected
    assert list(path.parent.iterdir()) == [path]


def test_builtin_extractor_keeps_historical_artifact_identity():
    assert (
        extractor_identity(EvidenceOnlyMechanismExtractor())
        == "wmloop.retrieve.mechanism_discovery.EvidenceOnlyMechanismExtractor"
    )


def test_pipeline_retrieval_retains_cross_domain_lenses_and_explicit_intent():
    context = {
        "irg_guided": {
            "request": {
                "model_family": "video",
                "cross_domain_lenses": ["belief filtering"],
                "failure_signatures": ["history_drift"],
            }
        }
    }
    plan = build_literature_query_plan(
        explicit_query="trajectory consistency",
        retrieval_context=context,
        probe_manifest=None,
    )
    query = build_literature_query(
        explicit_query="trajectory consistency",
        retrieval_context=context,
        probe_manifest=None,
        irg_failure_signatures=(),
        irg_model_family=None,
    )

    # Search the user's goal and cross-domain lens first. The current
    # observation remains available as a follow-up query if that search is
    # empty, rather than narrowing the primary search to a diagnosis.
    assert query == plan["queries"][0]["query"]
    assert 'all:"trajectory" AND all:"consistency"' in query
    assert 'all:"belief" OR all:"filtering"' in query
    assert plan["queries"][1]["view"] == "current_diagnostic"
    assert 'all:"history" AND all:"drift"' in plan["queries"][1]["query"]


def test_pipeline_retrieval_without_observation_does_not_invent_query():
    assert build_irg_context(None, ()) == (None, (), None)
    assert (
        build_literature_query(
            explicit_query=None,
            retrieval_context={"state": "cold_start"},
            probe_manifest=None,
            irg_failure_signatures=(),
            irg_model_family=None,
        )
        is None
    )


def test_malformed_irg_keeps_pipeline_domain_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[]")
    with pytest.raises(AutonomousPipelineError, match="MODEL_IRG_INVALID"):
        build_irg_context(path, ("protected",))


def test_components_import_without_compatibility_facades(tmp_path):
    code = """
import importlib
import sys
pairs = {
    "wmloop.control.commands.run": "wmloop.cli",
    "wmloop.control.onboarding_scan.source": "wmloop.control.onboarding",
    "wmloop.execute.training_smoke.evidence": "wmloop.orchestrator_training_eval_smoke",
    "wmloop.execute.pipeline.state": "wmloop.execute.autonomous_pipeline",
    "wmloop.retrieve.discovery.extraction": "wmloop.retrieve.mechanism_discovery",
    "wmloop.experiments.knowledge.graph": "wmloop.experiments.portable_knowledge_graph",
}
for implementation, facade in pairs.items():
    before = set(sys.modules)
    importlib.import_module(implementation)
    assert facade not in set(sys.modules) - before, (implementation, facade)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", [
    "wmloop.retrieve.index",
    "wmloop.evaluate.future_video_metrics",
    "wmloop.control.onboarding_scan.runtime",
    "wmloop.process_lifecycle",
])
def test_cpu_components_do_not_load_execution_or_torch(tmp_path, module):
    code = f"""
import importlib
import sys
importlib.import_module({module!r})
forbidden = sorted(name for name in sys.modules
                   if name in {{'wmloop.execute', 'torch'}}
                   or name.startswith(('wmloop.execute.', 'torch.')))
assert not forbidden, forbidden
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_process_lifecycle_compatibility_exports_preserve_object_identity():
    from wmloop import process_lifecycle as shared
    from wmloop.execute import process_lifecycle as compatibility

    for name in compatibility.__all__:
        assert getattr(compatibility, name) is getattr(shared, name)


@pytest.mark.parametrize(
    "modules",
    [
        ("wmloop.control.model_portrait", "wmloop.geometry.model_irg"),
        ("wmloop.geometry.model_irg", "wmloop.control.model_portrait"),
        ("wmloop.retrieve.discovery.extraction", "wmloop.retrieve.mechanism_discovery"),
        (
            "wmloop.experiments.knowledge.projection",
            "wmloop.experiments.portable_knowledge_graph",
        ),
    ],
)
def test_public_components_do_not_depend_on_import_order(tmp_path, modules):
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({module!r})" for module in modules
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "package",
    ["wmloop.control", "wmloop.geometry", "wmloop.retrieve", "wmloop.experiments"],
)
def test_lazy_public_exports_remain_accessible_in_fresh_process(tmp_path, package):
    code = f"""
import importlib
package = importlib.import_module({package!r})
for name in package.__all__:
    assert name in dir(package)
    getattr(package, name)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
