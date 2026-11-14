"""Portable adapter resources retain the installed scientific schema boundary."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wmloop.constitution import ConstitutionalFreezeError, verify_constitutional_freeze
from wmloop.control.adapter_profiles import (
    AdapterProfileError, compile_adapter_execution, resolve_adapter_asset_bindings,
)
from wmloop.control.research_request import compile_research_plan, plan_to_campaign_payload, ResearchRequestError
from wmloop.execute.model_run_budget import estimate_model_run_gpu_hours
from wmloop.storage import canonical_bytes, sha256_bytes


SCHEMA_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FIELDS = {
    "constitution_freeze": "ADAPTER_CONSTITUTION_INVALID",
    "evaluator_contract": "ADAPTER_EVALUATOR_INVALID",
    "probe_contract": "ADAPTER_PROBE_INVALID",
    "candidate_catalog": "ADAPTER_CANDIDATE_CATALOG_INVALID",
}


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value, ensure_ascii=False))
    return path


def _resources(root: Path) -> dict[str, str]:
    payload = b'{"frozen":"scientific contract"}\n'
    frozen = root / "resources" / "frozen.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(payload)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "wmloop-constitutional-freeze",
        "constitution_id": "portable-fixture",
        "goal_id": "quality",
        "verdict_probe_ids": ["fixture"],
        "entries": [{
            "component": "fixture",
            "path": str(frozen.relative_to(root)),
            "sha256": sha256_bytes(payload),
            "size": len(payload),
        }],
    }
    manifest["constitution_sha256"] = sha256_bytes(canonical_bytes(manifest, ensure_ascii=False))
    files = {"constitution_freeze": _write_json(root / "resources" / "constitution.json", manifest)}
    for field in ("evaluator_contract", "probe_contract", "candidate_catalog"):
        files[field] = _write_json(root / "resources" / f"{field}.json", {"fixture": field})
    return {field: str(path.relative_to(root)) for field, path in files.items()}


@dataclass
class ProfileProject:
    installation: Path
    user: Path
    model: Path
    source: Path
    data: Path
    profile_path: Path
    profile: dict[str, Any]

    def compile(self, *, profile_path: Path | None = None, installed: bool = False):
        if profile_path is None:
            profile_path = self.profile_path
            _write_json(profile_path, self.profile)
        return compile_adapter_execution(
            campaign_id="portable-user",
            model=self.model,
            source_root=self.source,
            data=self.data,
            goal="improve quality",
            budget="1 gpu-hour",
            campaign_root=self.user / "state" / "campaigns" / "portable-user",
            adapter_profile_path=profile_path,
            runtime_python=Path(sys.executable),
            project_root=None if installed else self.installation,
        )


@pytest.fixture
def profile_project(tmp_path: Path) -> ProfileProject:
    installation, user = tmp_path / "installation", tmp_path / "user-project"
    for name in ("adapter_profile", "constitutional_manifest"):
        destination = installation / "configs" / "schemas" / f"{name}.schema.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCHEMA_ROOT / "configs" / "schemas" / destination.name, destination)
    model, source, data = (user / name for name in ("model", "source", "data"))
    for directory in (model, source, data):
        directory.mkdir(parents=True)
    (model / "checkpoint.bin").write_bytes(b"CPU fixture only")
    (source / "evaluate.py").write_text("# interface marker\n", encoding="utf-8")
    profile: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-adapter-profile",
        "profile_id": "portable-fixture",
        "aliases": ["portable"],
        "model_family": "fixture",
        "capability_level": "L1",
        "execution_kind": "pipeline",
        "repo_markers": ["evaluate.py"],
        "goal_keywords": ["quality"],
        **_resources(user),
        "runtime_candidates": ["{env:VERDIWM_RUNTIME_PYTHON}"],
        "asset_bindings": [{"parameter": "--checkpoint", "candidates": ["{model}/checkpoint.bin"]}],
        "runner": {
            "protocol_version": 1,
            "adapter_id": "portable-fixture",
            "requires_training": True,
            "train": ["{python}", "evaluate.py"],
            "evaluate": ["{python}", "evaluate.py"],
            "checkpoint_receipt": "receipts/checkpoint.json",
            "evaluation_receipt": "receipts/evaluation.json",
        },
        "probe_imports": False,
    }
    return ProfileProject(installation, user, model, source, data, user / "profile.json", profile)


@pytest.mark.parametrize("absolute", [False, True])
def test_external_profile_resources_compile_from_unrelated_cwd(
    profile_project: ProfileProject, tmp_path: Path, monkeypatch, absolute: bool
):
    project = profile_project
    expected = {field: project.user / str(project.profile[field]) for field in RESOURCE_FIELDS}
    if absolute:
        project.profile.update({field: str(path) for field, path in expected.items()})
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    resolved = project.compile()

    assert resolved.constitution_freeze == str(expected["constitution_freeze"])
    for field in set(RESOURCE_FIELDS) - {"constitution_freeze"}:
        assert resolved.execution[field] == str(expected[field])
    assert resolved.execution["repo_root"] == str(project.source)
    assert resolved.execution["asset_bindings"]["--checkpoint"] == str(project.model / "checkpoint.bin")
    assert not (project.user / "configs").exists()


def test_external_profile_uses_installed_schemas_by_default(profile_project: ProfileProject):
    assert profile_project.compile(installed=True).profile_id == "portable-fixture"


def test_protocol_v3_profile_uses_model_run_manifest_placeholder(
    profile_project: ProfileProject,
):
    project = profile_project
    _set_protocol_v3_diagnostic_profile(project)

    execution = project.compile().execution

    assert execution["model_run_adapter"]["runner"]["protocol_version"] == 3
    assert execution["agibot_support_loader"] == project.profile[
        "agibot_engineering_diagnostic"
    ]["support_loader"]
    assert execution["agibot_validation_sample_count"] == 128
    assert execution["stage_timeouts_seconds"] == {
        "train": 2400,
        "engineering_diagnostic": 1200,
    }
    assert estimate_model_run_gpu_hours(execution["stage_timeouts_seconds"], 1) == 1.0

    project.profile["runner"]["train"][-1] = "{model_run_manifest}"
    with pytest.raises(AdapterProfileError, match="^ADAPTER_PROFILE_INVALID:"):
        project.compile()


def _set_protocol_v3_diagnostic_profile(project: ProfileProject) -> None:
    project.profile["runner"] = {
        "protocol_version": 3,
        "adapter_id": "portable-fixture-v3",
        "requires_training": True,
        "train": [
            "{source_root}/scripts/training/train_world_model.py",
            "--config",
            "{agibot_native_config}",
            "--init_from",
            "{asset:--init_from}",
        ],
        "evaluate": [
            "evaluate.py", "{model_run_manifest}", "{trained_checkpoint}"
        ],
        "checkpoint_receipt": "receipts/checkpoint.json",
        "evaluation_receipt": "receipts/evaluation.json",
    }
    project.profile["asset_bindings"] = [
        {"parameter": "--init_from", "kind": "directory", "candidates": ["{model}"]}
    ]
    project.profile["agibot_engineering_diagnostic"] = {
        "support_loader": {
            "num_history": 5,
            "num_frames": 3,
            "sample_stride_video_frames": 8,
            "history_selector": "sparse",
            "history_offsets": [],
            "history_dilations": [1, 2],
            "history_collapse_prob": 0.15,
            "history_eval_dilation": 2,
            "generated_history_enabled": False,
            "condition_view": "arm_slot_eef_pose",
            "body_conditioning": True,
        },
        "validation_sample_count": 128,
        "stage_timeouts_seconds": {"train": 2400, "engineering_diagnostic": 1200},
    }


@pytest.mark.parametrize("invalid", ["loader", "timeout"])
def test_protocol_v3_profile_diagnostic_settings_are_schema_checked(
    profile_project: ProfileProject, invalid: str
):
    project = profile_project
    _set_protocol_v3_diagnostic_profile(project)
    if invalid == "loader":
        project.profile["agibot_engineering_diagnostic"]["support_loader"][
            "condition_view"
        ] = "other"
    else:
        project.profile["agibot_engineering_diagnostic"]["stage_timeouts_seconds"][
            "train"
        ] = 604801
    with pytest.raises(AdapterProfileError, match="^ADAPTER_PROFILE_INVALID:"):
        project.compile()


def test_builtin_agibot_profile_is_portable_and_compiles(
    profile_project: ProfileProject, monkeypatch
):
    project = profile_project
    profile_path = SCHEMA_ROOT / "configs/adapters/wan22_agibot_engineering_diagnostic_v1.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for marker in profile["repo_markers"]:
        marker_path = project.source / marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("# native interface marker\n", encoding="utf-8")
    base_model = project.user / "wan-diffusers-base"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        json.dumps({"model_type": "ti2v"}), encoding="utf-8"
    )
    (base_model / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"fixture.weight": "diffusion-part.safetensors"}}),
        encoding="utf-8",
    )
    (base_model / "diffusion-part.safetensors").write_bytes(b"CPU fixture base weights")
    (base_model / "Wan2.2_VAE.pth").write_bytes(b"CPU fixture VAE")
    monkeypatch.setenv("VERDIWM_AGIBOT_BASE_MODEL", str(base_model))

    resolved = compile_adapter_execution(
        campaign_id="agibot-portable-user",
        model=project.model,
        source_root=project.source,
        data=project.data,
        goal="Agibot engineering diagnostic",
        budget="1 gpu-hour",
        campaign_root=project.user / "state" / "campaigns" / "agibot-portable-user",
        adapter="auto",
        adapter_profile_path=profile_path,
        runtime_python=Path(sys.executable),
        project_root=SCHEMA_ROOT,
    )

    assert resolved.profile_id == "wan22-agibot-engineering-diagnostic-v1"
    assert resolved.execution["asset_bindings"]["--base-model"] == str(base_model)
    assert resolved.execution["asset_bindings"]["--init_from"] == str(project.model)
    assert resolved.execution["asset_bindings"]["--vae-checkpoint"] == str(
        base_model / "Wan2.2_VAE.pth"
    )
    assert resolved.execution["stage_timeouts_seconds"] == {
        "train": 2400,
        "engineering_diagnostic": 1200,
    }
    assert all(
        not Path(value).is_absolute()
        for value in profile["runtime_candidates"]
        + [candidate for binding in profile["asset_bindings"] for candidate in binding["candidates"]]
    )


def test_in_repo_profile_retains_installation_relative_resources(profile_project: ProfileProject):
    project = profile_project
    project.profile.update(_resources(project.installation))
    path = _write_json(project.installation / "configs" / "adapters" / "fixture.json", project.profile)

    resolved = project.compile(profile_path=path)

    assert resolved.constitution_freeze == str(project.installation / "resources" / "constitution.json")
    assert resolved.execution["evaluator_contract"] == str(project.installation / "resources" / "evaluator_contract.json")


def test_generated_external_overlay_retains_installed_resource_bindings(profile_project: ProfileProject):
    project = profile_project
    project.profile.update(_resources(project.installation))
    shutil.rmtree(project.user / "resources")

    resolved = project.compile()

    assert resolved.constitution_freeze == str(project.installation / "resources" / "constitution.json")
    for field in set(RESOURCE_FIELDS) - {"constitution_freeze"}:
        assert resolved.execution[field] == str(project.installation / str(project.profile[field]))


def test_external_overlay_accepts_absolute_installed_bindings(profile_project: ProfileProject):
    project = profile_project
    installed = _resources(project.installation)
    project.profile.update({field: str(project.installation / value) for field, value in installed.items()})

    resolved = project.compile()

    assert resolved.constitution_freeze == str(project.installation / installed["constitution_freeze"])
    assert resolved.execution["probe_contract"] == str(project.installation / installed["probe_contract"])


@pytest.mark.parametrize("field,code", RESOURCE_FIELDS.items())
def test_local_directory_does_not_fall_back_to_installed_file(
    profile_project: ProfileProject, field: str, code: str
):
    project = profile_project
    _resources(project.installation)
    local = project.user / str(project.profile[field])
    local.unlink()
    local.mkdir()

    with pytest.raises(AdapterProfileError, match=f"^{code}$"):
        project.compile()


def test_invalid_local_freeze_does_not_fall_back_to_installed_copy(profile_project: ProfileProject):
    project = profile_project
    _resources(project.installation)
    (project.user / "resources" / "frozen.json").write_bytes(b"changed scientific contract")

    with pytest.raises(AdapterProfileError, match="^ADAPTER_CONSTITUTION_INVALID$"):
        project.compile()


@pytest.mark.parametrize("field,code", RESOURCE_FIELDS.items())
@pytest.mark.parametrize("outside", ["missing", "relative_escape", "absolute_escape"])
def test_resource_paths_are_confined(
    profile_project: ProfileProject, tmp_path: Path, field: str, code: str, outside: str
):
    project = profile_project
    external = _write_json(tmp_path / "outside.json", {})
    project.profile[field] = {
        "missing": "resources/missing.json",
        "relative_escape": "../outside.json",
        "absolute_escape": str(external),
    }[outside]

    with pytest.raises(AdapterProfileError, match=f"^{code}$"):
        project.compile()


@pytest.mark.parametrize("field,code", RESOURCE_FIELDS.items())
@pytest.mark.parametrize("parent_link", [False, True])
def test_resource_symlinks_are_rejected_before_resolve(
    profile_project: ProfileProject, field: str, code: str, parent_link: bool
):
    project = profile_project
    original = project.user / str(project.profile[field])
    if parent_link:
        link = project.user / "linked-resources"
        link.symlink_to(original.parent, target_is_directory=True)
        project.profile[field] = str(link / original.name)
    else:
        link = original.with_name("linked-" + original.name)
        link.symlink_to(original)
        project.profile[field] = str(link)

    with pytest.raises(AdapterProfileError, match=f"^{code}$"):
        project.compile()


@pytest.mark.parametrize("parent_link", [False, True])
def test_profile_symlinks_are_rejected_before_resolve(profile_project: ProfileProject, parent_link: bool):
    project = profile_project
    original = _write_json(project.profile_path, project.profile)
    if parent_link:
        link = project.user.parent / "linked-project"
        link.symlink_to(project.user, target_is_directory=True)
        profile_path = link / original.name
    else:
        profile_path = project.user / "linked-profile.json"
        profile_path.symlink_to(original)

    with pytest.raises(AdapterProfileError, match="^ADAPTER_PROFILE_INVALID:"):
        project.compile(profile_path=profile_path)


@pytest.mark.parametrize("parent_link", [False, True])
def test_freeze_entry_symlinks_are_rejected_before_resolve(profile_project: ProfileProject, parent_link: bool):
    project = profile_project
    resources = project.user / "resources"
    if parent_link:
        moved = project.user / "original-resources"
        resources.rename(moved)
        resources.symlink_to(moved, target_is_directory=True)
        # Keep the manifest itself direct so this exercises the entry verifier.
        manifest = project.user / "constitution.json"
        shutil.copyfile(moved / "constitution.json", manifest)
        project.profile["constitution_freeze"] = manifest.name
    else:
        frozen = resources / "frozen.json"
        moved = resources / "original-frozen.json"
        frozen.rename(moved)
        frozen.symlink_to(moved)

    with pytest.raises(AdapterProfileError, match="^ADAPTER_CONSTITUTION_INVALID$"):
        project.compile()


def test_schema_root_separates_schema_lookup_from_frozen_resources(profile_project: ProfileProject):
    project = profile_project
    manifest = json.loads((project.user / str(project.profile["constitution_freeze"])).read_text())

    verify_constitutional_freeze(manifest, root=project.user, schema_root=project.installation)
    with pytest.raises(ConstitutionalFreezeError, match="^CONSTITUTION_MANIFEST_INVALID:"):
        verify_constitutional_freeze(manifest, root=project.user)

    fields = _resources(project.installation)
    installed = json.loads((project.installation / fields["constitution_freeze"]).read_text())
    verify_constitutional_freeze(installed, root=project.installation)


def _blocked_support() -> dict[str, str]:
    return {
        "state": "requires_interface_extension",
        "reason": "The trainer does not bind its checkpoint output.",
        "action": "Bind and validate the trainer output before enabling execution.",
        "documentation": "docs/adapter.md",
    }


def test_known_unsupported_profile_blocks_execution_but_allows_read_only_assets(profile_project):
    project = profile_project
    project.profile["execution_support"] = _blocked_support()
    # Known support must be checked before runtime/resource discovery can make
    # the profile look usable or obscure its actionable interface boundary.
    project.profile["constitution_freeze"] = "not-yet-implemented.json"
    with pytest.raises(AdapterProfileError, match="^ADAPTER_EXECUTION_REQUIRES_INTERFACE_EXTENSION:") as caught:
        project.compile()
    assert caught.value.execution_support == _blocked_support()
    assert caught.value.repairable is False
    assert _blocked_support()["action"] in str(caught.value)
    assets = resolve_adapter_asset_bindings(
        model=project.model, source_root=project.source, data=project.data,
        adapter_profile_path=project.profile_path, project_root=project.installation,
    )
    assert assets == {"--checkpoint": str(project.model / "checkpoint.bin")}


@pytest.mark.parametrize("support", [
    {},
    {**_blocked_support(), "state": "ready"},
    {**_blocked_support(), "reason": True},
    {**_blocked_support(), "documentation": ""},
    {**_blocked_support(), "skip_validation": True},
])
def test_execution_support_metadata_is_schema_checked(profile_project, support):
    profile_project.profile["execution_support"] = support
    with pytest.raises(AdapterProfileError, match="^ADAPTER_PROFILE_INVALID:"):
        profile_project.compile()


def test_builtin_training_profile_cannot_be_planned_as_ready(profile_project):
    project = profile_project
    builtin = SCHEMA_ROOT / "configs/adapters/coachworld_wan22_droid_v1.json"
    for marker in json.loads(builtin.read_text())["repo_markers"]:
        marker_path = project.source / marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("# interface marker only\n", encoding="utf-8")
    plan = compile_research_plan(
        project_root=project.user, model=project.model, source=project.source,
        data=project.data, goal="improve video prediction", adapter_profile=builtin,
        evaluator_contract=project.user / "resources/evaluator_contract.json",
        runtime_python=Path(sys.executable),
    )
    assert plan["state"] == "blocked"
    blocker = next(item for item in plan["blockers"] if item["code"] == "ADAPTER_EXECUTION_REQUIRES_INTERFACE_EXTENSION")
    assert "requires_interface_extension" in blocker["message"]
    assert "arm-slot" in blocker["action"]
    assert "docs/WAN22_DROID_ACWM_ONBOARDING.md" in blocker["action"]
    assert plan["input_bindings"] == {}
    assert plan["side_effects"]["gpu_execution_started"] is False
    with pytest.raises(ResearchRequestError, match="^RESEARCH_PLAN_BLOCKED$"):
        plan_to_campaign_payload(plan)
