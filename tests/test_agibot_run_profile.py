from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from wmloop.control.campaign_api import CampaignAPIError
from wmloop.control.commands import run as run_command


ROOT = Path(__file__).resolve().parents[1]


class _Store:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def create(self, payload: dict[str, Any]) -> dict[str, str]:
        self.payload = payload
        return {"campaign_id": "agibot-test"}

    def confirm(self, _campaign_id: str) -> dict[str, str]:
        return {"campaign_id": "agibot-test", "status": "queued"}


def _args(tmp_path: Path, *, profile: Path | None = None) -> Namespace:
    native_config = tmp_path / "training.yaml"
    native_config.write_text("world_model: {}\nwm_training: {}\n", encoding="utf-8")
    return Namespace(
        compiled_manifest=None,
        engineering_manifest=None,
        engineering_repo_root=None,
        schema_root=ROOT,
        rounds=None,
        stop_on_success=None,
        literature_query=None,
        model_irg=None,
        irg_protected_metric=[],
        cpbe_request=None,
        cpbe_history=None,
        campaign_id=None,
        adapter_profile=profile,
        adapter="auto",
        runtime_python=None,
        asset=[],
        training_scale_plan=None,
        world_size=None,
        training_config=native_config,
        state_root=tmp_path / "state",
        queue_only=True,
        max_parallel=1,
        auto_repair_adapter=False,
        repair_base_profile=None,
        llm_config=None,
        repair_attempts=1,
    )


def _run(
    tmp_path: Path,
    args: Namespace,
    monkeypatch,
    *,
    plan_builder=None,
) -> _Store:
    model, data, source = (tmp_path / name for name in ("model", "data", "source"))
    for path in (model, data, source):
        path.mkdir(exist_ok=True)
    store = _Store()
    monkeypatch.setattr(
        run_command,
        "build_training_scale_plan",
        plan_builder or (lambda **_kwargs: {"stage": "engineering_diagnostic"}),
    )
    result = run_command.run(
        args,
        resolve_inputs=lambda _args: {
            "state_root": tmp_path / "state",
            "goal": "Agibot engineering diagnostic",
            "model": str(model),
            "data": str(data),
            "source": str(source),
            "budget": f"{getattr(args, 'world_size', None) or 1}gpu-hour",
            "adapter": "legacy-project-adapter",
            "mode": None,
            "target_metrics": None,
            "campaign_id": None,
            "adapter_profile": args.adapter_profile,
            "runtime_python": None,
        },
        store_factory=lambda _root: store,
        dispatch=lambda *_args, **_kwargs: {},
        configured_llm_adapter=lambda _path: {},
        load_training_scale_plan=lambda *_args, **_kwargs: {},
        is_repairable_error=lambda _exc: False,
    )
    assert result == 0
    return store


def test_training_config_selects_builtin_agibot_profile_and_ignores_legacy_auto_adapter(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    store = _run(tmp_path, _args(tmp_path), monkeypatch)

    assert store.payload is not None
    assert store.payload["adapter_profile_path"] == str(
        ROOT / "configs/adapters/wan22_agibot_engineering_diagnostic_v1.json"
    )
    assert store.payload["adapter"] == "auto"
    assert store.payload["agibot_native_config"] == str(
        (tmp_path / "training.yaml").resolve()
    )
    assert "queued" in capsys.readouterr().out


def test_training_config_preserves_explicit_profile(tmp_path: Path, monkeypatch, capsys) -> None:
    user_profile = tmp_path / "custom-profile.json"
    user_profile.write_text("{}\n", encoding="utf-8")
    store = _run(tmp_path, _args(tmp_path, profile=user_profile), monkeypatch)

    assert store.payload is not None
    assert store.payload["adapter_profile_path"] == str(user_profile)
    assert store.payload["adapter"] == "legacy-project-adapter"
    capsys.readouterr()


def test_training_config_binds_requested_world_size_to_auto_plan(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    args = _args(tmp_path)
    args.world_size = 4
    observed: dict[str, Any] = {}

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "stage": "engineering_diagnostic",
            "parallelism": {"world_size": kwargs["world_size"]},
        }

    store = _run(tmp_path, args, monkeypatch, plan_builder=build_plan)

    assert observed["world_size"] == 4
    assert store.payload is not None
    assert store.payload["training_scale_plan"]["parallelism"]["world_size"] == 4
    capsys.readouterr()


def test_world_size_rejects_explicit_training_scale_plan(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    args.world_size = 2
    args.training_scale_plan = tmp_path / "explicit-plan.json"

    with pytest.raises(
        CampaignAPIError,
        match="RUN_WORLD_SIZE_CONFLICTS_WITH_EXPLICIT_TRAINING_SCALE_PLAN",
    ):
        _run(tmp_path, args, monkeypatch)


def test_world_size_requires_native_training_config(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    args.world_size = 2
    args.training_config = None

    with pytest.raises(
        CampaignAPIError, match="RUN_WORLD_SIZE_REQUIRES_TRAINING_CONFIG"
    ):
        _run(tmp_path, args, monkeypatch)
