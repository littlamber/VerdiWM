"""Readable and bounded public campaign status output."""

from __future__ import annotations

import json

from wmloop.cli import _parser, main


class _Store:
    def __init__(self, records: list[dict[str, object]]):
        self.records = records

    def get(self, campaign_id: str) -> dict[str, object]:
        return next(row for row in self.records if row["campaign_id"] == campaign_id)

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        rows = [row for row in self.records if status is None or row.get("status") == status]
        return rows[:limit]


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "campaign_id": "wan-rounds",
        "status": "failed",
        "goal": "Improve PSNR by at least 2 dB",
        "model": "/private/models/wan5b",
        "dataset": "/private/datasets/agibot",
        "execution": {"output_root": "/private/output/wan-rounds"},
        "execution_result": {
            "rounds_completed": 2,
            "stop_reason": "MAX_ITERATIONS_REACHED",
            "stages": [{"stage": "train", "returncode": 1}],
        },
        "execution_error": {
            "stage": "train",
            "message": "MODEL_RUN_TRAIN_FAILED: see private traceback",
            "receipt_path": "/private/output/wan-rounds/execution-receipt.json",
        },
        "revision": {"manifest": "raw private manifest must not be printed"},
    }
    record.update(overrides)
    return record


def test_status_text_shows_actionable_summary_without_raw_manifest_or_input_paths(
    monkeypatch, capsys
) -> None:
    from wmloop.control.commands import lifecycle

    monkeypatch.setattr(lifecycle, "_store", lambda _: _Store([_record()]))

    assert main(["status", "wan-rounds", "--format", "text"]) == 0

    output = capsys.readouterr().out
    assert "Campaign: wan-rounds" in output
    assert "状态: failed" in output
    assert "目标: Improve PSNR by at least 2 dB" in output
    assert "轮数: 2" in output and "MAX_ITERATIONS_REACHED" in output
    assert "失败阶段: train" in output and "MODEL_RUN_TRAIN_FAILED" in output
    assert "/private/output/wan-rounds/execution-receipt.json" in output
    assert "/private/models/wan5b" not in output
    assert "/private/datasets/agibot" not in output
    assert "raw private manifest" not in output
    assert "private traceback" not in output


def test_status_text_list_caps_campaign_count_and_long_fields(monkeypatch, capsys) -> None:
    from wmloop.control.commands import lifecycle

    records = [
        _record(
            campaign_id=f"campaign-{index}",
            status="completed",
            goal="G" * 10_000,
            execution_result={"rounds_completed": 2, "stop_reason": "S" * 10_000},
            execution_error=None,
            execution={},
        )
        for index in range(35)
    ]
    monkeypatch.setattr(lifecycle, "_store", lambda _: _Store(records))

    assert main(["status", "--format", "text", "--limit", "1000"]) == 0

    output = capsys.readouterr().out
    assert "Campaign: campaign-0" in output
    assert "Campaign: campaign-19" in output
    assert "Campaign: campaign-20" not in output
    assert "另有 15 个 campaign 未显示" in output
    assert len(output) <= lifecycle._STATUS_TEXT_MAX_CHARS
    assert "G" * 141 not in output
    assert "S" * 73 not in output


def test_status_text_hides_paths_embedded_in_goal_and_model_or_data_receipt(
    monkeypatch, capsys
) -> None:
    from wmloop.control.commands import lifecycle

    model_root = "/private/models/wjy/Wan5B"
    dataset_root = "/private/data/agibot"
    receipt = f"{model_root}/outputs/execution-receipt.json"
    record = _record(
        goal=f"Improve PSNR using checkpoint {model_root} and data {dataset_root}",
        model=model_root,
        dataset=dataset_root,
        execution_error={"stage": "restore", "code": "CHECKPOINT_RESTORE_FAILED", "receipt_path": receipt},
    )
    monkeypatch.setattr(lifecycle, "_store", lambda _: _Store([record]))

    assert main(["status", "wan-rounds", "--format", "text"]) == 0

    output = capsys.readouterr().out
    assert model_root not in output
    assert dataset_root not in output
    assert receipt not in output
    assert "[模型/数据目录内的回执，路径已隐藏]" in output
    assert "Improve PSNR using checkpoint [路径已隐藏] and data [路径已隐藏]" in output


def test_status_format_defaults_to_compatible_json_and_accepts_explicit_json() -> None:
    parser = _parser()

    default = parser.parse_args(["status"])
    explicit = parser.parse_args(["status", "--format", "json"])

    assert default.format == explicit.format == "json"


def test_run_and_cancel_default_to_text_but_explicit_json_reaches_handler(monkeypatch) -> None:
    seen: list[tuple[str, str | None]] = []

    def fake_run(args, **_dependencies):
        seen.append(("run", args.format))
        return 0

    def fake_cancel(args):
        seen.append(("cancel", args.format))
        return 0

    monkeypatch.setattr("wmloop.cli._run_command", fake_run)
    monkeypatch.setattr("wmloop.cli._cancel", fake_cancel)

    assert main(["run"]) == 0
    assert main(["run", "--format", "json"]) == 0
    assert main(["cancel", "demo"]) == 0
    assert main(["cancel", "demo", "--format", "json"]) == 0
    assert seen == [
        ("run", "text"),
        ("run", "json"),
        ("cancel", "text"),
        ("cancel", "json"),
    ]


def test_status_text_describes_queued_and_cancelled_records_without_fake_stop_reason(
    monkeypatch, capsys
) -> None:
    from wmloop.control.commands import lifecycle

    records = [
        _record(campaign_id="queued", status="queued", execution_result={}),
        _record(campaign_id="cancelled", status="cancelled", execution_result={}),
    ]
    monkeypatch.setattr(lifecycle, "_store", lambda _: _Store(records))

    assert main(["status", "queued", "--format", "text"]) == 0
    queued_output = capsys.readouterr().out
    assert "轮数: 尚无结果 · 停止原因: 尚未开始" in queued_output
    assert "尚未停止" not in queued_output

    assert main(["status", "cancelled", "--format", "text"]) == 0
    cancelled_output = capsys.readouterr().out
    assert "轮数: 尚无结果 · 停止原因: 用户取消" in cancelled_output
    assert "尚未停止" not in cancelled_output


def test_run_text_summary_is_bounded_and_json_remains_available(
    monkeypatch, capsys, tmp_path
) -> None:
    from wmloop.control.commands import run as run_command

    class Store:
        def create(self, payload):
            return {"campaign_id": "run-summary"}

        def confirm(self, campaign_id):
            return {
                "campaign_id": campaign_id,
                "status": "queued",
                "goal": "Improve PSNR",
                "execution_result": {
                    "verification_samples": ["private-sample-id"] * 5000,
                },
            }

        def get(self, campaign_id):
            return self.confirm(campaign_id) | {"status": "completed"}

    args = _parser().parse_args(["run", "--queue-only"])
    args.format = "text"
    resolve_inputs = lambda _: {
        "state_root": tmp_path / "state",
        "goal": "Improve PSNR",
        "model": str(tmp_path / "private-model"),
        "data": str(tmp_path / "private-data"),
        "budget": None,
        "adapter": "auto",
        "mode": None,
        "source": None,
        "target_metrics": None,
        "campaign_id": None,
        "adapter_profile": None,
        "runtime_python": None,
    }

    assert run_command.run(
        args,
        resolve_inputs=resolve_inputs,
        store_factory=lambda _: Store(),
        dispatch=lambda *args, **kwargs: {"state": "completed"},
        configured_llm_adapter=lambda _: {},
        load_training_scale_plan=lambda *args, **kwargs: {},
        is_repairable_error=lambda _: False,
    ) == 0
    output = capsys.readouterr().out
    assert "Campaign: run-summary" in output
    assert "private-sample-id" not in output
    assert str(tmp_path / "private-model") not in output
    assert len(output) < 2000

def test_cancel_text_summary_hides_internal_record_and_keeps_json_contract(
    monkeypatch, capsys, tmp_path
) -> None:
    from wmloop.control.commands import lifecycle

    class Store:
        def cancel(self, campaign_id):
            return {
                "campaign_id": campaign_id,
                "status": "cancelled",
                "goal": "Improve PSNR",
                "model": str(tmp_path / "private-model"),
                "execution_result": {
                    "verification_samples": ["private-sample-id"] * 5000,
                },
            }

    monkeypatch.setattr(lifecycle, "_store", lambda _: Store())
    args = _parser().parse_args(["cancel", "cancel-me", "--format", "text"])
    assert lifecycle.cancel(args) == 0
    output = capsys.readouterr().out
    assert "Campaign: cancel-me" in output
    assert "停止原因: 用户取消" in output
    assert "private-sample-id" not in output
    assert str(tmp_path / "private-model") not in output
    assert len(output) < 2000

    args.format = "json"
    assert lifecycle.cancel(args) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["campaign_id"] == "cancel-me"
    assert machine["execution_result"]["verification_samples"] == ["private-sample-id"] * 5000
