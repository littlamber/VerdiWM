"""Text status regression for actionable, redacted model-run failures."""

from __future__ import annotations

import json

from wmloop.cli import main


def test_status_text_extracts_safe_cause_from_agibot_failure_receipt(tmp_path, monkeypatch, capsys):
    """The excerpt is the sanitized failure summary from the real Agibot run."""

    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "execution-receipt.json").write_text(
        json.dumps(
            {
                "artifact_type": "verdiwm-model-run-execution-receipt",
                "stages": [
                    {
                        "stage": "train",
                        "returncode": 1,
                        "stderr_tail": (
                            "Torch Elastic rank 0: RuntimeError: H1 requires exact "
                            "base-model restore except new reference weights: "
                            "missing=['action_encoder.domain_prompt.weight', "
                            "'body_encoder.slot_embedding', 'body_encoder.time_embedding', "
                            "'body_encoder.value_project']"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record = {
        "campaign_id": "agibot-psnr-pipeline-audit",
        "status": "failed",
        "goal": "Improve PSNR by at least 2 dB",
        "execution": {"output_root": str(output_root)},
        "execution_error": {
            "stage": "train",
            "message": "MODEL_RUN_TRAIN_FAILED",
        },
    }

    from wmloop.control.commands import lifecycle

    class Store:
        def get(self, campaign_id: str) -> dict[str, object]:
            assert campaign_id == record["campaign_id"]
            return record

    monkeypatch.setattr(lifecycle, "_store", lambda _: Store())

    assert main(["status", "agibot-psnr-pipeline-audit", "--format", "text"]) == 0

    output = capsys.readouterr().out
    assert "失败阶段: train · 错误码: MODEL_RUN_TRAIN_FAILED" in output
    assert "根因: Torch Elastic rank 0: RuntimeError: H1 requires exact" in output
    assert "body_encoder.slot_embedding" in output
    assert "stderr_tail" not in output
    cause_line = next(line for line in output.splitlines() if "根因:" in line)
    assert str(output_root) not in cause_line
