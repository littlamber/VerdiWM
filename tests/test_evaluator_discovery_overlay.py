from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.evaluator_discovery import EvaluatorDiscoveryError, discover_evaluator_candidates
from wmloop.control.evaluator_overlay import EvaluatorOverlayError, materialize_evaluator_overlay
from wmloop.control.onboarding_scan.evaluator import _load_evaluator_contract
from wmloop.storage import sha256_bytes


def test_discovery_reports_protocol_scope_without_freezing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate_dir = source / "configs" / "evaluation"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "long_eval.json"
    candidate.write_text(
        json.dumps(
            {
                "role": "validation",
                "collection_root": "data/old",
                "protocol": {"fps": 5, "num_video_frames": 300, "closed_loop_chunks": 8},
                "split_contract": {"source_split": "val", "episode_disjoint": True},
                "metrics": ["trajectory_accuracy"],
            }
        ),
        encoding="utf-8",
    )
    report = discover_evaluator_candidates(source)
    assert report["state"] == "candidates_available"
    assert report["selection_requires_confirmation"] is True
    assert report["selection"] is None
    row = report["candidates"][0]
    assert row["horizon_seconds"] == 60.0
    assert row["minute_level_supported"] is True
    assert row["split"] == "val"
    assert row["requires_user_confirmation"] is True
    assert report["selection_hint"]["candidate_id"] == row["candidate_id"]


@pytest.mark.parametrize("missing", [True, False])
def test_discovery_rejects_missing_or_non_directory_source(tmp_path: Path, missing: bool) -> None:
    target = tmp_path / "source"
    if not missing:
        target.write_text("fixture")
    with pytest.raises(EvaluatorDiscoveryError, match="EVALUATOR_SOURCE_DIRECTORY_REQUIRED"):
        discover_evaluator_candidates(target)


@pytest.mark.parametrize("text", ['{"protocol":{"fps":NaN}}', '{"protocol":{"closed_loop_chunks":Infinity}}', '{"protocol":{"fps":1e400}}', 'protocol:\n  fps: .inf\n', 'manifest: &cycle\n  child: *cycle\n', 'manifest: !!set\n  invalid:\n'])
def test_discovery_rejects_nonfinite_and_non_json_yaml_metadata(tmp_path: Path, text: str) -> None:
    source = tmp_path / "source"
    directory = source / "configs" / "evaluation"
    directory.mkdir(parents=True)
    candidate = directory / ("eval.json" if text.startswith("{") else "eval.yaml")
    candidate.write_text(text, encoding="utf-8")
    report = discover_evaluator_candidates(source)
    assert report["state"] == "none"
    assert report["warnings"]
    json.dumps(report, allow_nan=False)


def test_discovery_does_not_claim_support_for_invalid_numeric_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    directory = source / "configs" / "evaluation"
    directory.mkdir(parents=True)
    (directory / "eval.json").write_text(json.dumps({"protocol": {"fps": -1, "closed_loop_chunks": 2.5, "horizon_seconds": 120}, "split": "val"}))
    row = discover_evaluator_candidates(source)["candidates"][0]
    assert row["fps"] is None
    assert row["closed_loop_chunks"] is None
    assert row["minute_level_supported"] is False
    assert "PROTOCOL_NUMERIC_METADATA_INVALID" in row["protocol_warnings"]


def test_discovery_marks_overflowing_horizon_invalid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    directory = source / "configs" / "evaluation"
    directory.mkdir(parents=True)
    (directory / "eval.json").write_text(json.dumps({"protocol": {"fps": 1e-300, "num_frames": 1e300}}))
    report = discover_evaluator_candidates(source)
    row = report["candidates"][0]
    assert row["horizon_seconds"] is None
    assert row["minute_level_supported"] is False
    assert "HORIZON_METADATA_INVALID" in row["protocol_warnings"]
    json.dumps(report, allow_nan=False)


def test_discovery_limits_files_across_both_search_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wmloop.control.evaluator_discovery as module
    monkeypatch.setattr(module, "_MAX_FILES", 3)
    source = tmp_path / "source"
    for name in ("evaluation", "evaluators"):
        directory = source / "configs" / name
        directory.mkdir(parents=True)
        for index in range(3):
            (directory / f"eval-{index}.json").write_text('{"metrics":["psnr"]}')
    report = discover_evaluator_candidates(source)
    assert report["candidate_count"] == 3
    assert any(row["code"] == "EVALUATOR_SCAN_LIMIT_REACHED" for row in report["warnings"])


def test_discovery_bounds_irrelevant_tree_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wmloop.control.evaluator_discovery as module
    monkeypatch.setattr(module, "_MAX_SCAN_ENTRIES", 4)
    source = tmp_path / "source"
    directory = source / "configs" / "evaluation"
    directory.mkdir(parents=True)
    for index in range(8):
        (directory / f"irrelevant-{index}.txt").write_text("not evaluator metadata")
    report = discover_evaluator_candidates(source)
    assert report["candidate_count"] == 0
    assert any(row["code"] == "EVALUATOR_SCAN_LIMIT_REACHED" for row in report["warnings"])


def test_discovery_does_not_follow_linked_candidate_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "configs").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "eval.json").write_text('{"metrics":["psnr"]}')
    (source / "configs" / "evaluation").symlink_to(outside, target_is_directory=True)
    report = discover_evaluator_candidates(source)
    assert report["candidate_count"] == 0
    assert any(row["code"] == "EVALUATOR_SCAN_SYMLINK" for row in report["warnings"])


def test_discovery_limits_candidate_file_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wmloop.control.evaluator_discovery as module
    monkeypatch.setattr(module, "_MAX_MANIFEST_BYTES", 32)
    source = tmp_path / "source"
    directory = source / "configs" / "evaluation"
    directory.mkdir(parents=True)
    (directory / "eval.json").write_text(json.dumps({"description": "x" * 100}))
    report = discover_evaluator_candidates(source)
    assert report["candidate_count"] == 0
    assert report["warnings"][0]["code"] == "EVALUATOR_CANDIDATE_TOO_LARGE"


def test_overlay_keeps_protocol_metadata_without_inventing_a_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = tmp_path / "model"
    data = tmp_path / "data"
    source.mkdir()
    model.mkdir()
    data.mkdir()
    candidate = source / "eval.json"
    original = {
        "repo_root": "/old/robocoach",
        "source_root": "/old/robocoach",
        "model_root": "/old/checkpoints",
        "data_root": "/old/video_latent",
        "protocol": {"fps": 5},
        "metrics": ["psnr", "ssim"],
    }
    candidate.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "verdi"
    receipt = materialize_evaluator_overlay(
        candidate,
        source_root=source,
        model_root=model,
        data_root=data,
        output_root=output,
    )
    assert receipt["state"] == "binding_required"
    assert receipt["rewrites"]
    overlay = Path(str(receipt["overlay_path"]))
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    manifest = payload["candidate_manifest"]
    assert manifest["repo_root"] == str(source)
    assert manifest["source_root"] == str(source)
    assert manifest["model_root"] == str(model)
    assert manifest["data_root"] == str(data)
    assert manifest["metrics"] == ["psnr", "ssim"]
    assert "command" not in manifest
    assert "verifier" not in manifest
    assert _load_evaluator_contract(overlay)["state"] == "binding_required"
    assert candidate.read_text(encoding="utf-8") == json.dumps(original)
    blocked = materialize_evaluator_overlay(
        candidate,
        source_root=source,
        model_root=model,
        data_root=data,
        output_root=tmp_path / "frozen",
        confirm=True,
    )
    assert blocked["state"] == "binding_required"
    assert blocked["confirmation"]["confirmed"] is False
    assert blocked["optimization_launch_allowed"] is False
    assert all(row["action"] for row in blocked["blockers"])


def _overlay_inputs(tmp_path: Path, payload: dict[str, object]) -> dict[str, Path]:
    source, model, data = (tmp_path / name for name in ("source", "model", "data"))
    for path in (source, model, data):
        path.mkdir()
    script = source / "scripts" / "evaluate.py"
    script.parent.mkdir()
    script.write_text("raise RuntimeError('discovery must never execute model code')\n", encoding="utf-8")
    candidate = source / "eval.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    return {"candidate_path": candidate, "source_root": source, "model_root": model, "data_root": data, "output_root": tmp_path / "overlay"}


def _contract() -> dict[str, object]:
    return {
        "evaluator_id": "existing-paired-evaluator-v1",
        "command": ["{python}", "{repo_root}/scripts/evaluate.py"],
        "input_artifacts": ["checkpoint", "paired_target"],
        "output_artifacts": ["metrics.json"],
        "metrics": ["psnr", "ssim"],
        "verifier": "existing-frozen-verifier-v1",
    }


def test_complete_contract_requires_confirmation_and_preserves_original_contract(tmp_path: Path) -> None:
    contract = _contract()
    inputs = _overlay_inputs(tmp_path, contract)
    review = materialize_evaluator_overlay(**inputs)
    assert review["state"] == "ready_for_confirmation"
    assert review["blockers"] == []
    assert review["optimization_launch_allowed"] is False
    assert _load_evaluator_contract(Path(review["overlay_path"]))["state"] == "binding_required"
    inputs["output_root"] = tmp_path / "frozen"
    frozen = materialize_evaluator_overlay(**inputs, confirm=True)
    assert frozen["state"] == "frozen"
    assert frozen["confirmation"] == {"required": False, "confirmed": True}
    assert json.loads(Path(frozen["overlay_path"]).read_text()) == contract
    assert sha256_bytes(Path(frozen["overlay_path"]).read_bytes()) == frozen["overlay_sha256"]
    assert sha256_bytes(inputs["candidate_path"].read_bytes()) == frozen["source_sha256"]
    assert _load_evaluator_contract(Path(frozen["overlay_path"]))["state"] == "ready"
    assert json.loads(Path(frozen["receipt_path"]).read_text()) == frozen


def test_complete_yaml_contract_produces_json_consumable_by_onboarding(tmp_path: Path) -> None:
    import yaml
    contract = _contract()
    inputs = _overlay_inputs(tmp_path, contract)
    candidate = inputs["candidate_path"].with_suffix(".yaml")
    candidate.write_text(yaml.safe_dump(contract), encoding="utf-8")
    inputs["candidate_path"] = candidate
    receipt = materialize_evaluator_overlay(**inputs, confirm=True)
    assert receipt["state"] == "frozen"
    assert Path(receipt["overlay_path"]).suffix == ".json"
    assert _load_evaluator_contract(Path(receipt["overlay_path"]))["state"] == "ready"


@pytest.mark.parametrize("field,value", [("metrics", []), ("metrics", [1]), ("command", []), ("verifier", ""), ("entrypoint_probe", []), ("command", ["{python}", "missing.py"]), ("command", ["{python}", "-m", "missing.module"])])
def test_invalid_or_missing_entrypoint_contract_cannot_freeze(tmp_path: Path, field: str, value: object) -> None:
    contract = _contract()
    contract[field] = value
    inputs = _overlay_inputs(tmp_path, contract)
    receipt = materialize_evaluator_overlay(**inputs, confirm=True)
    assert receipt["state"] == "binding_required"
    assert receipt["confirmation"]["confirmed"] is False
    assert receipt["blockers"]
    assert _load_evaluator_contract(Path(receipt["overlay_path"]))["state"] == "binding_required"


def test_empty_manifest_does_not_invent_metrics(tmp_path: Path) -> None:
    receipt = materialize_evaluator_overlay(**_overlay_inputs(tmp_path, {}), confirm=True)
    review = json.loads(Path(receipt["overlay_path"]).read_text())
    assert receipt["state"] == "binding_required"
    assert review["candidate_manifest"] == {}


def test_source_root_rewrite_preserves_entrypoint_subpath(tmp_path: Path) -> None:
    contract = _contract() | {"source_root": "/old/external-checkout", "command": ["{python}", "/old/external-checkout/scripts/evaluate.py"]}
    inputs = _overlay_inputs(tmp_path, contract)
    receipt = materialize_evaluator_overlay(**inputs, confirm=True)
    assert receipt["state"] == "frozen"
    rewritten = json.loads(Path(receipt["overlay_path"]).read_text())
    assert rewritten["command"][1] == str(inputs["source_root"] / "scripts" / "evaluate.py")


def test_ambiguous_checkpoint_file_is_preserved_for_binding(tmp_path: Path) -> None:
    contract = _contract() | {"checkpoint": "/old/checkpoints/model.pt"}
    receipt = materialize_evaluator_overlay(**_overlay_inputs(tmp_path, contract), confirm=True)
    assert receipt["state"] == "binding_required"
    review = json.loads(Path(receipt["overlay_path"]).read_text())
    assert review["candidate_manifest"]["checkpoint"] == contract["checkpoint"]
    assert any(row["code"] == "EVALUATOR_PATH_BINDING_REQUIRED" for row in receipt["blockers"])


@pytest.mark.parametrize("field", ["candidate_path", "source_root", "model_root", "data_root", "output_root"])
def test_overlay_rejects_symlink_inputs_and_output(tmp_path: Path, field: str) -> None:
    inputs = _overlay_inputs(tmp_path, _contract())
    target = inputs[field]
    linked = tmp_path / f"linked-{field}"
    linked.symlink_to(target, target_is_directory=field != "candidate_path")
    inputs[field] = linked
    with pytest.raises(EvaluatorOverlayError):
        materialize_evaluator_overlay(**inputs, confirm=True)


def test_overlay_rejects_symlink_parent(tmp_path: Path) -> None:
    inputs = _overlay_inputs(tmp_path, _contract())
    linked = tmp_path / "linked-parent"
    linked.symlink_to(inputs["source_root"], target_is_directory=True)
    inputs["candidate_path"] = linked / "eval.json"
    with pytest.raises(EvaluatorOverlayError, match="EVALUATOR_CANDIDATE_INVALID"):
        materialize_evaluator_overlay(**inputs)


def test_overlay_refuses_existing_output_without_overwriting(tmp_path: Path) -> None:
    inputs = _overlay_inputs(tmp_path, _contract())
    receipt = materialize_evaluator_overlay(**inputs)
    before = {path: path.read_bytes() for path in inputs["output_root"].rglob("*") if path.is_file()}
    with pytest.raises(EvaluatorOverlayError, match="EVALUATOR_OVERLAY_OUTPUT_EXISTS"):
        materialize_evaluator_overlay(**inputs, confirm=True)
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(Path(receipt["receipt_path"]).read_text())["state"] == "ready_for_confirmation"


def test_concurrent_overlay_creation_has_one_winner(tmp_path: Path) -> None:
    inputs = _overlay_inputs(tmp_path, _contract())

    def attempt() -> str:
        try:
            return str(materialize_evaluator_overlay(**inputs, confirm=True)["state"])
        except EvaluatorOverlayError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert outcomes.count("frozen") == 1
    assert sum("EVALUATOR_OVERLAY_OUTPUT_EXISTS" in value for value in outcomes) == 1
    receipt_path = next(inputs["output_root"].rglob("*.overlay.json"))
    receipt = json.loads(receipt_path.read_text())
    assert _load_evaluator_contract(Path(receipt["overlay_path"]))["state"] == "ready"


def test_overlay_write_failure_publishes_no_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wmloop.control.evaluator_overlay as module
    inputs = _overlay_inputs(tmp_path, _contract())
    atomic = module.atomic_write

    def fail_receipt(path: Path, payload: bytes) -> None:
        if path.name.endswith(".overlay.json"):
            raise OSError("injected receipt failure")
        atomic(path, payload)

    monkeypatch.setattr(module, "atomic_write", fail_receipt)
    with pytest.raises(EvaluatorOverlayError, match="EVALUATOR_OVERLAY_WRITE_FAILED"):
        materialize_evaluator_overlay(**inputs, confirm=True)
    assert not inputs["output_root"].exists()
    assert not list(tmp_path.glob(".overlay.*/"))


def test_overlay_schema_rejects_ready_or_executable_review_and_false_confirmation(tmp_path: Path) -> None:
    receipt = materialize_evaluator_overlay(**_overlay_inputs(tmp_path, _contract()))
    review = json.loads(Path(receipt["overlay_path"]).read_text())
    validate_document("evaluator_overlay_review", review)
    validate_document("evaluator_overlay_receipt", receipt)
    with pytest.raises(ContractValidationError):
        validate_document("evaluator_overlay_review", review | {"command": ["{python}", "eval.py"]})
    with pytest.raises(ContractValidationError):
        validate_document("evaluator_overlay_review", review | {"state": "ready"})
    with pytest.raises(ContractValidationError):
        validate_document("evaluator_overlay_receipt", receipt | {"state": "frozen"})


def test_overlay_rejects_input_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = tmp_path / "model"
    data = tmp_path / "data"
    source.mkdir()
    model.mkdir()
    data.mkdir()
    candidate = source / "eval.json"
    candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluatorOverlayError, match="EVALUATOR_OVERLAY_OUTPUT_OVERLAP"):
        materialize_evaluator_overlay(
            candidate,
            source_root=source,
            model_root=model,
            data_root=data,
            output_root=source / "overlay",
        )
