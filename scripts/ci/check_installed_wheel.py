"""Exercise public entrypoints outside the checkout in a clean wheel install."""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def probe_effect_queue() -> None:
    """Construct a real queue without a source checkout or model execution."""
    from wmloop.experiments.effect_label_queue import build_effect_label_gate_queue

    root = Path.cwd() / "queue-fixture"
    vendor = root / "model-code"
    reports = root / "reports"
    retained = reports / "retained"
    for directory in (vendor, retained, root / "data", root / "weights"):
        directory.mkdir(parents=True)
    checkpoint = retained / "candidate.pt"
    checkpoint.write_bytes(b"test_fixture_only")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (retained / "manifest.json").write_text(json.dumps({"test_fixture": True, "retained_path": str(checkpoint), "sha256": digest}))
    (reports / "source.json").write_text(json.dumps({"test_fixture": True, "state": "ready", "environment": "reacher"}))
    for name in ("freeze.json", "heldout.json"):
        (root / name).write_text('{"test_fixture": true}')
    plan = root / "plan.json"
    plan.write_text(json.dumps({"test_fixture": True, "artifact_type": "verdiwm-acwm-effect-label-completion-plan", "actions": [{"action": "official_gate_existing_checkpoint", "environment": "reacher", "primitive": "next_forcing", "seed": 1, "source_manifest": "source.json", "checkpoint_ref": "retained/candidate.pt", "checkpoint_sha256": digest}]}))
    build_effect_label_gate_queue(completion_plan_path=plan, reports_root=reports, repo_root=root, output_root=root / "queue", runtime_python=Path(sys.executable), data_root=root / "data", checkpoint_root=root / "weights", gpus=(0,), vendor_root=vendor, dataset_freeze=root / "freeze.json", heldout_protocol=root / "heldout.json")
    queue = json.loads((root / "queue/autoloop-queue.json").read_text())
    command = queue["rows"][0]["launch_argv_template"]
    if command[:3] != [sys.executable, "-m", "wmloop.experiments.formal_visualization"]:
        raise RuntimeError("WHEEL_QUEUE_ENTRYPOINT_INVALID")
    subprocess.run(command[:3] + ["--help"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def probe() -> None:
    from wmloop.contracts import ContractValidationError, validate_document
    from wmloop.experiments.formal_visualization import AcwmFormalVisualizationError, run_export
    from wmloop.control.workbench import WorkbenchServer
    from experiments.ctrl_world_autonomous_transfer_v1 import workflow
    import wmloop

    if not Path(wmloop.__file__).is_relative_to(Path(sys.prefix)):
        raise RuntimeError("WHEEL_PROBE_IMPORTED_CHECKOUT")
    document = {
        "schema_version": 1, "artifact_type": "verdiwm-community-bundle-signature",
        "algorithm": "ed25519-v1", "publisher_id": "fixture",
        "signing_key_id": "key-" + "0" * 16, "public_key": "A" * 43 + "=",
        "manifest_sha256": "0" * 64, "signature": "A" * 86 + "==",
        "claim_boundary": "Installation fixture only",
    }
    validate_document("community_bundle_signature", document)
    try:
        validate_document("community_bundle_signature", {})
    except ContractValidationError:
        pass
    else:
        raise RuntimeError("WHEEL_SCHEMA_NOT_ENFORCED")
    try:
        run_export(output_root=Path("unused"), split="invalid")
    except AcwmFormalVisualizationError as exc:
        if str(exc) != "ACWM_FORMAL_VIS_SPLIT_INVALID":
            raise
    else:
        raise RuntimeError("WHEEL_EXPORT_VALIDATION_NOT_ENFORCED")
    from http.client import HTTPConnection
    from threading import Thread
    server = WorkbenchServer(("127.0.0.1", 0), state_root=Path("state"))
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/assets/workbench.js")
        response = connection.getresponse()
        if response.status != 200 or b"function renderCampaigns" not in response.read():
            raise RuntimeError("WHEEL_UI_SOURCE_MISSING")
        connection.close()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/api/campaigns")
        response = connection.getresponse()
        if response.status != 401:
            raise RuntimeError("WHEEL_API_AUTH_MISSING")
        response.read()
        connection.close()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/api/campaigns", headers={"Authorization": "Bearer " + server.auth_token})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("WHEEL_API_AUTH_FAILED")
        response.read()
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    logging.info("Installed wheel: schema validation, exporter, UI assets and entrypoints passed")
    probe_effect_queue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--prepare-locked-cache", action="store_true", help="Populate the hashed locked dependency cache before offline validation.")
    parser.add_argument("--dependency-mode", choices=("locked", "resolved"), default="locked")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe()
        return
    if args.wheel is None and not args.prepare_locked_cache:
        parser.error("wheel is required")
    wheel = args.wheel.resolve(strict=True) if args.wheel else None
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="verdiwm-installed-") as temporary:
        root = Path(temporary)
        python = root / "env/bin/python"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(root / "env")], check=True, cwd=root, env=environment)
        if args.dependency_mode == "locked" or args.prepare_locked_cache:
            requirements = root / "requirements.txt"
            subprocess.run(["uv", "export", "--locked", "--no-default-groups", "--no-emit-project", "--output-file", str(requirements)], check=True, env=environment, stdout=subprocess.DEVNULL)
            installation = subprocess.run(["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r", str(requirements)], cwd=root, env=environment, capture_output=True, text=True)
            sys.stderr.write(installation.stderr)
            if installation.returncode:
                # Publish only fixed diagnostic flags; raw subprocess text can
                # include paths and stays in the runner's normal private log.
                detail = installation.stderr.casefold()
                logging.error("::error::Locked dependency installation failed: offline=%s cache_miss=%s hash_error=%s", environment.get("UV_OFFLINE") == "1", "cache" in detail or "offline" in detail, "hash" in detail)
                installation.check_returncode()
            if args.prepare_locked_cache:
                logging.info("Locked dependency cache prepared; wheel validation has not run.")
                return
            subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], check=True, cwd=root, env=environment)
        else:
            # Exercise the wheel's own dependency ranges without the repo lock.
            subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True, cwd=root, env=environment)
        for name, arguments in (
            ("verdi", ["--help"]),
            ("verdi", ["--version"]),
            ("verdiwm", ["--help"]),
            ("verdiwm", ["evidence", "project", "--help"]),
            ("verdiwm", ["community", "verify", "--help"]),
            ("verdiwm-workbench", ["--help"]),
            ("verdiwm-effect-label-queue", ["--help"]),
            ("verdiwm-open-method-study", ["--help"]),
            ("verdiwm-generate-method", ["--help"]),
            ("verdiwm-calibrate-method", ["--help"]),
            ("verdiwm-run-open-method-study", ["--help"]),
            ("verdiwm-ctrl-world-autonomous-transfer", ["--help"]),
        ):
            subprocess.run([str(root / "env/bin" / name), *arguments], check=True, cwd=root, env=environment, stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-I", str(Path(__file__).resolve()), "--probe"], check=True, cwd=root, env=environment)
        subprocess.run([str(python), "-I", "-m", "wmloop.experiments.formal_visualization", "--help"], check=True, cwd=root, env=environment, stdout=subprocess.DEVNULL)
        if args.report:
            from wmloop.storage import atomic_write, checked_path, sha256_file

            packages = json.loads(subprocess.run(["uv", "pip", "list", "--python", str(python), "--format", "json"], check=True, cwd=root, env=environment, capture_output=True).stdout)
            report = {"status": "passed", "dependency_mode": args.dependency_mode, "offline": environment.get("UV_OFFLINE") == "1", "wheel_sha256": sha256_file(wheel), "packages": [{"name": p["name"], "version": p["version"]} for p in packages]}
            target = checked_path(args.report, code="INSTALL_REPORT_SYMLINK")
            atomic_write(target, (json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
