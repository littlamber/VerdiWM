from __future__ import annotations

import json
from pathlib import Path

from wmloop.cli import main
from wmloop.control.llm_setup import inspect_llm_config
from wmloop.execute.configured_llm_broker import load_config


def _write_config(path: Path, body: str) -> None:
    path.write_text("[llm]\n" + body, encoding="utf-8")


def test_config_without_token_file_uses_explicit_environment(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        'base_url = "https://api.example.com"\nmodel = "demo"\ntoken_environment_key = "MY_VERDI_TOKEN"\n',
    )
    monkeypatch.setenv("MY_VERDI_TOKEN", "test-token")
    config = load_config(config_path)
    assert config["token_file"] is None
    assert config["token_environment_key"] == "MY_VERDI_TOKEN"
    assert inspect_llm_config(config_path)["state"] == "ready"


def test_conventional_auth_file_is_used_only_when_it_is_a_regular_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    auth = tmp_path / "auth"
    auth.write_text("test-token", encoding="utf-8")
    auth.chmod(0o600)
    _write_config(config_path, 'base_url = "https://api.example.com"\nmodel = "demo"\n')
    assert load_config(config_path)["token_file"] == auth

    auth.unlink()
    auth.symlink_to(tmp_path / "real-auth")
    assert load_config(config_path)["token_file"] is None


def test_llm_status_does_not_print_token_and_reports_permissions(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.toml"
    auth = tmp_path / "auth"
    auth.write_text("super-secret-token", encoding="utf-8")
    auth.chmod(0o644)
    _write_config(config_path, 'base_url = "https://api.example.com"\nmodel = "demo"\ntoken_file = "auth"\n')
    assert main(["llm", "status", "--config", str(config_path)]) == 2
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["state"] == "blocked"
    assert payload["credential"]["problem"] == "token_file_permissions"
    assert "super-secret-token" not in output


def test_llm_guide_is_available_without_a_config(capsys, tmp_path: Path) -> None:
    assert main(["llm", "--config", str(tmp_path / "config.toml")]) == 0
    output = capsys.readouterr().out
    assert "read -r -s" in output
    assert "verdi llm status" in output
    assert "不是你的 Wan、JEPA" in output


def test_no_auth_local_provider_is_ready_without_a_token(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        'base_url = "http://127.0.0.1:11434"\nmodel = "local"\napi_style = "chat_completions"\nauth_required = false\n',
    )
    report = inspect_llm_config(config_path)
    assert report["state"] == "ready"
    assert report["credential"]["source"] == "none"


def test_no_auth_provider_ignores_an_insecure_legacy_auth_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    auth = tmp_path / "auth"
    auth.write_text("not-used", encoding="utf-8")
    auth.chmod(0o644)
    _write_config(
        config_path,
        'base_url = "http://127.0.0.1:11434"\nmodel = "local"\nauth_required = false\n',
    )
    report = inspect_llm_config(config_path)
    assert report["state"] == "ready"
    assert report["credential"]["required"] is False


def test_status_handles_malformed_endpoint_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, 'base_url = "https://[broken"\nmodel = "demo"\n')
    report = inspect_llm_config(config_path)
    assert report["state"] == "invalid"
    assert report["error"] == "OPENAI_BROKER_ENDPOINT_INVALID"
