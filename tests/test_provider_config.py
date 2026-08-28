from pathlib import Path
import time

import pytest

from verdi_core.providers import OpenAICompatibleProvider


def test_provider_reads_user_config_without_sourcing_shell(monkeypatch, tmp_path: Path):
    (tmp_path / "ai.env").write_text(
        'export VERDI_AI_BASE_URL="https://example.test/v1"\n'
        'export VERDI_AI_API_' + 'KEY="test-key-placeholder"\n'
        'export VERDI_AI_MODEL="model-a"\n'
        'export VERDI_AI_REASONING_EFFORT="high"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VERDI_CONFIG_DIR", str(tmp_path))
    for name in ("VERDI_AI_BASE_URL", "VERDI_AI_API_KEY", "VERDI_AI_MODEL", "VERDI_AI_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)

    provider = OpenAICompatibleProvider.from_env()

    assert provider is not None
    assert provider.base_url == "https://example.test/v1"
    assert provider.model == "model-a"
    assert provider.api_key == "test-key-placeholder"
    assert provider.reasoning_effort == "high"


def test_environment_overrides_local_config(monkeypatch, tmp_path: Path):
    (tmp_path / "ai.env").write_text(
        'VERDI_AI_BASE_URL="https://local.test/v1"\nVERDI_AI_MODEL="local"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VERDI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VERDI_AI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("VERDI_AI_MODEL", "env")

    provider = OpenAICompatibleProvider.from_env()

    assert provider is not None
    assert (provider.base_url, provider.model) == ("https://env.test/v1", "env")


def test_environment_configures_total_timeout(monkeypatch):
    monkeypatch.setenv("VERDI_AI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("VERDI_AI_MODEL", "env")
    monkeypatch.setenv("VERDI_AI_TOTAL_TIMEOUT", "7")
    provider = OpenAICompatibleProvider.from_env()
    assert provider is not None
    assert provider.total_timeout == 7


def test_provider_can_use_env_file_when_toml_parser_is_unavailable(monkeypatch, tmp_path: Path):
    (tmp_path / "ai.env").write_text(
        'VERDI_AI_BASE_URL="https://local.test/v1"\nVERDI_AI_MODEL="local"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VERDI_CONFIG_DIR", str(tmp_path))
    for name in ("VERDI_AI_BASE_URL", "VERDI_AI_API_KEY", "VERDI_AI_MODEL", "VERDI_AI_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    import verdi_core.providers as providers

    monkeypatch.setattr(providers, "tomllib", None)
    provider = providers.OpenAICompatibleProvider.from_env()

    assert provider is not None
    assert (provider.base_url, provider.model) == ("https://local.test/v1", "local")


def test_provider_enforces_wall_clock_deadline(monkeypatch):
    """A peer that keeps a socket open cannot stall autonomous recovery."""
    import verdi_core.providers as providers

    class HangingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            time.sleep(0.2)
            return b'{"choices": []}'

    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *_args, **_kwargs: HangingResponse())
    provider = OpenAICompatibleProvider("https://example.test/v1", "model", total_timeout=0.03, timeout=0.03, max_retries=0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="total timeout"):
        provider.complete(role="engineering_agent", prompt="repair")
    assert time.monotonic() - started < 0.15
