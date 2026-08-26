from pathlib import Path

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
