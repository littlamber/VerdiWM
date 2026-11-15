from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wmloop.execute._agibot_native_config_preflight import (
    _PREFLIGHT_SCRIPT,
    _loader_failure_reason,
    _optional_native_config_bool,
)


def test_embedded_native_preflight_script_compiles_and_uses_structured_output():
    compile(_PREFLIGHT_SCRIPT, "<agibot-native-config-preflight>", "exec")

    assert "sys.stdout.write(" in _PREFLIGHT_SCRIPT
    assert "print(" not in _PREFLIGHT_SCRIPT
    assert "_optional_native_config_bool" in _PREFLIGHT_SCRIPT


def test_checkpoint_integrity_guidance_is_specific_to_user_selected_inputs():
    normalized = " ".join(_PREFLIGHT_SCRIPT.split())

    assert "Select a checkpoint compatible with the chosen native " in normalized
    assert "source and training configuration" in normalized
    assert "checkpoint-134750" not in normalized


def test_optional_native_config_flags_default_to_disabled_when_absent():
    world_model = SimpleNamespace()

    assert _optional_native_config_bool(world_model, "history_ref_enabled") is False
    assert _optional_native_config_bool(world_model, "history_ref_null") is False


@pytest.mark.parametrize("value", [True, False])
def test_optional_native_config_flag_preserves_explicit_booleans(value: bool):
    world_model = SimpleNamespace(history_ref_enabled=value)

    assert _optional_native_config_bool(world_model, "history_ref_enabled") is value


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_optional_native_config_flag_rejects_non_booleans(value: object):
    world_model = SimpleNamespace(history_ref_enabled=value)

    with pytest.raises(TypeError, match="history_ref_enabled must be a boolean"):
        _optional_native_config_bool(world_model, "history_ref_enabled")


def test_native_preflight_failure_redacts_absolute_paths():
    stdout = json.dumps(
        {
            "error_type": "OSError",
            "reason": "could not read /srv/private/wan/checkpoint/dit_model.safetensors",
        }
    )

    reason = _loader_failure_reason(stdout)

    assert "/srv/private" not in reason
    assert "<path>" in reason
