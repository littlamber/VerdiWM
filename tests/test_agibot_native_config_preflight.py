from __future__ import annotations

import json

from wmloop.execute._agibot_native_config_preflight import (
    _PREFLIGHT_SCRIPT,
    _loader_failure_reason,
)


def test_embedded_native_preflight_script_compiles_and_uses_structured_output():
    compile(_PREFLIGHT_SCRIPT, "<agibot-native-config-preflight>", "exec")

    assert "sys.stdout.write(" in _PREFLIGHT_SCRIPT
    assert "print(" not in _PREFLIGHT_SCRIPT


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
