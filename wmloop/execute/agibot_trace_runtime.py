"""Compatibility API for the standard-library Agibot trace runtime.

The implementation is split by configuration, recording, and validation so
that the native model environment can use the trace bridge without importing
Verdi's control-plane dependencies.
"""

from __future__ import annotations

if __package__:
    from ._agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_ARTIFACT_TYPE,
        TRACE_CONFIG_TYPE,
        _TRACE_KEY,
        _finite_float,
        _is_finite_number,
        _is_sha256,
        _rank_from_environment,
        _read_json,
        _validate_sample,
        _write_json,
        canonical_digest,
    )
    from ._agibot_trace_config import (
        build_bridge_command,
        load_trace_config,
        trace_shard_path,
        validate_native_training_command,
        write_trace_config,
    )
    from ._agibot_trace_recording import (
        _FrozenValidationView,
        _Recorder,
        _parameter_sample_delta,
        _sample_optimizer_parameters,
        install,
    )
    from ._agibot_trace_validation import (
        _event_sort_key,
        _read_shard,
        _validate_config,
        _validate_footer,
        _validate_global_events,
        _validate_header,
        _validate_rank_events,
        _validate_rank_validation_losses,
        _validate_train_batch_shape,
        merge_trace_shards,
    )
else:
    from _agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_ARTIFACT_TYPE,
        TRACE_CONFIG_TYPE,
        _TRACE_KEY,
        _finite_float,
        _is_finite_number,
        _is_sha256,
        _rank_from_environment,
        _read_json,
        _validate_sample,
        _write_json,
        canonical_digest,
    )
    from _agibot_trace_config import (
        build_bridge_command,
        load_trace_config,
        trace_shard_path,
        validate_native_training_command,
        write_trace_config,
    )
    from _agibot_trace_recording import (
        _FrozenValidationView,
        _Recorder,
        _parameter_sample_delta,
        _sample_optimizer_parameters,
        install,
    )
    from _agibot_trace_validation import (
        _event_sort_key,
        _read_shard,
        _validate_config,
        _validate_footer,
        _validate_global_events,
        _validate_header,
        _validate_rank_events,
        _validate_rank_validation_losses,
        _validate_train_batch_shape,
        merge_trace_shards,
    )
