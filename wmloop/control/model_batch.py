"""Stable compatibility API; implementations are split by responsibility."""

from .batch_state import ModelBatchError
from .model_batch_plan import (
    _batch_campaign_id,
    compile_model_batch,
    load_model_batch_plan,
    _compile_model_row,
    _required_path,
    _sha256_path,
    _write_plan,
    _digest,
)
from .model_batch_status import (
    load_model_batch_execution,
    build_model_batch_execution_binding,
    summarize_model_batch,
)
from .model_batch_run import (
    _locked_batch,
    run_model_batch,
    _save_execution_snapshot,
    _campaign_store,
    _compile_batch_payload,
    _write_execution_manifest,
)
