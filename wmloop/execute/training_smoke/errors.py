"""Training and frozen evaluation smoke components: errors."""

from __future__ import annotations


class OrchestratorTrainingEvalSmokeError(RuntimeError):
    """The M3 training+evaluation smoke failed closed."""
