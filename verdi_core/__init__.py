"""Small, model-agnostic VerdiWM control plane."""

from .contracts import canonical_digest
from .autonomy import CodeAuthor, PatchReviewer, WorkspaceManager, assess_replicates
from .resources import GPU, GPUInventory
from .human_eval import HumanVideoBatch, evaluate_labels
from .evidence import MetricSpec, classify_paired_effect, compare_metrics, verify_artifacts
from .training import AdaptiveTrainingController, TrainingPolicy
from .workers import RepairingWorker
from .campaign import CampaignPolicy, CampaignSupervisor
from .knowledge_graph import (
    GRAPH_SCHEMA_VERSION,
    LAYERS,
    PUBLIC_PROBE_FAMILIES,
    build_graph_document,
    build_transfer_index,
    export_bundle,
    import_settlement_entries,
    project_model_portrait,
    write_static_viewer,
)
from .transfer import TransferAssessment, rank_transfer_candidates
from .engineering import EngineeringAgent, EngineeringSandbox, EngineeringTools, EngineeringPolicyError
from .autonomous import autonomous_campaign

__all__ = ["canonical_digest", "CodeAuthor", "PatchReviewer", "WorkspaceManager", "assess_replicates", "GPU", "GPUInventory", "HumanVideoBatch", "evaluate_labels", "MetricSpec", "classify_paired_effect", "compare_metrics", "verify_artifacts", "AdaptiveTrainingController", "TrainingPolicy", "RepairingWorker", "CampaignPolicy", "CampaignSupervisor", "GRAPH_SCHEMA_VERSION", "LAYERS", "PUBLIC_PROBE_FAMILIES", "build_graph_document", "build_transfer_index", "export_bundle", "import_settlement_entries", "project_model_portrait", "write_static_viewer", "TransferAssessment", "rank_transfer_candidates", "EngineeringAgent", "EngineeringSandbox", "EngineeringTools", "EngineeringPolicyError", "autonomous_campaign"]
