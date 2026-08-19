#!/usr/bin/env bash
set -euo pipefail

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

uv build --wheel --out-dir "$build_dir"
wheel_path="$(find "$build_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/__init__.py" in names; assert "wmloop/execute/experiment_scheduler.py" in names; assert "configs/schemas/goal_spec.schema.json" in names; assert "configs/schemas/cross_backbone_experiment.schema.json" in names; assert "configs/schemas/experiment_stage_receipt.schema.json" in names; assert "configs/schemas/backbone_primitive_registry.schema.json" in names; assert "configs/schemas/auto_experiment_candidate_batch.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/onboarding.py" in names; assert "configs/schemas/model_onboarding_report.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/onboarding_conformance.py" in names; assert "configs/schemas/model_conformance_receipt.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/onboarding_admission.py" in names; assert "wmloop/control/onboarding_compiler.py" in names; assert "wmloop/execute/external_evaluator_workload.py" in names; assert "configs/onboarding/ctrl_world_replay_evaluator_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/execute/autonomous_pipeline.py" in names; assert "configs/schemas/autonomous_pipeline_manifest.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/diagnose/probe_campaign.py" in names; assert "wmloop/retrieve/index.py" in names; assert "wmloop/retrieve/literature.py" in names; assert "wmloop/retrieve/method_staging.py" in names; assert "configs/schemas/diagnostic_probe_contract.schema.json" in names; assert "configs/schemas/literature_method_candidate.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/execute/campaign_daemon.py" in names; assert "wmloop/execute/pipeline_daemon.py" in names; assert "configs/schemas/campaign_daemon_manifest.schema.json" in names; assert "configs/schemas/pipeline_daemon_manifest.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/cli.py" in names; assert "wmloop/control/adapter_profiles.py" in names; assert "wmloop/control/campaign_api.py" in names; assert "wmloop/control/campaign_dispatcher.py" in names; assert "wmloop/experiments/ctrl_world_settlement_import.py" in names; assert "configs/adapters/ctrl_world_predictive_v2.json" in names; assert "configs/schemas/adapter_profile.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/method_candidate_compiler.py" in names; assert "configs/schemas/method_candidate_catalog.schema.json" in names; assert "configs/schemas/method_candidate_compilation.schema.json" in names; assert "scripts/run_ctrl_world_local_fingerprint_probe.py" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/retrieve/evidence_capsule.py" in names; assert "wmloop/retrieve/mechanism_discovery.py" in names; assert "wmloop/geometry/memory.py" in names; assert "configs/retrieval/mechanism_tag_ontology_v1.json" in names; assert "configs/retrieval/primitive_mechanism_profiles_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/geometry/portable_experience.py" in names; assert "configs/schemas/portable_experience.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/evaluate/system_utility.py" in names; assert "configs/schemas/system_utility_audit.schema.json" in names; assert "configs/experiments/system_utility_audit_v1.json" in names; assert "configs/experiments/ctrl_world_experience_utility_canary_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/acwm_dual_evaluation.py" in names; assert "configs/schemas/acwm_dual_evaluation.schema.json" in names; assert "configs/experiments/ctrl_world_acwm_dual_evaluation_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/verify/acwm_frozen_verifier.py" in names; assert "configs/schemas/acwm_frozen_verifier_policy.schema.json" in names; assert "configs/schemas/acwm_frozen_verdict.schema.json" in names; assert "configs/schemas/acwm_verification_manifest.schema.json" in names; assert "configs/experiments/ctrl_world_acwm_frozen_verifier_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/experiments/acwm_verified_evidence.py" in names; assert "configs/schemas/acwm_verified_evidence.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/execute/automatic_materialization.py" in names; assert "configs/schemas/automatic_materialization_plan.schema.json" in names; assert "configs/schemas/materialized_method_descriptor.schema.json" in names; assert "configs/schemas/automatic_materialization_receipt.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/acwm_campaign.py" in names; assert "configs/schemas/acwm_candidate_batch.schema.json" in names; assert "configs/schemas/acwm_autoloop.schema.json" in names; assert "configs/schemas/acwm_research_intake.schema.json" in names; assert "configs/experiments/ctrl_world_acwm_guidance_batch_v1.json" in names; assert "configs/experiments/ctrl_world_acwm_guidance_confirm_batch_v1.json" in names; assert "configs/experiments/ctrl_world_acwm_autoloop_v1.json" in names; assert "configs/experiments/ctrl_world_acwm_research_intake_v1.json" in names; assert "experiments/ctrl_world_acwm_guidance_v1/autoloop.py" in names; assert "experiments/ctrl_world_acwm_guidance_v1/research_intake.py" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/experiments/engineering.py" in names; assert "wmloop/experiments/training_scale.py" in names; assert "wmloop/retrieve/training_recipes.py" in names; assert "configs/schemas/experiment_engineering_manifest.schema.json" in names; assert "configs/schemas/training_scale_plan.schema.json" in names; assert "configs/schemas/world_model_training_recipe_registry.schema.json" in names; assert "configs/retrieval/world_model_training_recipes_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/research_proposal.py" in names; assert "wmloop/control/workflow_plugins.py" in names; assert "configs/schemas/research_proposal.schema.json" in names; assert "configs/schemas/compiled_experiment_manifest.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/intermediate_ir.py" in names; assert "wmloop/geometry/evidence_ir.py" in names; assert "configs/plugins/core_workflows_v1.json" in names; assert "configs/schemas/model_capability_ir.schema.json" in names; assert "configs/schemas/experiment_ir.schema.json" in names; assert "configs/schemas/evidence_ir.schema.json" in names; assert "configs/schemas/workflow_plugin_registry.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/model_portrait.py" in names; assert "configs/schemas/model_portrait.schema.json" in names; assert "configs/schemas/portrait_readiness_receipt.schema.json" in names; assert "configs/schemas/portrait_observation_work_order.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/adaptive_observation.py" in names; assert "configs/plugins/observation_module_abis_v1.json" in names; assert "configs/schemas/observation_module_abi_registry.schema.json" in names; assert "configs/schemas/adaptive_probe_plan.schema.json" in names; assert "configs/schemas/interface_extension_spec.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/module_composition.py" in names; assert "configs/plugins/automatic_module_abis_v1.json" in names; assert "configs/schemas/automatic_module_abi_registry_v2.schema.json" in names; assert "configs/schemas/module_composition_receipt.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/capability_gap_planner.py" in names; assert "configs/schemas/goal_ir.schema.json" in names; assert "configs/schemas/capability_requirement_graph.schema.json" in names; assert "configs/schemas/capability_gap_plan_receipt.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/experiment_portfolio.py" in names; assert "configs/schemas/experiment_hypothesis_batch.schema.json" in names; assert "configs/schemas/experiment_portfolio.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/module_manufacturing.py" in names; assert "configs/schemas/module_manufacturing_work_order.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/resource_portfolio.py" in names; assert "configs/schemas/resource_portfolio_receipt.schema.json" in names; assert "experiments/ctrl_world_autonomous_transfer_v1/scale_plan.json" in names; assert "experiments/droid_ctrl_world_conversion_v1/scale_plan.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/geometry/community_knowledge.py" in names; assert "wmloop/experiments/portable_knowledge_graph.py" in names; assert "configs/schemas/portrait_transition.schema.json" in names; assert "configs/schemas/protocol_contract.schema.json" in names; assert "configs/schemas/transformation_contract.schema.json" in names; assert "configs/schemas/knowledge_lifecycle.schema.json" in names; assert "configs/schemas/portable_knowledge_quality_audit.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/control/acwm_materialized_campaign.py" in names; assert "wmloop/verify/acwm_materialized_frozen_verifier.py" in names; assert "wmloop/experiments/materialized_transfer_evidence.py" in names; assert "wmloop/experiments/verified_transfer_knowledge.py" in names; assert "configs/experiments/ctrl_world_acwm_materialized_frozen_verifier_v1.json" in names; assert "configs/schemas/acwm_materialized_candidate_batch.schema.json" in names; assert "configs/schemas/acwm_materialized_frozen_verifier_policy.schema.json" in names; assert "configs/schemas/acwm_materialized_frozen_verdict.schema.json" in names; assert "configs/schemas/acwm_materialized_verification_manifest.schema.json" in names; assert "configs/schemas/materialized_transfer_evidence.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "experiments/ctrl_world_research_loop_v2/research_intake.py" in names; assert "configs/experiments/ctrl_world_acwm_research_intake_v2.json" in names; assert "configs/schemas/acwm_research_intake_v2.schema.json" in names; assert "configs/schemas/acwm_research_idea_v2.schema.json" in names; assert "configs/schemas/acwm_research_work_order_v2.schema.json" in names; assert "configs/schemas/acwm_source_transfer_assessment_v2.schema.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "experiments/ctrl_world_hybrid_memory_transfer_v1/materialize.py" in names; assert "experiments/ctrl_world_hybrid_memory_transfer_v1/evaluate.py" in names; assert "experiments/ctrl_world_hybrid_memory_transfer_v1/run.py" in names; assert "configs/experiments/ctrl_world_hybrid_relevance_memory_materialization_v1.json" in names' "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "experiments/ctrl_world_autonomous_transfer_v1/controller.py" in names; assert "experiments/ctrl_world_autonomous_transfer_v1/state.py" in names; assert "experiments/ctrl_world_autonomous_transfer_v1/workflow.py" in names; assert "experiments/ctrl_world_autonomous_transfer_v1/replanning.py" in names; assert "configs/experiments/ctrl_world_autonomous_transfer_loop_v1.json" in names; assert "configs/schemas/ctrl_world_autonomous_transfer_loop.schema.json" in names; assert "configs/schemas/closed_loop_next_task.schema.json" in names; assert "configs/schemas/closed_loop_quality_audit.schema.json" in names; assert "configs/schemas/terminal_archive_receipt.schema.json" in names' "$wheel_path"

uv run python -m py_compile \
  wmloop/geometry/types.py \
  wmloop/geometry/irg.py \
  wmloop/geometry/transfer.py \
  wmloop/geometry/memory.py \
  wmloop/geometry/evidence_ir.py \
  wmloop/geometry/portable_experience.py \
  wmloop/geometry/community_knowledge.py \
  wmloop/control/model_portrait.py \
  wmloop/control/adaptive_observation.py \
  wmloop/control/module_composition.py \
  wmloop/control/capability_gap_planner.py \
  wmloop/control/experiment_portfolio.py \
  wmloop/control/module_manufacturing.py \
  wmloop/control/resource_portfolio.py \
  wmloop/geometry/evolution.py \
  wmloop/experiments/spec.py \
  wmloop/experiments/lobo.py \
  wmloop/experiments/ledger.py \
  wmloop/experiments/report.py \
  wmloop/experiments/ctrl_world_receipt_merge.py \
  wmloop/experiments/ctrl_world_fingerprint_settlement.py \
  wmloop/experiments/cosmos3_fingerprint.py \
  wmloop/experiments/cosmos3_directional_settlement.py \
  wmloop/experiments/probe_evolution.py \
  wmloop/experiments/probe_semantic_compile.py \
  wmloop/experiments/certificate_ablation.py \
  wmloop/experiments/cross_backbone_reuse_audit.py \
  wmloop/experiments/evidence_graph.py \
  wmloop/experiments/acwm_verified_evidence.py \
  wmloop/experiments/ctrl_world_settlement_import.py \
  wmloop/experiments/engineering.py \
  wmloop/experiments/training_scale.py \
  wmloop/retrieve/training_recipes.py \
  wmloop/control/research_proposal.py \
  wmloop/control/workflow_plugins.py \
  wmloop/control/intermediate_ir.py \
  wmloop/control/acwm_dual_evaluation.py \
  wmloop/control/acwm_campaign.py \
  wmloop/control/acwm_materialized_campaign.py \
  wmloop/verify/acwm_frozen_verifier.py \
  wmloop/verify/acwm_materialized_frozen_verifier.py \
  wmloop/experiments/materialized_transfer_evidence.py \
  wmloop/experiments/verified_transfer_knowledge.py \
  wmloop/experiments/portable_knowledge_graph.py \
  scripts/evaluate_ctrl_world_acwm_dual.py \
  experiments/ctrl_world_acwm_guidance_v1/run.py \
  experiments/ctrl_world_acwm_guidance_v1/autoloop.py \
  experiments/ctrl_world_acwm_guidance_v1/research_intake.py \
  experiments/ctrl_world_research_loop_v2/research_intake.py \
  experiments/ctrl_world_hybrid_memory_transfer_v1/materialize.py \
  experiments/ctrl_world_hybrid_memory_transfer_v1/evaluate.py \
  experiments/ctrl_world_hybrid_memory_transfer_v1/run.py \
  experiments/ctrl_world_masked_intermediate_adapter_v1/materialize.py \
  experiments/ctrl_world_masked_intermediate_adapter_v1/evaluate.py \
  experiments/ctrl_world_autonomous_transfer_v1/replanning.py \
  experiments/ctrl_world_autonomous_transfer_v1/state.py \
  experiments/ctrl_world_autonomous_transfer_v1/workflow.py \
  experiments/ctrl_world_autonomous_transfer_v1/controller.py \
  wmloop/evaluate/adapters/ctrl_world.py \
  wmloop/primitives/adapters/backbone_registry.py \
  wmloop/primitives/adapters/ctrl_world_hooks.py \
  wmloop/primitives/adapters/cosmos3_hooks.py \
  wmloop/control/agent_engineering_policy.py \
  wmloop/control/backbone_capability_matrix.py \
  wmloop/control/cosmos3_gpu_runtime_receipt.py \
  wmloop/control/onboarding.py \
  wmloop/control/onboarding_conformance.py \
  wmloop/control/onboarding_admission.py \
  wmloop/control/onboarding_compiler.py \
  wmloop/control/method_candidate_compiler.py \
  wmloop/control/adapter_profiles.py \
  wmloop/control/campaign_api.py \
  wmloop/control/campaign_dispatcher.py \
  wmloop/cli.py \
  wmloop/execute/experiment_scheduler.py \
  wmloop/execute/external_evaluator_workload.py \
  wmloop/execute/autonomous_pipeline.py \
  wmloop/execute/automatic_materialization.py \
  wmloop/execute/campaign_daemon.py \
  wmloop/execute/pipeline_daemon.py \
  wmloop/evaluate/system_utility.py \
  wmloop/diagnose/probe_campaign.py \
  wmloop/retrieve/index.py \
  wmloop/retrieve/literature.py \
  wmloop/retrieve/method_staging.py \
  wmloop/retrieve/evidence_capsule.py \
  wmloop/retrieve/mechanism_discovery.py \
  wmloop/propose/primitive_materialization_prompt.py \
  wmloop/diagnose/diagnostic_probe_materialization_prompt.py \
  wmloop/diagnose/diagnostic_probe_routing_admission.py \
  wmloop/diagnose/probes/acwm_push_cube_contact_event_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_cube_action_binding_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_cube_rigid_pose_slip_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_stack_cube_support_relation_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_stack_cube_contact_instability_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_stack_cube_object_identity_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_pour_water_container_boundary_leak_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_pour_water_free_surface_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_pour_water_fluid_volume_transport_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_sand_granular_frontier_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_sand_mass_redistribution_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_sand_particle_boundary_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_reacher_target_conditioning_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_reacher_inverse_dynamics_confidence_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_reacher_endpoint_control_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_rope_topology_change_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_rope_endpoint_path_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_push_rope_deformable_contact_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_cloth_move_surface_fold_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_cloth_move_cloth_identity_drift_diagnostic_v1.py \
  wmloop/diagnose/probes/acwm_cloth_move_deformable_memory_diagnostic_v1.py \
  scripts/export/verdiwm_minimal_loop_bundle.py \
  scripts/export/acwm_public_experience_bundle.py \
  scripts/export/validate_public_example.py \
  scripts/export/build_github_staging.py \
  scripts/export/merge_ctrl_world_receipt_indexes.py \
  scripts/export/ctrl_world_fingerprint_settlement.py \
  scripts/export/ctrl_world_fingerprint_public_bundle.py \
  scripts/export/cosmos3_forward_dynamics_public_bundle.py \
  scripts/export/cosmos3_fingerprint_public_bundle.py \
  scripts/export/cosmos3_directional_settlement.py \
  scripts/export/cosmos3_directional_settlement_public_bundle.py \
  scripts/export/probe_evolution.py \
  scripts/export/probe_evolution_settlement.py \
  scripts/export/acwm_formal_visualization.py \
  scripts/export/acwm_training_seed_replication_queue.py \
  scripts/export/acwm_training_seed_replication_summary.py \
  scripts/evaluate_cosmos3_paired_gt.py \
  scripts/run_cosmos3_fingerprint_campaign.py \
  scripts/integrations/run_cosmos3_droid_lerobot_fd.py \
  scripts/integrations/run_cosmos3_inference_with_r31_probe.py \
  scripts/export/r31_cross_backbone_runtime_bundle.py \
  scripts/run_ctrl_world_bounded_smoke.py \
  scripts/run_ctrl_world_local_fingerprint_probe.py \
  scripts/aggregate_ctrl_world_local_fingerprint.py \
  scripts/freeze_ctrl_world_directional_selector.py

uv run pytest -q \
  tests/test_experiment_scheduler.py \
  tests/test_auto_experiment_control_plane.py \
  tests/test_model_onboarding.py \
  tests/test_onboarding_conformance.py \
  tests/test_onboarding_compiler.py \
  tests/test_method_candidate_compiler.py \
  tests/test_external_evaluator_workload.py \
  tests/test_autonomous_pipeline.py \
  tests/test_probe_retrieval.py \
  tests/test_literature_method_staging.py \
  tests/test_evidence_capsule.py \
  tests/test_mechanism_discovery.py \
  tests/test_transferable_experience.py \
  tests/test_portable_experience.py \
  tests/test_system_utility.py \
  tests/test_acwm_dual_evaluation.py \
  tests/test_acwm_campaign.py \
  tests/test_acwm_frozen_verifier.py \
  tests/test_automatic_materialization.py \
  tests/test_automatic_module_generation.py \
  tests/test_module_composition.py \
  tests/test_capability_gap_planner.py \
  tests/test_experiment_portfolio.py \
  tests/test_module_manufacturing.py \
  tests/test_resource_portfolio.py \
  tests/test_acwm_autoloop.py \
  tests/test_acwm_research_intake.py \
  tests/test_acwm_research_intake_v2.py \
  tests/test_hybrid_relevance_materializer.py \
  tests/test_masked_intermediate_action_materializer.py \
  tests/test_acwm_materialized_campaign.py \
  tests/test_acwm_materialized_frozen_verifier.py \
  tests/test_verified_transfer_knowledge.py \
  tests/test_portable_knowledge_graph.py \
  tests/test_portrait_portable_knowledge.py \
  tests/test_closed_loop_replanning.py \
  tests/test_ctrl_world_autonomous_transfer.py \
  tests/test_experiment_engineering.py \
  tests/test_training_scale.py \
  tests/test_training_recipes.py \
  tests/test_research_proposal.py \
  tests/test_intermediate_ir.py \
  tests/test_campaign_api.py \
  tests/test_campaign_dispatcher.py \
  tests/test_verdiwm_cli.py \
  tests/test_ctrl_world_settlement_import.py \
  tests/test_campaign_daemon.py \
  tests/test_pipeline_daemon.py \
  tests/test_verdiwm_geometry.py \
  tests/test_verdiwm_public_release.py \
  tests/test_portrait_first_public_example.py \
  tests/test_acwm_public_experience_bundle.py \
  tests/test_agent_engineering_policy.py \
  tests/test_backbone_capability_matrix.py \
  tests/test_backbone_instance.py \
  tests/test_cross_backbone_experiments.py \
  tests/test_ctrl_world_instance_adapters.py \
  tests/test_ctrl_world_predictive_instance.py \
  tests/test_ctrl_world_fingerprint.py \
  tests/test_ctrl_world_predictive_campaign_runner.py \
  tests/test_ctrl_world_receipt_merge.py \
  tests/test_ctrl_world_fingerprint_settlement.py \
  tests/test_ctrl_world_fingerprint_public_bundle.py \
  tests/test_cosmos3_gpu_runtime_receipt.py \
  tests/test_cosmos3_forward_dynamics_public_bundle.py \
  tests/test_cosmos3_paired_gt.py \
  tests/test_cosmos3_fingerprint.py \
  tests/test_cosmos3_fingerprint_campaign_runner.py \
  tests/test_cosmos3_fingerprint_public_bundle.py \
  tests/test_cosmos3_probe_evolution.py \
  tests/test_cosmos3_directional_settlement.py \
  tests/test_cosmos3_directional_settlement_public_bundle.py \
  tests/test_cosmos3_split_v2_protocol.py \
  tests/test_cosmos3_r31_probe_runtime.py \
  tests/test_ctrl_world_probe_evolution.py \
  tests/test_ctrl_world_r31_probe_runtime.py \
  tests/test_probe_semantic_compile.py \
  tests/test_certificate_ablation.py \
  tests/test_cross_backbone_reuse_audit.py \
  tests/test_r31_cross_backbone_runtime_bundle.py \
  tests/test_primitive_materialization_prompt.py \
  tests/test_diagnostic_probe_materialization_prompt.py \
  tests/test_diagnostic_probe_routing_admission.py \
  tests/test_acwm_push_cube_contact_event_diagnostic_v1.py \
  tests/test_acwm_push_cube_action_binding_diagnostic_v1.py \
  tests/test_acwm_push_cube_rigid_pose_slip_diagnostic_v1.py \
  tests/test_acwm_stack_cube_support_relation_diagnostic_v1.py \
  tests/test_acwm_stack_cube_contact_instability_diagnostic_v1.py \
  tests/test_acwm_stack_cube_object_identity_diagnostic_v1.py \
  tests/test_acwm_pour_water_container_boundary_leak_diagnostic_v1.py \
  tests/test_acwm_pour_water_free_surface_diagnostic_v1.py \
  tests/test_acwm_pour_water_fluid_volume_transport_diagnostic_v1.py \
  tests/test_acwm_push_sand_granular_frontier_diagnostic_v1.py \
  tests/test_acwm_push_sand_mass_redistribution_diagnostic_v1.py \
  tests/test_acwm_push_sand_particle_boundary_diagnostic_v1.py \
  tests/test_acwm_reacher_target_conditioning_diagnostic_v1.py \
  tests/test_acwm_reacher_inverse_dynamics_confidence_diagnostic_v1.py \
  tests/test_acwm_reacher_endpoint_control_diagnostic_v1.py \
  tests/test_acwm_push_rope_topology_change_diagnostic_v1.py \
  tests/test_acwm_push_rope_endpoint_path_diagnostic_v1.py \
  tests/test_acwm_push_rope_deformable_contact_diagnostic_v1.py \
  tests/test_acwm_cloth_move_surface_fold_diagnostic_v1.py \
  tests/test_acwm_cloth_move_cloth_identity_drift_diagnostic_v1.py \
  tests/test_acwm_cloth_move_deformable_memory_diagnostic_v1.py \
  tests/test_acwm_formal_visualization.py \
  tests/test_acwm_training_seed_replication.py

uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2

uv run python scripts/export/acwm_public_experience_bundle.py validate \
  --output-root examples/acwm_experience_atlas_v1
