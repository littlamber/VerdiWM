#!/usr/bin/env bash
set -euo pipefail

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

uv build --wheel --out-dir "$build_dir"
wheel_path="$(find "$build_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$wheel_path"
python -c 'import sys, zipfile; names=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert "wmloop/__init__.py" in names; assert "configs/schemas/goal_spec.schema.json" in names; assert "configs/schemas/cross_backbone_experiment.schema.json" in names; assert "configs/schemas/experiment_stage_receipt.schema.json" in names; assert "configs/schemas/backbone_primitive_registry.schema.json" in names' "$wheel_path"

uv run python -m py_compile \
  wmloop/geometry/types.py \
  wmloop/geometry/irg.py \
  wmloop/geometry/transfer.py \
  wmloop/geometry/memory.py \
  wmloop/geometry/evolution.py \
  wmloop/experiments/spec.py \
  wmloop/experiments/lobo.py \
  wmloop/experiments/ledger.py \
  wmloop/experiments/report.py \
  wmloop/experiments/ctrl_world_receipt_merge.py \
  wmloop/experiments/ctrl_world_fingerprint_settlement.py \
  wmloop/evaluate/adapters/ctrl_world.py \
  wmloop/primitives/adapters/backbone_registry.py \
  wmloop/primitives/adapters/ctrl_world_hooks.py \
  wmloop/control/agent_engineering_policy.py \
  wmloop/control/backbone_capability_matrix.py \
  wmloop/control/cosmos3_gpu_runtime_receipt.py \
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
  scripts/run_ctrl_world_bounded_smoke.py

uv run pytest -q \
  tests/test_verdiwm_geometry.py \
  tests/test_verdiwm_public_release.py \
  tests/test_acwm_public_experience_bundle.py \
  tests/test_agent_engineering_policy.py \
  tests/test_backbone_capability_matrix.py \
  tests/test_backbone_instance.py \
  tests/test_cross_backbone_experiments.py \
  tests/test_ctrl_world_instance_adapters.py \
  tests/test_ctrl_world_fingerprint.py \
  tests/test_ctrl_world_predictive_campaign_runner.py \
  tests/test_ctrl_world_receipt_merge.py \
  tests/test_ctrl_world_fingerprint_settlement.py \
  tests/test_ctrl_world_fingerprint_public_bundle.py \
  tests/test_cosmos3_gpu_runtime_receipt.py \
  tests/test_cosmos3_forward_dynamics_public_bundle.py \
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
  tests/test_acwm_cloth_move_deformable_memory_diagnostic_v1.py

uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2

uv run python scripts/export/acwm_public_experience_bundle.py validate \
  --output-root examples/acwm_experience_atlas_v1
