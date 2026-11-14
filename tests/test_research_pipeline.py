"""CPU integration of research state with pipeline orchestration, not model effects."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from wmloop.execute import autonomous_pipeline as pipeline


def setup_pipeline(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    evaluator = tmp_path/'evaluator.json'
    template = tmp_path/'template.json'
    template.write_text(json.dumps({'objective':'Improve held-out rollout consistency', 'falsification_criterion':'Reject if protected action fidelity degrades'}))
    evaluator.write_text(json.dumps({'scheduler_template':str(template), 'metrics':['rollout_error']}))
    monkeypatch.setattr(pipeline, '_input_document', lambda *args, **kwargs: {'test_binding': 'fixed'})
    monkeypatch.setattr(pipeline, 'run_onboarding', lambda options: {'state':'ready_for_conformance_smoke'})
    monkeypatch.setattr(pipeline, 'run_conformance', lambda options: {'verdict':'PASS'})
    output = tmp_path/'output'
    def compile_queue(**kwargs):
        path = output/'compiled'/'queue'/'queue.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'selected':[{'candidate_id':'repair-a'}]}))
        return {'queue_path':str(path)}
    monkeypatch.setattr(pipeline, 'compile_and_plan', compile_queue)
    def execute(**kwargs):
        execution = {'candidate_states':{'repair-a':'completed'}}
        (output/'compiled'/'queue'/'execution.json').write_text(json.dumps(execution))
        return execution
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    options = pipeline.AutonomousPipelineOptions(repo_root=source, output_root=output, evaluator_contract=evaluator, budget_total_gpu_hours=1)
    return options, execute


def state(options):
    return json.loads((options.output_root/'research'/'research-state.json').read_text())


def test_pipeline_records_phases_and_resumes_terminal_without_rerunning(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    result = pipeline.run_autonomous_pipeline(options)
    assert result['state'] == 'settled'
    current = state(options)
    assert current['phase'] == 'settled'
    assert current['goal']['description'] == 'Improve held-out rollout consistency'
    assert current['candidate_actions'] == [{'candidate_id':'repair-a'}]
    assert current['budget']['gpu_hours_remaining'] is None
    phases = {json.loads(path.read_text())['phase'] for path in (options.output_root/'research'/'research-history').glob('*.json')}
    assert phases == {'observe','hypothesize','select','implement','verify','remember'}
    def recover(**kwargs):
        assert kwargs['recover_only'] is True
        return json.loads((options.output_root/'compiled'/'queue'/'execution.json').read_bytes())
    monkeypatch.setattr(pipeline, 'run_selected_queue', recover)
    assert pipeline.run_autonomous_pipeline(options) == result
    assert state(options) == current


def test_interruption_and_resume_preserve_prior_evidence(tmp_path, monkeypatch):
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        raise RuntimeError('executor interrupted')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='executor interrupted'):
        pipeline.run_autonomous_pipeline(options)
    previous = state(options)
    assert previous['phase'] == 'blocked'
    assert 'execution' in previous['last_transition']['reason']
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'settled'
    assert state(options)['revision'] > previous['revision']
    assert (options.output_root/'research'/'research-history'/f"{previous['state_id']}.json").is_file()


def test_onboarding_blocker_is_projected_without_experiment_execution(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, 'run_onboarding', lambda options: {'state':'blocked'})
    monkeypatch.setattr(pipeline, 'run_selected_queue', lambda **kwargs: pytest.fail('Blocked onboarding ran experiment'))
    assert pipeline.run_autonomous_pipeline(options)['blocked_stage'] == 'onboarding'
    assert state(options)['phase'] == 'blocked'


def test_negative_outcomes_are_snapshotted_before_retry(tmp_path, monkeypatch):
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    def negative(**kwargs):
        result = {'candidate_states':{'repair-a':'blocked'}, 'results':{'repair-a:gate':{'verdict':'FAIL'}}}
        (options.output_root/'compiled'/'queue'/'execution.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(pipeline, 'run_selected_queue', negative)
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'blocked'
    previous = state(options)
    refs = [Path(ref) for ref in previous['evidence_refs'] if 'execution-snapshots' in ref]
    assert len(refs) == 1
    snapshot = refs[0].read_bytes()
    assert json.loads(snapshot)['results']['repair-a:gate']['verdict'] == 'FAIL'
    monkeypatch.setattr(pipeline, 'run_selected_queue', execute)
    pipeline.run_autonomous_pipeline(options)
    assert refs[0].read_bytes() == snapshot
    assert str(refs[0]) in state(options)['evidence_refs']


def test_partial_execution_snapshot_survives_an_exception(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        (options.output_root/'compiled'/'queue'/'execution.json').write_text(json.dumps({'results':{'repair-a:screen':{'verdict':'PASS'}}}))
        raise RuntimeError('interrupted after first stage')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='interrupted after first stage'):
        pipeline.run_autonomous_pipeline(options)
    refs = [Path(ref) for ref in state(options)['evidence_refs'] if 'execution-snapshots' in ref]
    assert len(refs) == 1
    assert json.loads(refs[0].read_text())['results']['repair-a:screen']['verdict'] == 'PASS'


def test_same_output_cannot_run_concurrently(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    options, execute = setup_pipeline(tmp_path, monkeypatch)
    entered, release = Event(), Event()
    def run(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return execute(**kwargs)
    monkeypatch.setattr(pipeline, 'run_selected_queue', run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(pipeline.run_autonomous_pipeline, options)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(pipeline.AutonomousPipelineError, match='ALREADY_RUNNING'):
                pipeline.run_autonomous_pipeline(options)
        finally:
            release.set()
        assert first.result(timeout=5)['state'] == 'settled'
    # Lock is released at terminal completion, with no manual cleanup.
    assert pipeline.run_autonomous_pipeline(options)['state'] == 'settled'


def test_scheduler_template_change_rejected_even_when_goal_is_unchanged(tmp_path, monkeypatch):
    real_input_document = pipeline._input_document
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, '_input_document', real_input_document)
    monkeypatch.setattr(pipeline, 'compute_source_revision', lambda *args, **kwargs: 'fixed-revision')
    monkeypatch.setattr(pipeline, 'compute_source_tree_revision', lambda *args, **kwargs: 'fixed-tree')
    pipeline.run_autonomous_pipeline(options)
    path = tmp_path/'template.json'
    value = json.loads(path.read_text())
    value['candidates'] = [{'candidate_id':'changed-method'}]
    path.write_text(json.dumps(value))
    with pytest.raises(pipeline.AutonomousPipelineError, match='INPUT_MISMATCH'):
        pipeline.run_autonomous_pipeline(options)


def test_runtime_timeout_reaches_onboarding_and_is_bound_on_resume(tmp_path, monkeypatch):
    real_input_document = pipeline._input_document
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    options = replace(options, runtime_timeout_seconds=45)
    monkeypatch.setattr(pipeline, '_input_document', real_input_document)
    monkeypatch.setattr(pipeline, 'compute_source_revision', lambda *args, **kwargs: 'fixed-revision')
    monkeypatch.setattr(pipeline, 'compute_source_tree_revision', lambda *args, **kwargs: 'fixed-tree')
    observed = []
    monkeypatch.setattr(pipeline, 'run_onboarding', lambda value: observed.append(value.runtime_timeout_seconds) or {'state': 'ready_for_conformance_smoke'})
    pipeline.run_autonomous_pipeline(options)
    assert observed == [45]
    lock = json.loads((options.output_root/'pipeline-input.lock.json').read_bytes())
    assert lock['input']['runtime_timeout_seconds'] == 45
    with pytest.raises(pipeline.AutonomousPipelineError, match='INPUT_MISMATCH'):
        pipeline.run_autonomous_pipeline(replace(options, runtime_timeout_seconds=60))


def test_pipeline_cli_exposes_runtime_timeout_in_stdout_json_contract(tmp_path, monkeypatch, capsys):
    options = []
    monkeypatch.setattr(pipeline, 'run_autonomous_pipeline', lambda value: options.append(value) or {'verdict': 'PASS'})
    assert pipeline.main([str(tmp_path), '--output-root', str(tmp_path/'out'),
                          '--evaluator-contract', str(tmp_path/'evaluator.json'),
                          '--runtime-timeout-seconds', '60']) == 0
    assert options[0].runtime_timeout_seconds == 60
    assert json.loads(capsys.readouterr().out) == {'verdict': 'PASS'}


def test_corrupt_partial_receipt_does_not_mask_executor_failure(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    def fail(**kwargs):
        (options.output_root/'compiled'/'queue'/'execution.json').write_bytes(b'{incomplete')
        raise RuntimeError('original executor failure')
    monkeypatch.setattr(pipeline, 'run_selected_queue', fail)
    with pytest.raises(RuntimeError, match='original executor failure'):
        pipeline.run_autonomous_pipeline(options)
    refs = [Path(ref) for ref in state(options)['evidence_refs'] if 'execution-snapshots' in ref]
    assert refs[0].read_bytes() == b'{incomplete'
    assert state(options)['phase'] == 'blocked'


def method_catalog(candidate_id):
    return {
        "schema_version": 1, "artifact_type": "verdiwm-method-candidate-catalog",
        "catalog_id": "pipeline-catalog-fixture", "model_family": "cpu_fixture",
        "candidates": [{
            "candidate_id": candidate_id, "primitive_reference": None,
            "source": "local_cpu_fixture",
            "mechanism_hypothesis": "Exercise pipeline candidate selection without model claims.",
            "required_hooks": ["H3"], "failure_signatures": ["fixture_symptom"],
            "applicability_conditions": ["CPU fixture only"],
            "failure_boundaries": ["No model or GPU claim"],
            "estimated_gpu_hours": 0.01, "historical_candidate_ids": [],
            "required_files": [], "candidate_template": {"candidate_id": candidate_id},
        }],
        "capability_gaps": [],
        "claim_boundary": "Synthetic catalog for pipeline orchestration only.",
    }


@pytest.mark.parametrize("with_user_catalog", [False, True])
@pytest.mark.parametrize("with_blocked_work_order", [False, True])
def test_no_ready_literature_candidates_preserve_existing_execution_inputs(
    tmp_path, monkeypatch, with_user_catalog, with_blocked_work_order,
):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    options = replace(options, literature_query="CPU fixture research")
    original_catalog = None
    if with_user_catalog:
        original_catalog = tmp_path / "user-catalog.json"
        original_catalog.write_text(json.dumps(method_catalog("user-candidate")))
        options = replace(options, candidate_catalog=original_catalog)
    source_bytes = (tmp_path / "template.json").read_bytes()
    observed_catalogs = []
    original_compile = pipeline.compile_and_plan

    def compile_candidates(**kwargs):
        observed_catalogs.append(kwargs["candidate_catalog"])
        return original_compile(**kwargs)

    monkeypatch.setattr(pipeline, "compile_and_plan", compile_candidates)
    monkeypatch.setattr(pipeline, "run_literature_retrieval", lambda **_: {
        "state": "offline", "record_count": 0, "rows": [],
    })

    def stage_methods(**kwargs):
        destination = kwargs["output_root"]
        destination.mkdir(parents=True)
        orders = {}
        if with_blocked_work_order:
            order = tmp_path / "blocked-work-order.json"
            order.write_text(json.dumps({"literature_method": {"candidate_id": "unimplemented"}}))
            orders["unimplemented"] = str(order)
        manifest = {
            "artifact_type": "wmloop-literature-method-staging-manifest",
            "work_order_paths": orders,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(pipeline, "run_literature_method_staging", stage_methods)
    monkeypatch.setattr(pipeline, "run_literature_method_prompt_batch", lambda **_: {"state": "ready"})
    # The real materializer emits an empty science catalog, even when it has
    # retained blocked work orders. Neither case grants execution authority.
    result = pipeline.run_autonomous_pipeline(options)
    assert result["state"] == "settled"
    assert observed_catalogs == [original_catalog]
    assert (tmp_path / "template.json").read_bytes() == source_bytes
    assert result["literature"]["state"] == "offline"
    materialization = result["literature_method_materialization"]
    assert materialization["ready_count"] == 0
    assert materialization["blocked_count"] == int(with_blocked_work_order)
    assert materialization["candidate_selection"] == {
        "state": "degraded", "reason": "LITERATURE_NO_EXECUTABLE_CANDIDATES",
        "retained_source": "user_catalog" if with_user_catalog else "evaluator_template",
    }
    catalog = json.loads(Path(materialization["candidate_catalog_path"]).read_bytes())
    assert catalog["candidates"] == []
    assert len(catalog["capability_gaps"]) == int(with_blocked_work_order)


def test_ready_literature_catalog_still_reaches_candidate_compilation(tmp_path, monkeypatch):
    options, _ = setup_pipeline(tmp_path, monkeypatch)
    options = replace(options, literature_query="CPU fixture research")
    generated = tmp_path / "generated-catalog.json"
    generated.write_text(json.dumps(method_catalog("new-candidate")))
    observed_catalogs = []
    original_compile = pipeline.compile_and_plan

    def compile_candidates(**kwargs):
        observed_catalogs.append(kwargs["candidate_catalog"])
        return original_compile(**kwargs)

    monkeypatch.setattr(pipeline, "compile_and_plan", compile_candidates)
    monkeypatch.setattr(pipeline, "run_literature_retrieval", lambda **_: {"state": "ready"})
    monkeypatch.setattr(pipeline, "run_literature_method_staging", lambda **_: {"state": "ready"})
    monkeypatch.setattr(pipeline, "run_literature_method_prompt_batch", lambda **_: {"state": "ready"})
    monkeypatch.setattr(pipeline, "run_literature_method_materialization", lambda **_: {
        "state": "ready", "ready_count": 1, "candidate_catalog_path": str(generated),
    })
    result = pipeline.run_autonomous_pipeline(options)
    assert result["state"] == "settled"
    assert observed_catalogs == [generated]
    assert "candidate_selection" not in result["literature_method_materialization"]
