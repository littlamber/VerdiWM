"""Current observations and verified memory guide bounded literature search."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.error
import urllib.parse

import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.retrieve import literature
from wmloop.retrieve.query_planning import (
    LiteratureQueryPlanError,
    build_literature_query_plan,
    write_literature_query_plan,
)
from wmloop.retrieve.query_terms import query_terms


def plan(*, intent="improve video generation temporal consistency reduce long horizon drift", signature="long_horizon_drift", family="video_diffusion", context=None):
    return build_literature_query_plan(
        explicit_query=intent,
        retrieval_context=context or {},
        probe_manifest={"model_family": family, "failure_signatures": [signature], "probe_id": "current-probe"},
    )


def memory(*, outcome="negative", paired=True, signature="long_horizon_drift"):
    query = {
        "primitive": "history_sparse_dilation", "metric_name": "psnr_db", "metric_direction": "higher",
        "protocol_sha256": "a" * 64, "context_sha256": "b" * 64, "seed": 1,
        "control_parameters": {"dilation": 1}, "candidate_parameters": {"dilation": 2},
    }
    return {"capsule": {"selected_evidence": [{
        "failure_signature": signature, "primitive": "history_sparse_dilation", "verdict": "PASS",
        "evidence": {"archive_trial_id": "prior-trial", "receipt_ref": "cas://sha256/" + "c" * 64,
                     "result_ref": "cas://sha256/" + "d" * 64, "receipt_hash": "c" * 64},
        "paired_observations": [{"query": query, "outcome": outcome}] if paired else [],
    }]}}


def test_current_observation_changes_search_despite_same_explicit_intent():
    first = plan(signature="history_drift")
    second = plan(signature="motion_collapse")
    assert first["explicit_intent"] == second["explicit_intent"]
    assert first["query"] == second["query"], "The frozen goal view must not be replaced by this round's diagnosis."
    assert second["focus_terms"] == ["motion", "collapse"]
    assert second["current_diagnostic"]["manifest_sha256"] != first["current_diagnostic"]["manifest_sha256"]
    assert 'all:"motion" AND all:"collapse"' in second["queries"][-1]["query"]
    assert second["queries"][-1]["view"] == "current_diagnostic"


def test_stale_irg_signature_does_not_override_new_observation():
    context = {"irg_guided": {"request": {"failure_signatures": ["obsolete_symptom"], "cross_domain_lenses": ["belief filtering"]}}}
    actual = plan(signature="motion_collapse", context=context)
    assert actual["signature_source"] == "current_diagnostic"
    assert "obsolete" not in actual["query"]
    assert "belief" in actual["query"]
    assert "obsolete" not in actual["queries"][-1]["query"]


def test_verified_paired_signs_are_context_not_success_keywords():
    cold = plan()
    warm = plan(context=memory())
    assert warm["query"] != cold["query"]
    assert warm["memory_context"][0]["used_as_lens"] is True
    assert warm["memory_context"][0]["paired_outcomes"][0]["outcome"] == "negative"
    assert warm["memory_context"][0]["evidence"]["archive_trial_id"] == "prior-trial"
    assert "negative" not in warm["query"] and "PASS" not in warm["query"] and "cas:" not in warm["query"]
    assert warm["authority"] == "ranking_only"
    assert warm == plan(context=memory()), "Identical evidence must produce a stable plan."
    positive = plan(context=memory(outcome="positive"))
    assert positive["query"] == warm["query"], "A sign alone does not establish a new mechanism."
    assert positive["memory_context"] != warm["memory_context"]


def test_settled_agibot_psnr_memory_is_reused_as_bounded_lens():
    context = {
        "agibot_psnr_memory": {
            "state": "matched",
            "query": {"dataset_sha256": "a" * 64, "support_sha256": "b" * 64, "evaluator_digest": "c" * 64},
            "match_count": 1,
            "matches": [{
                "trial_id": "agibot-trial-1",
                "verdict": "REJECT",
                "effect_verified": False,
                "statistics": {"mean_paired_delta_db": 1.0, "lower_bound_db": 0.2, "upper_bound_db": 1.8},
                "evidence": {
                    "receipt_ref": "cas://sha256/" + "d" * 64,
                    "verdict_ref": "cas://sha256/" + "e" * 64,
                    "measurement_ref": "cas://sha256/" + "f" * 64,
                },
                "hypothesis": "history-aware temporal consistency for video prediction",
                "method": "adaptive history conditioning",
                "report_only_metrics": {"ssim": {}, "lpips": {"state": "not_measured"}},
            }],
            "claim_boundary": "ranking only",
        }
    }
    actual = plan(context=context)
    assert actual["agibot_psnr_memory"][0]["trial_id"] == "agibot-trial-1"
    assert actual["agibot_psnr_memory"][0]["used_as_lens"] is True
    assert actual["memory_context"] == []
    assert "history" in actual["query"] or "temporal" in actual["query"]
    validate_document("literature_query_plan", actual)


@pytest.mark.parametrize("kwargs", [{"paired": False}, {"signature": "obsolete_symptom"}])
def test_plain_pass_or_stale_memory_does_not_invent_an_effect_lens(kwargs):
    actual = plan(context=memory(**kwargs))
    assert actual["query"] == plan()["query"]
    assert actual["memory_context"][0]["used_as_lens"] is False


def test_actual_wan_adapter_identifiers_do_not_become_domain_constraints():
    actual = plan(
        intent="Improve video prediction quality, temporal consistency, and reduce long-horizon error",
        family="arm_slot", signature="paired_rollout_error_high",
    )
    assert actual["domain_terms"] == ["video", "prediction"]
    assert actual["focus_terms"] == ["rollout", "error"]
    assert actual["domain_source"] == "explicit_intent"
    assert "arm" not in actual["query"] and "slot" not in actual["query"]
    assert "paired" not in actual["query"] and 'all:"high"' not in actual["query"]
    assert actual["current_diagnostic"]["model_family"] == "arm_slot"
    assert actual["current_diagnostic"]["failure_signatures"] == ["paired_rollout_error_high"]


def test_chinese_intent_retains_provenance_and_uses_canonical_english_evidence():
    actual = plan(intent="改善视频生成的时间一致性，并减少长期漂移")
    assert actual["explicit_intent"]["text"] == "改善视频生成的时间一致性，并减少长期漂移"
    assert actual["domain_source"] == "model_family"
    assert actual["domain_terms"] == ["video", "diffusion"]
    assert "改善" not in actual["query"] and 'all:"drift"' not in actual["query"]
    assert 'all:"drift"' in actual["queries"][-1]["query"]


def test_unknown_family_with_chinese_intent_explicitly_skips_and_explains(tmp_path, monkeypatch):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unknown task contacted network"))
    actual = plan(intent="改善视频生成的时间一致性", family="arm_slot")
    assert actual["state"] == "skipped" and actual["query"] is None
    assert actual["current_diagnostic"]["model_family"] == "arm_slot"
    assert actual["current_diagnostic"]["failure_signatures"] == ["long_horizon_drift"]
    assert actual["explicit_intent"]["text"] == "改善视频生成的时间一致性"
    result = literature.run_literature_retrieval(query=None, query_plan=actual, output_root=tmp_path)
    assert result["state"] == "skipped"
    assert result["reason"] == "LITERATURE_TASK_TERMS_UNAVAILABLE"
    assert "video prediction" in result["suggested_action"]
    assert result["delivery"]["fresh_request"] is False
    from wmloop.execute.pipeline.retrieval import _runtime_retrieval_manifest
    projected = _runtime_retrieval_manifest({"capsule": {}, "literature_status": {
        key: actual[key] for key in ("state", "reason", "suggested_action")}})
    assert projected["literature_status"]["reason"] == result["reason"]


def test_numeric_domain_words_preserve_discovery_and_cold_start_compatibility():
    assert query_terms("3D prediction and 2D vision 123", maximum=10) == ("3d", "prediction", "2d", "vision")
    actual = plan(intent=None, family="3d_diffusion")
    assert actual["state"] == "planned" and actual["domain_terms"] == ["3d", "diffusion"]
    assert 'all:"3d"' in actual["query"]
    versioned = plan(intent=None, family="wan22_video_diffusion")
    assert versioned["domain_terms"] == ["video", "diffusion"]
    assert "wan22" not in versioned["query"]


def test_optimizer_oom_diagnosis_does_not_replace_goal_view_or_add_threshold_units():
    intent = "Improve PSNR by more than 2 dB for Agibot robot action-conditioned video prediction"
    oom = plan(intent=intent, family="arm_slot", signature="optimizer_state_oom")
    other_diagnostic = plan(intent=intent, family="arm_slot", signature="gpu_memory_exhausted")
    terms = query_terms(intent, maximum=20, intent=True)

    assert "more" not in terms and "than" not in terms and "least" not in terms and "db" not in terms
    assert oom["domain_terms"] == ["robot", "video", "prediction"]
    assert all(f'all:"{term}"' in oom["query"] for term in ("robot", "video", "prediction"))
    assert 'all:"action-conditioned"' in oom["query"]
    assert 'all:"psnr"' in oom["query"]
    assert all(f'all:"{term}"' not in oom["query"] for term in ("more", "than", "least", "db", "agibot"))
    assert "oom" not in oom["query"] and "optimizer" not in oom["query"]
    assert oom["query"] == other_diagnostic["query"]
    assert 'all:"optimizer" AND all:"state" AND all:"oom"' in oom["queries"][-1]["query"]
    assert "memory" in other_diagnostic["queries"][-1]["query"]


def test_query_is_bounded_and_current_observation_survives_long_intent():
    actual = plan(intent="video prediction " + "irrelevant " * 10000, signature="motion_collapse")
    assert actual["explicit_intent"]["truncated"] is True
    assert len(actual["explicit_intent"]["text"]) == 4096
    assert all(len(row["query"]) <= 1024 for row in actual["queries"])
    assert 'all:"motion"' not in actual["query"]
    assert 'all:"motion" AND all:"collapse"' in actual["queries"][-1]["query"]


def test_plan_persistence_is_immutable_locked_and_schema_validated(tmp_path):
    first = plan(signature="history_drift")
    second = plan(signature="motion_collapse")
    path = tmp_path / "plan.json"
    def write(document):
        try:
            write_literature_query_plan(path, document)
            return True
        except LiteratureQueryPlanError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [first, second]))
    assert sorted(results) == [False, True]
    saved = json.loads(path.read_bytes())
    assert saved in [first, second]
    validate_document("literature_query_plan", saved)
    assert write_literature_query_plan(path, saved) == path
    with pytest.raises(ContractValidationError):
        write_literature_query_plan(tmp_path / "bad.json", {**saved, "verdict": "ACCEPT"})
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LiteratureQueryPlanError, match="OUTPUT_INVALID"):
        write_literature_query_plan(linked / "other.json", saved)


ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2601.12345</id><title>Video prediction with reliable history</title>
<summary>A data-only candidate for reducing temporal drift.</summary>
<published>2026-01-01</published><link title="pdf" href="https://arxiv.org/pdf/2601.12345"/>
</entry></feed>'''
EMPTY = b'<feed xmlns="http://www.w3.org/2005/Atom"/>'


class Response:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def getcode(self):
        return 200
    def read(self, limit):
        return self.payload[:limit]


def test_structured_query_and_network_trace_match_actual_request(tmp_path, monkeypatch):
    observed = []
    def fetch(request, **kwargs):
        observed.append(request.full_url)
        return Response(ATOM)
    monkeypatch.setattr(literature.urllib.request, "urlopen", fetch)
    actual = plan()
    result = literature.run_literature_retrieval(query=actual["query"], query_plan=actual, output_root=tmp_path)
    assert result["state"] == "network" and result["record_count"] == 1
    assert urllib.parse.parse_qs(urllib.parse.urlparse(observed[0]).query)["search_query"] == [actual["query"]]
    trace = json.loads((tmp_path / "search-trace.json").read_bytes())
    validate_document("literature_search_trace", trace)
    assert trace["attempts"][0]["http_status"] == 200 and trace["attempts"][0]["error_code"] is None
    assert trace["attempts"][0]["response_sha256"]
    assert trace["attempts"][0]["started_at"] <= trace["attempts"][0]["completed_at"]


def test_empty_strict_lens_falls_back_with_truthful_staging_provenance(tmp_path, monkeypatch):
    responses = iter([EMPTY, ATOM])
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: Response(next(responses)))
    actual = plan(context=memory())
    result = literature.run_literature_retrieval(query=actual["query"], query_plan=actual, output_root=tmp_path)
    fallback = actual["queries"][-1]["query"]
    assert len(result["search_trace"]["attempts"]) == 2
    assert result["search_trace"]["attempts"][0]["view"] == "evidence_lens"
    assert result["search_trace"]["attempts"][1]["view"] == "current_diagnostic"
    assert result["search_trace"]["attempts"][1]["query"] == fallback
    assert 'all:"long" AND all:"horizon" AND all:"drift"' in fallback
    assert result["query"] == actual["query"] and result["effective_query"] == fallback
    assert json.loads((tmp_path / "records.json").read_bytes())["query"] == fallback
    candidate = json.loads(Path(result["rows"][0]["path"]).read_bytes())
    assert candidate["proposed_manifest"]["query"] == fallback


@pytest.mark.parametrize("failure,code", [
    (urllib.error.HTTPError("https://example.invalid", 503, "down", {}, None), "LITERATURE_HTTP_503"),
    (urllib.error.URLError(TimeoutError()), "LITERATURE_TIMEOUT"),
    (Response(b"<broken"), "LITERATURE_ATOM_INVALID"),
    (Response(b"<html/>"), "LITERATURE_ATOM_FEED_INVALID"),
    (Response(b"x" * 2_000_001), "LITERATURE_RESPONSE_TOO_LARGE"),
    (Response(b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors</id></entry></feed>'), "LITERATURE_API_ERROR"),
])
def test_provider_failures_keep_root_cause_and_do_not_trigger_query_fallback(tmp_path, monkeypatch, failure, code):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(literature.urllib.request, "urlopen", fetch)
    actual = plan(context=memory())
    result = literature.run_literature_retrieval(query=actual["query"], query_plan=actual, output_root=tmp_path)
    assert result["state"] == "offline" and len(calls) == 1
    assert result["search_trace"]["attempts"][0]["error_code"] == code


def test_empty_valid_cache_is_distinguished_from_offline(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"query": "video prediction", "records": []}))
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("unavailable")))
    records, state = literature.search_arxiv("video prediction", cache_path=cache)
    assert records == () and state == "cached"


@pytest.mark.parametrize("payload", [[], {"query": "video prediction", "records": [1]},
                                    {"query": "video prediction", "records": [{"arxiv_id": 123}]}])
def test_malformed_cache_does_not_crash_or_supply_records(tmp_path, monkeypatch, payload):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(payload))
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("unavailable")))
    assert literature.search_arxiv("video prediction", cache_path=cache) == ((), "offline")


def test_empty_external_query_plan_is_rejected_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Invalid plan contacted network"))
    actual = {**plan(), "queries": []}
    with pytest.raises(literature.LiteratureRetrievalError, match="QUERY_PLAN_EMPTY"):
        literature.run_literature_retrieval(query=actual["query"], query_plan=actual, output_root=tmp_path)
    assert not (tmp_path / "input-binding.json").exists()


def test_partial_staging_recovers_identical_candidates_without_new_network(tmp_path, monkeypatch):
    # Explicit two-entry feed keeps the provider result stable across recovery.
    atom = ATOM.replace(b"</feed>", ATOM[ATOM.index(b"<entry>"):ATOM.index(b"</feed>")].replace(b"2601.12345", b"2601.12346") + b"</feed>")
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: Response(atom))
    original_stage = literature.stage_literature_candidate
    def interrupt(candidate, **kwargs):
        path = original_stage(candidate, **kwargs)
        raise RuntimeError("interrupt after candidate committed")
    with monkeypatch.context() as fault:
        fault.setattr(literature, "stage_literature_candidate", interrupt)
        with pytest.raises(RuntimeError, match="candidate committed"):
            literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    assert len(list((tmp_path / "candidates").glob("*.json"))) == 1
    assert (tmp_path / "search-result.json").is_file() and not (tmp_path / "manifest.json").exists()
    original_candidate = next((tmp_path / "candidates").glob("*.json"))
    original_bytes = original_candidate.read_bytes()
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Recovery repeated network"))
    recovered = literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    assert recovered["record_count"] == recovered["staged_count"] == 2
    assert all(row["state"] == "staged" for row in recovered["rows"])
    assert original_candidate.read_bytes() == original_bytes
    assert recovered["delivery"] == {"mode": "recovery", "fresh_request": False}


def test_partial_input_binding_rejects_changed_goal_and_recovers_cache_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: Response(ATOM))
    original_write = literature._write_json
    def interrupt(path, payload):
        original_write(path, payload)
        if path.name == "cache.json":
            raise RuntimeError("interrupt after cache committed")
    with monkeypatch.context() as fault:
        fault.setattr(literature, "_write_json", interrupt)
        with pytest.raises(RuntimeError, match="cache committed"):
            literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    with pytest.raises(literature.LiteratureRetrievalError, match="INPUT_MISMATCH"):
        literature.run_literature_retrieval(query="different goal", output_root=tmp_path)
    assert literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)["staged_count"] == 1


def test_simultaneous_identical_retrieval_uses_one_network_transaction(tmp_path, monkeypatch):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        return Response(ATOM)
    monkeypatch.setattr(literature.urllib.request, "urlopen", fetch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: literature.run_literature_retrieval(query="video prediction", output_root=tmp_path), range(2)))
    assert len(calls) == 1 and [row["staged_count"] for row in results] == [1, 1]
    assert {row["delivery"]["mode"] for row in results} == {"fresh", "resume"}


def test_output_write_failure_is_not_misreported_as_network_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: Response(ATOM))
    monkeypatch.setattr(literature, "_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        literature.search_arxiv("video prediction", cache_path=tmp_path / "cache.json")


def test_resume_preserves_original_evidence_and_reports_no_new_network(tmp_path, monkeypatch):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: Response(ATOM))
    first = literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Resume contacted network"))
    resumed = literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    assert resumed["delivery"] == {"mode": "resume", "fresh_request": False}
    assert resumed["search_trace"] == first["search_trace"]
    assert all(path.read_bytes() == content for path, content in files.items())
    # A pre-trace manifest remains readable and does not gain invented proof.
    old = {key: value for key, value in first.items() if key not in {"delivery", "search_trace", "search_trace_path", "effective_query"}}
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    legacy = literature.run_literature_retrieval(query="video prediction", output_root=tmp_path)
    assert "search_trace" not in legacy and legacy["delivery"]["fresh_request"] is False


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, 0, 61])
def test_invalid_timeouts_fail_before_contacting_provider(monkeypatch, timeout):
    monkeypatch.setattr(literature.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Invalid input contacted network"))
    with pytest.raises(literature.LiteratureRetrievalError, match="TIMEOUT_INVALID"):
        literature.search_arxiv("video prediction", timeout_seconds=timeout)
