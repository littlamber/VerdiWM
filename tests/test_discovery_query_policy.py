from __future__ import annotations

import json
from pathlib import Path

from wmloop.retrieve.evidence_gate import build_retrieval_evidence_gate
from wmloop.retrieve.literature import (
    LiteratureRecord,
    run_literature_retrieval_batch,
)
from wmloop.retrieve.mechanism_discovery import DiscoveryRequest, build_multiview_queries
from wmloop.retrieve.query_policy import load_discovery_query_policy


ROOT = Path(__file__).resolve().parents[1]


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        symptom_description="long horizon action drift",
        failure_signatures=("temporal_drift",),
        target_metrics=("action_following",),
        protected_metrics=("subject_consistency",),
        available_hooks=("latent_history",),
        model_family="fixture",
    )


def test_query_policy_is_versioned_and_data_driven() -> None:
    policy = load_discovery_query_policy(root=ROOT)
    queries = build_multiview_queries(_request(), policy=policy)

    assert policy.policy_id == "world-model-cross-domain-discovery-v1"
    assert len(policy.sha256) == 64
    assert queries[0]["view"] == "diagnostic_symptom"
    assert "temporal" in policy.domain_tags(("temporal rollout drift",))
    assert policy.lenses_for(("temporal",))


def test_literature_batch_deduplicates_and_caps_results(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_search(query, *, max_results, timeout_seconds, cache_path):
        del timeout_seconds, cache_path
        calls.append((query, max_results))
        shared = LiteratureRecord("2401.00001v1", "Shared", "Shared abstract", "https://arxiv.org/pdf/2401.00001", "2024")
        unique = LiteratureRecord(
            f"2401.{len(calls) + 1:05d}v1",
            f"Unique {len(calls)}",
            "Unique abstract",
            f"https://arxiv.org/pdf/2401.{len(calls) + 1:05d}",
            "2024",
        )
        return (shared, unique), "network"

    monkeypatch.setattr("wmloop.retrieve.literature.search_arxiv", fake_search)
    manifest = run_literature_retrieval_batch(
        queries=("query one", "query two", "query three"),
        output_root=tmp_path / "literature",
        max_results=3,
    )

    assert manifest["state"] == "network"
    assert manifest["record_count"] == 3
    assert manifest["staged_count"] == 3
    assert len(calls) == 3
    records = json.loads((tmp_path / "literature" / "records.json").read_text())
    assert len(records["records"]) == 3


def test_network_research_gate_fails_closed_without_live_records() -> None:
    gate = build_retrieval_evidence_gate(
        [
            {
                "artifact_type": "verdiwm-literature-retrieval-manifest",
                "state": "offline",
                "record_count": 0,
            }
        ],
        require_network=True,
    )

    assert gate["state"] == "blocked"
    assert set(gate["blockers"]) == {
        "external_records_required",
        "live_network_retrieval_required",
    }
