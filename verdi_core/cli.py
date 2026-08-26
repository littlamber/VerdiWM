"""CLI for the clean release."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time

from adapters.fixture_world import FixtureWorldAdapter
from adapters.fixture_research import FixtureResearchAI
from adapters.fixture_research import FixtureEngineeringAI

from .loop import run_loop
from .ingestion import DocumentIngestor
from .storage import SQLiteState
from .research import ResearchSystem
from .runtime import RuntimeBindings
from .providers import OpenAICompatibleProvider
from .retrieval import OfflineRetriever, OnlineRetriever
from .search_sources import ArxivBackend, CrossrefBackend, FanoutBackend, GitHubBackend, SemanticScholarBackend
from .campaign import CampaignPolicy, CampaignSupervisor
from .autonomous import autonomous_campaign
from .knowledge_graph import export_bundle, import_settlement_entries
from .transfer import rank_transfer_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verdi")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    demo = commands.add_parser("demo")
    demo.add_argument("--state-root", type=Path, required=True)
    graph = commands.add_parser("graph")
    graph.add_argument("--state-root", type=Path, required=True)
    graph.add_argument("--output-root", type=Path)
    graph.add_argument("--portable", action="store_true")
    graph_import = commands.add_parser("graph-import")
    graph_import.add_argument("--state-root", type=Path, required=True)
    graph_import.add_argument("--input", type=Path, required=True)
    graph_import.add_argument("--model-id", default="ctrl-world")
    graph_import.add_argument("--campaign-id", default="ctrl-world")
    bundle = commands.add_parser("graph-bundle")
    bundle.add_argument("--state-root", type=Path, required=True)
    bundle.add_argument("--output-root", type=Path, required=True)
    bundle.add_argument("--portable", action="store_true", default=True)
    transfer = commands.add_parser("transfer")
    transfer.add_argument("--state-root", type=Path, required=True)
    transfer.add_argument("--target-model-id", required=True)
    transfer.add_argument("--architecture", action="append", default=[])
    transfer.add_argument("--diagnostic", action="append", default=[])
    transfer.add_argument("--capability", action="append", default=[])
    transfer.add_argument("--hook", action="append", default=[])
    init = commands.add_parser("init")
    init.add_argument("--path", type=Path, required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--state-root", type=Path, required=True)
    ingest.add_argument("--path", type=Path, required=True)
    cycle = commands.add_parser("cycle")
    cycle.add_argument("--state-root", type=Path, required=True)
    cycle.add_argument("--objective", required=True)
    cycle.add_argument("--offline", action="store_true")
    knowledge = commands.add_parser("knowledge")
    knowledge.add_argument("--state-root", type=Path, required=True)
    knowledge.add_argument("--outcome")
    export = commands.add_parser("export")
    export.add_argument("--state-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--state-root", type=Path, required=True)
    runs = commands.add_parser("runs")
    runs.add_argument("--state-root", type=Path, required=True)
    campaign = commands.add_parser("campaign")
    campaign_commands = campaign.add_subparsers(dest="campaign_command", required=True)
    campaign_run = campaign_commands.add_parser("run")
    campaign_run.add_argument("--state-root", type=Path, required=True)
    campaign_run.add_argument("--run-id", required=True)
    campaign_run.add_argument("--model-id", required=True)
    campaign_run.add_argument("--objective", required=True)
    campaign_run.add_argument("--ideas", type=Path, required=True)
    campaign_run.add_argument("--runner", required=True, help="Python callable module:function(idea, stage, context)")
    campaign_policy = campaign_run.add_mutually_exclusive_group()
    campaign_policy.add_argument("--stop-on-first-positive", action="store_true", default=None, help="stop at the first replicated held-out positive (default unless --settle-count is used)")
    campaign_policy.add_argument("--continue-after-positive", action="store_true", default=None, help="keep testing ideas after a positive result")
    campaign_run.add_argument("--settle-count", type=int, help="stop after this many ideas have settled into the knowledge graph")
    campaign_run.add_argument("--watch", action="store_true")
    campaign_run.add_argument("--poll-seconds", type=float, default=30.0)
    campaign_auto = campaign_commands.add_parser("autonomous-run", help="run a campaign with AI-managed isolated repair")
    campaign_auto.add_argument("--state-root", type=Path, required=True)
    campaign_auto.add_argument("--run-id", required=True)
    campaign_auto.add_argument("--model-id", required=True)
    campaign_auto.add_argument("--objective", required=True)
    campaign_auto.add_argument("--ideas", type=Path, required=True)
    campaign_auto.add_argument("--runner", required=True, help="Python callable module:function(idea, stage, context)")
    campaign_auto.add_argument("--replanner", help="optional Python callable module:function(campaign, context)")
    campaign_auto.add_argument("--worktree-root", type=Path, required=True)
    campaign_auto.add_argument("--output-root", type=Path, required=True)
    campaign_auto.add_argument("--repository", type=Path, help="source git checkout; repairs use detached worktrees")
    campaign_auto.add_argument("--readable-root", type=Path, action="append", default=[])
    campaign_auto.add_argument("--offline", action="store_true", help="use deterministic fixture engineering AI")
    campaign_auto.add_argument("--settle-count", type=int)
    campaign_auto.add_argument("--continue-after-positive", action="store_true")
    campaign_auto.add_argument("--watch", action="store_true")
    campaign_auto.add_argument("--poll-seconds", type=float, default=30.0)
    campaign_resume = campaign_commands.add_parser("resume")
    campaign_resume.add_argument("--state-root", type=Path, required=True)
    campaign_resume.add_argument("--run-id", required=True)
    campaign_resume.add_argument("--model-id", required=True)
    campaign_resume.add_argument("--runner", required=True)
    campaign_resume.add_argument("--watch", action="store_true")
    campaign_resume.add_argument("--poll-seconds", type=float, default=30.0)
    campaign_status = campaign_commands.add_parser("status")
    campaign_status.add_argument("--state-root", type=Path, required=True)
    campaign_status.add_argument("--run-id", required=True)
    campaign_release = campaign_commands.add_parser("release-human")
    campaign_release.add_argument("--state-root", type=Path, required=True)
    campaign_release.add_argument("--run-id", required=True)
    campaign_release.add_argument("--idea-id", required=True)
    campaign_release.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        provider = OpenAICompatibleProvider.from_env()
        print(json.dumps({
            "state": "ready",
            "kernel": "model-agnostic",
            "adapter_contract": "v1",
            "ai": {
                "configured": provider is not None,
                "provider": provider.provider_id if provider else None,
                "model": provider.model if provider else None,
                "reasoning_effort": provider.reasoning_effort if provider else None,
            },
        }, sort_keys=True))
        return 0
    if args.command == "demo":
        print(json.dumps(run_loop(FixtureWorldAdapter(), state_root=args.state_root), indent=2, sort_keys=True))
        return 0
    if args.command == "init":
        args.path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"project_id": "verdi-project", "objective": "improve the declared metric", "constraints": ["safety"], "budget": 1.0, "model": {"adapter": "fixture"}, "data": {"root": str(args.path.parent / "data"), "heldout_split": "heldout", "signals": ["quality", "safety"]}}
        args.path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"state": "initialized", "manifest": str(args.path)}, sort_keys=True))
        return 0
    if args.command == "ingest":
        ingestor = DocumentIngestor(args.state_root / "retrieval" / "inbox")
        documents = [ingestor.ingest_file(args.path)] if args.path.is_file() else ingestor.scan_inbox()
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        for document in documents:
            state.put_document(document.document_id, document.__dict__)
        print(json.dumps({"state": "ingested", "count": len(documents), "statuses": [doc.status for doc in documents]}, sort_keys=True))
        return 0
    if args.command == "cycle":
        ai = FixtureResearchAI() if args.offline else OpenAICompatibleProvider.from_env()
        if ai is None:
            print(json.dumps({"state": "abstain", "reason": "cycle requires OpenAI-compatible AI; use --offline for the deterministic fixture"}, sort_keys=True))
            return 0
        retriever = OfflineRetriever([("Fixture paper", "A bounded repair reduces long-horizon drift.")], state_root=args.state_root) if args.offline else OnlineRetriever(FanoutBackend((ArxivBackend(), CrossrefBackend(), SemanticScholarBackend(), GitHubBackend())), state_root=args.state_root)
        system = ResearchSystem(RuntimeBindings(FixtureWorldAdapter(), ai=ai, state_root=args.state_root, options={"budget": 1.0}), retriever=retriever)
        print(json.dumps(system.run_cycle(objective=args.objective, constraints=["safety"]), indent=2, sort_keys=True))
        return 0
    if args.command == "knowledge":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        print(json.dumps(state.search_evidence(outcome=args.outcome), indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(state.export_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"state": "exported", "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "runs":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        print(json.dumps(state.list_rows("runs"), indent=2, sort_keys=True))
        return 0
    if args.command == "resume":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        print(json.dumps({"state": "ready", "queued": [row["experiment_id"] for row in state.list_rows("experiments") if row["state"] in {"queued", "runtime_failed"}]}, sort_keys=True))
        return 0
    if args.command == "campaign":
        if args.campaign_command == "status":
            state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
            rows = [row for row in state.list_rows("runs", limit=100000) if row["run_id"] == args.run_id]
            if not rows:
                print(json.dumps({"state": "missing", "run_id": args.run_id}, sort_keys=True))
                return 1
            payload = json.loads(rows[0]["payload_json"])
            print(json.dumps({"run_id": args.run_id, "state": payload["state"], "ideas": {idea_id: {"state": item["state"], "stage": item.get("stage_index", 0), "settlement": item.get("settlement")} for idea_id, item in payload["ideas"].items()}}, indent=2, sort_keys=True))
            return 0
        if args.campaign_command == "release-human":
            # Use the same supervisor path as library callers so batch
            # manifests and label sets are validated before resuming.
            state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
            rows = [row for row in state.list_rows("runs", limit=100000) if row["run_id"] == args.run_id]
            if not rows:
                print(json.dumps({"state": "missing", "run_id": args.run_id}, sort_keys=True))
                return 1
            payload = json.loads(rows[0]["payload_json"])
            supervisor = CampaignSupervisor(args.state_root, model_id=payload.get("model_id", "unknown"), stage_runner=lambda *_: {"state": "blocked"})
            supervisor.release_human(args.run_id, args.idea_id, json.loads(args.labels.read_text(encoding="utf-8")))
            print(json.dumps({"state": "released", "run_id": args.run_id, "idea_id": args.idea_id}, sort_keys=True))
            return 0
        module_name, function_name = args.runner.split(":", 1)
        runner = getattr(importlib.import_module(module_name), function_name)
        if args.campaign_command == "autonomous-run":
            if args.settle_count is not None and args.settle_count < 1:
                raise SystemExit("--settle-count must be at least 1")
            ai = FixtureEngineeringAI() if args.offline else OpenAICompatibleProvider.from_env()
            if ai is None:
                print(json.dumps({"state": "abstain", "reason": "autonomous-run requires VERDI_AI_* or --offline"}, sort_keys=True))
                return 0
            policy = CampaignPolicy(
                stop_on_first_positive=not args.continue_after_positive and args.settle_count is None,
                target_settled_ideas=args.settle_count,
                poll_seconds=args.poll_seconds,
            )
            supervisor = autonomous_campaign(
                args.state_root,
                model_id=args.model_id,
                stage_runner=runner,
                ai=ai,
                worktree_root=args.worktree_root,
                output_root=args.output_root,
                repository=args.repository,
                readable_roots=tuple(args.readable_root),
                policy=policy,
                replanner=(getattr(importlib.import_module(args.replanner.split(":", 1)[0]), args.replanner.split(":", 1)[1]) if args.replanner else None),
            )
            ideas = json.loads(args.ideas.read_text(encoding="utf-8"))
            if isinstance(ideas, dict):
                ideas = ideas.get("ideas", [])
            try:
                supervisor.load(args.run_id)
                existing_run = True
            except KeyError:
                existing_run = False
            if not existing_run:
                supervisor.create(run_id=args.run_id, objective=args.objective, ideas=ideas)
            first = True
            while True:
                result = supervisor.run_until_blocked(args.run_id) if first else supervisor.resume(args.run_id)
                first = False
                print(json.dumps({"run_id": args.run_id, "state": result["state"]}, sort_keys=True))
                if not args.watch or result["state"] in {"settled", "stopped", "blocked", "waiting_human"}:
                    break
                time.sleep(args.poll_seconds)
            return 0
        supervisor = CampaignSupervisor(args.state_root, model_id=args.model_id, stage_runner=runner, policy=CampaignPolicy(poll_seconds=args.poll_seconds))
        if args.campaign_command == "run":
            ideas = json.loads(args.ideas.read_text(encoding="utf-8"))
            if isinstance(ideas, dict):
                ideas = ideas.get("ideas", [])
            if args.settle_count is not None and args.settle_count < 1:
                raise SystemExit("--settle-count must be at least 1")
            stop_on_first_positive = args.stop_on_first_positive
            if stop_on_first_positive is None:
                stop_on_first_positive = not bool(args.continue_after_positive) and args.settle_count is None
            supervisor.policy = CampaignPolicy(
                max_ideas=supervisor.policy.max_ideas,
                min_replicates=supervisor.policy.min_replicates,
                stop_on_first_positive=stop_on_first_positive,
                target_settled_ideas=args.settle_count,
                human_labels_required_for_model_claim=supervisor.policy.human_labels_required_for_model_claim,
                poll_seconds=args.poll_seconds,
            )
            supervisor.create(run_id=args.run_id, objective=args.objective, ideas=ideas)
        first_iteration = True
        while True:
            result = supervisor.run_until_blocked(args.run_id) if args.campaign_command == "run" and first_iteration else supervisor.resume(args.run_id)
            first_iteration = False
            print(json.dumps({"run_id": args.run_id, "state": result["state"]}, sort_keys=True))
            if not args.watch or result["state"] in {"settled", "stopped"}:
                break
            time.sleep(args.poll_seconds)
        return 0
    if args.command == "graph":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        counts = {table: state.count(table) for table in ("documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges", "graph_nodes", "graph_edges", "knowledge_records", "transfer_assessments")}
        result = {"state": "ready", "counts": counts}
        if args.output_root:
            result["graph"] = str(state.export_graph(args.output_root / "graph.json", portable=args.portable))
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "graph-import":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise SystemExit("graph-import input must contain an entries list")
        summary = import_settlement_entries(state, entries, model_id=args.model_id, campaign_id=args.campaign_id)
        print(json.dumps({"state": "imported", **summary}, sort_keys=True))
        return 0
    if args.command == "graph-bundle":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        print(json.dumps(export_bundle(state, args.output_root, portable=args.portable), indent=2, sort_keys=True))
        return 0
    if args.command == "transfer":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        result = rank_transfer_candidates(state, target_model_id=args.target_model_id, architecture_facets=args.architecture, diagnostic_dimensions=args.diagnostic, capabilities=args.capability, hooks=args.hook)
        print(json.dumps({"state": "ranked", "target_model_id": args.target_model_id, "candidates": result}, indent=2, sort_keys=True))
        return 0
    path = args.state_root / "knowledge" / "knowledge.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"state": "ready", "record_count": len(records), "outcomes": [r["outcome"] for r in records]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
