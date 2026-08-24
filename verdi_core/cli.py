"""CLI for the clean release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.fixture_world import FixtureWorldAdapter
from adapters.fixture_research import FixtureResearchAI

from .loop import run_loop
from .ingestion import DocumentIngestor
from .storage import SQLiteState
from .research import ResearchSystem
from .runtime import RuntimeBindings
from .providers import OpenAICompatibleProvider
from .retrieval import OfflineRetriever, OnlineRetriever
from .search_sources import ArxivBackend, CrossrefBackend, FanoutBackend, GitHubBackend, SemanticScholarBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verdi")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    demo = commands.add_parser("demo")
    demo.add_argument("--state-root", type=Path, required=True)
    graph = commands.add_parser("graph")
    graph.add_argument("--state-root", type=Path, required=True)
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
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps({"state": "ready", "kernel": "model-agnostic", "adapter_contract": "v1"}, sort_keys=True))
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
    if args.command == "graph":
        state = SQLiteState(args.state_root / "knowledge" / "knowledge.sqlite3")
        print(json.dumps({"state": "ready", "counts": {table: state.count(table) for table in ("documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges")}}, sort_keys=True))
        return 0
    path = args.state_root / "knowledge" / "knowledge.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"state": "ready", "record_count": len(records), "outcomes": [r["outcome"] for r in records]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
