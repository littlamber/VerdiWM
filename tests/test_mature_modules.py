import json
import sys
from pathlib import Path

from verdi_core.evaluators import StatisticalEvaluator
from verdi_core.ingestion import DocumentIngestor
from verdi_core.scheduler import ExperimentJob, LocalScheduler
from verdi_core.storage import SQLiteState
from verdi_core.workers import ExperimentTask, LocalPythonWorker


def test_sqlite_upsert_and_knowledge_projection(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "state.sqlite3")
    state.put_idea("i1", {"title": "first", "mechanism": "m"})
    state.put_idea("i1", {"title": "updated", "mechanism": "m2"}, "admitted")
    assert state.count("ideas") == 1
    assert state.list_rows("ideas")[0]["status"] == "admitted"


def test_ingestor_html_code_and_ocr_fallback(tmp_path: Path) -> None:
    html = tmp_path / "paper.html"
    html.write_text("<html><title>Paper</title><body><h1>Method</h1><p>Evidence</p></body></html>")
    code = tmp_path / "method.py"
    code.write_text("def method(x):\n    return x\n")
    ingestor = DocumentIngestor(tmp_path / "inbox")
    assert "Evidence" in ingestor.ingest_file(html).text
    assert ingestor.ingest_file(code).status == "acquired_code"


def test_local_worker_and_retry(tmp_path: Path) -> None:
    worker = LocalPythonWorker([sys.executable, "-c", "import json,sys; x=json.load(sys.stdin); print(json.dumps({'raw_result': {'delta': 0.1, 'protected_ok': True}}))"])
    scheduler = LocalScheduler(1.0)
    result = scheduler.run([ExperimentJob("j1", "h1", 0.1, {})], lambda job: worker.execute(ExperimentTask(job.job_id, {}, "heldout")), retries=1)
    assert result[0]["state"] == "settled"


def test_statistical_evaluator_abstains_on_unstable_repeats() -> None:
    result = StatisticalEvaluator().evaluate({"raw_result": {"delta": 0.2, "protected_ok": True, "replicate_deltas": [0.2, -0.1]}}, split="heldout", metrics={})
    assert result["outcome"] == "abstain"


def test_scheduler_resume_reads_persisted_jobs(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "state.sqlite3")
    scheduler = LocalScheduler(1.0, state=state)
    job = ExperimentJob("resume-job", "h", 0.1, {"hypothesis_id": "h", "estimated_cost": 0.1})
    state.put_experiment(job.job_id, job.payload, "queued")
    results = scheduler.resume(lambda current: {"raw_result": {"delta": 0.0, "protected_ok": True}})
    assert results[0]["state"] == "settled"
