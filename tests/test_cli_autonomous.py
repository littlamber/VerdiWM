import json
from pathlib import Path
import subprocess

from verdi_core.cli import main
from verdi_core.storage import SQLiteState


def test_autonomous_campaign_cli_fixture_closes_loop(tmp_path: Path, capsys) -> None:
    state_root = tmp_path / "state"
    ideas = tmp_path / "ideas.json"
    ideas.write_text(json.dumps({"ideas": [{"idea_id": "fixture-idea", "title": "bounded repair"}]}), encoding="utf-8")
    code = main([
        "campaign", "autonomous-run",
        "--state-root", str(state_root),
        "--run-id", "fixture-run",
        "--model-id", "fixture-world-v1",
        "--objective", "improve quality",
        "--ideas", str(ideas),
        "--runner", "adapters.fixture_campaign:runner",
        "--worktree-root", str(tmp_path / "worktrees"),
        "--output-root", str(tmp_path / "artifacts"),
        "--offline",
    ])
    assert code == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["state"] == "stopped"
    state = SQLiteState(state_root / "knowledge" / "knowledge.sqlite3")
    runs = state.list_rows("runs")
    payload = json.loads(runs[0]["payload_json"])
    assert payload["state"] == "stopped"
    item = payload["ideas"]["fixture-idea"]
    assert item["settlement"]["outcome"] == "confirmed_positive"
    assert any(event["result"].get("engineering", {}).get("state") == "repaired_and_retried" for event in item["attempts"])


def test_autonomous_campaign_uses_detached_worktree_when_repository_given(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sentinel.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "sentinel.txt"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=test", "commit", "-qm", "init"], cwd=repo, check=True)
    ideas = tmp_path / "ideas.json"
    ideas.write_text(json.dumps({"ideas": [{"idea_id": "fixture-idea-detached", "title": "bounded repair"}]}), encoding="utf-8")
    assert main([
        "campaign", "autonomous-run", "--state-root", str(tmp_path / "state"),
        "--run-id", "run", "--model-id", "fixture", "--objective", "quality",
        "--ideas", str(ideas), "--runner", "adapters.fixture_campaign:runner",
        "--worktree-root", str(tmp_path / "worktrees"), "--output-root", str(tmp_path / "out"),
        "--repository", str(repo), "--offline",
    ]) == 0
    assert (repo / "sentinel.txt").read_text(encoding="utf-8") == "original\n"
    assert (tmp_path / "worktrees" / "run" / "fixture-idea-detached" / "static_check" / ".git").is_file()
