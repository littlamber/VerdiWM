import json
import subprocess
import sys


def test_offline_cycle_is_a_complete_release_smoke(tmp_path):
    state = tmp_path / "state"
    completed = subprocess.run([sys.executable, "-m", "verdi_core.cli", "cycle", "--offline", "--state-root", str(state), "--objective", "quality"], capture_output=True, text=True, check=True)
    report = json.loads(completed.stdout)
    assert report["idea_count"] == 1
    assert report["evidence_count"] == 1
    assert report["benchmark_review"]["status"] == "reviewed"
