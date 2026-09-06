from pathlib import Path

import pytest

from wmloop.control.research_stack import ResearchStackError, audit_research_stack, load_research_stack


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "research" / "research_stack_v1.json"


def test_research_stack_manifest_audits() -> None:
    components = load_research_stack(MANIFEST, repo_root=ROOT)
    assert len(components) == 7
    report = audit_research_stack(MANIFEST, repo_root=ROOT)
    assert report["state"] == "ready"
    assert report["adopted_component_count"] == 6
    assert all(item["authority"] in {"advisory", "proposal_only", "ranking_only"} for item in report["components"])


def test_research_stack_rejects_execution_authority(tmp_path: Path) -> None:
    payload = MANIFEST.read_text(encoding="utf-8").replace('"authority": "advisory"', '"authority": "execute"', 1)
    path = tmp_path / "stack.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ResearchStackError, match="AUTHORITY_INVALID"):
        load_research_stack(path, repo_root=ROOT)
