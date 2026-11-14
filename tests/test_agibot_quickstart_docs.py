from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUICKSTARTS = (
    ROOT / "docs/AGIBOT_ENGINEERING_DIAGNOSTIC_QUICKSTART.md",
    ROOT / "docs/AGIBOT_ENGINEERING_DIAGNOSTIC_QUICKSTART.en.md",
)
STATE_ROOT_EXPORT = 'export VERDIWM_STATE_ROOT="$HOME/.local/state/verdiwm/agibot-state"'


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def test_agibot_quickstarts_keep_campaign_output_outside_repository(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "home" / "user" / "work" / "VerdiWM"
    home = tmp_path / "home" / "user"
    repo_root.mkdir(parents=True)
    state_root = home / ".local/state/verdiwm/agibot-state"
    output_root = state_root / "runs" / "agibot-campaign"

    for quickstart in QUICKSTARTS:
        contents = quickstart.read_text(encoding="utf-8")
        assert STATE_ROOT_EXPORT in contents

        # Model the documented $HOME expansion while the command runs from the
        # checkout root. The run output is nested under state_root/runs.
        documented_path = Path(STATE_ROOT_EXPORT.split("=", 1)[1].strip('"'))
        state_path = Path(str(documented_path).replace("$HOME", str(home)))
        documented_output = state_path / "runs" / "agibot-campaign"
        assert state_path == state_root
        assert not _paths_overlap(state_path, repo_root)
        assert not _paths_overlap(documented_output, repo_root)
        assert not _paths_overlap(output_root, repo_root)
