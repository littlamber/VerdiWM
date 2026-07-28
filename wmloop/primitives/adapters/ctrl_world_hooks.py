"""Static H1-H5 capability audit for the observed Ctrl-World checkout."""

from __future__ import annotations

from pathlib import Path


HOOK_BINDINGS = {
    "H1": ("dataset/dataset_droid_exp33.py", "class Dataset_mix"),
    "H2": ("models/ctrl_world.py", "class CrtlWorld"),
    "H3": ("models/ctrl_world.py", '"loss_total"'),
    "H4": ("models/pipeline_ctrl_world.py", "class CtrlWorldDiffusionPipeline"),
    "H5": ("scripts/train_wm.py", "optimizer = torch.optim.AdamW"),
}


def audit_ctrl_world_hooks(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    rows = []
    for hook, (relative, anchor) in HOOK_BINDINGS.items():
        path = root / relative
        exists = path.is_file()
        anchor_present = exists and anchor in path.read_text(encoding="utf-8")
        rows.append({"hook": hook, "source_path": relative, "anchor": anchor, "available": bool(exists and anchor_present), "reason": "source_anchor_verified" if exists and anchor_present else "source_anchor_missing"})
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-hook-audit",
        "state": "ready" if all(row["available"] for row in rows) else "blocked",
        "available_hooks": [row["hook"] for row in rows if row["available"]],
        "hook_bindings": rows,
        "source_mutated": False,
        "gpu_started": False,
    }
