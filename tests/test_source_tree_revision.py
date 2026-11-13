from __future__ import annotations

from pathlib import Path

import pytest

from wmloop.control.onboarding import compute_source_tree_revision


@pytest.mark.parametrize("relative", [
    "sa_wm/dataset.py", "sa_wm/trajectory/loader.py",
    "configs/training/base.yaml", "configs/dataset/train.yaml",
])
def test_source_binding_includes_loaders_and_runtime_configuration(tmp_path: Path, relative: str) -> None:
    dependency = tmp_path / relative
    dependency.parent.mkdir(parents=True)
    dependency.write_text("initial content\n", encoding="utf-8")
    before = compute_source_tree_revision(tmp_path)
    dependency.write_text("changed content\n", encoding="utf-8")
    after = compute_source_tree_revision(tmp_path)
    assert before["state"] == after["state"] == "bound"
    assert before["kind"] == after["kind"] == "source_tree_sha256"
    assert before["revision"] != after["revision"]


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_source_binding_rejects_unbound_links(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("# main\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    dependency = external / "dependency.py"
    dependency.write_text("# external dependency\n", encoding="utf-8")
    if kind == "file":
        (source / "dependency.py").symlink_to(dependency)
    elif kind == "directory":
        (source / "sa_wm").symlink_to(external, target_is_directory=True)
    else:
        linked = tmp_path / "linked-source"
        linked.symlink_to(source, target_is_directory=True)
        source = linked
    result = compute_source_tree_revision(source)
    assert result["state"] == "unbound"
    assert "SOURCE_REVISION_SYMLINK_FORBIDDEN" in result["detail"]


def test_source_binding_keeps_existing_inventory_limit(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("# first\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("# second\n", encoding="utf-8")
    result = compute_source_tree_revision(tmp_path, max_files=1)
    assert result["state"] == "unbound"
    assert "scan limit" in result["detail"]
