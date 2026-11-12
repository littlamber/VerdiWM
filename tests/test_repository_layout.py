from __future__ import annotations

from pathlib import Path

from wmloop.repository_layout import (
    audit_component_dependencies,
    audit_repository_layout,
    large_python_modules,
)
from wmloop.runtime_env import default_lock_root


ROOT = Path(__file__).resolve().parents[1]


def test_current_repository_matches_declared_layout() -> None:
    assert audit_repository_layout(ROOT) == ()


def test_large_modules_are_reported_as_refactor_candidates_without_failing_layout() -> (
    None
):
    rows = large_python_modules(ROOT)
    assert rows
    assert rows[0]["bytes"] >= 50_000
    assert all(str(row["path"]).startswith("wmloop/") for row in rows)


def test_layout_audit_rejects_new_root_and_script_names(tmp_path: Path) -> None:
    (tmp_path / "wmloop/control").mkdir(parents=True)
    (tmp_path / "scripts/export").mkdir(parents=True)
    (tmp_path / "experiments/foo_v1").mkdir(parents=True)
    (tmp_path / "wmloop/control/__init__.py").write_text("")
    (tmp_path / "wmloop/bad.py").write_text("")
    (tmp_path / "scripts/export/bad_name.py").write_text("")
    (tmp_path / "experiments/foo_v1/main.py").write_text("")
    for name in (
        "artifacts",
        "results",
        "runs",
        "outputs",
        "logs",
        "cache",
        "build",
        "dist",
    ):
        (tmp_path / name).mkdir()
    issues = audit_repository_layout(tmp_path)
    codes = {issue.code for issue in issues}
    assert "WMLOOP_ROOT_MODULE_UNKNOWN" in codes
    assert "SCRIPT_NAME_INVALID" in codes
    assert "TOP_LEVEL_DIRECTORY_UNKNOWN" not in codes


def test_runtime_lock_root_is_configurable_and_has_no_machine_binding(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VERDIWM_LOCK_ROOT", str(tmp_path / "locks"))
    assert default_lock_root() == tmp_path / "locks"
    monkeypatch.delenv("VERDIWM_LOCK_ROOT")
    assert default_lock_root().name == "verdiwm-gpu-leases"


def test_component_audit_rejects_facade_back_import_and_cycles(tmp_path: Path) -> None:
    package = tmp_path / "wmloop/execute/pipeline"
    package.mkdir(parents=True)
    (package / "first.py").write_text(
        "from .second import value\nfrom wmloop.execute import autonomous_pipeline\n"
    )
    (package / "second.py").write_text("from .first import value\n")
    codes = {issue.code for issue in audit_component_dependencies(tmp_path)}
    assert codes == {"COMPONENT_IMPORTS_FACADE", "COMPONENT_IMPORT_CYCLE"}


def test_retrieval_cannot_import_execution_even_in_function(tmp_path: Path) -> None:
    package = tmp_path / "wmloop/retrieve"
    package.mkdir(parents=True)
    (package / "invalid.py").write_text(
        "def run():\n    from wmloop import execute\n    import torch\n"
    )
    issues = audit_component_dependencies(tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "DOMAIN_IMPORTS_EXECUTION"


def test_component_size_limit_does_not_allow_rebuilding_monolith(
    tmp_path: Path,
) -> None:
    package = tmp_path / "wmloop/control/commands"
    package.mkdir(parents=True)
    (package / "oversized.py").write_text("# one responsibility per component\n" * 801)
    assert {issue.code for issue in audit_component_dependencies(tmp_path)} == {
        "COMPONENT_TOO_LARGE"
    }


def test_installed_service_cannot_depend_on_checkout_script(tmp_path: Path) -> None:
    package = tmp_path / "wmloop/experiments"
    package.mkdir(parents=True)
    (package / "repair.py").write_text("def run():\n    from scripts.export import helper\n")
    issues = audit_component_dependencies(tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "LIBRARY_IMPORTS_SCRIPT"
