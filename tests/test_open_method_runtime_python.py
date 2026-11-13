from pathlib import Path
import venv

from wmloop.execute.open_method_runtime import resolve_runtime_python, run_local_command


def test_open_method_process_uses_virtualenv_launcher(tmp_path: Path) -> None:
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = environment / "bin" / "python"
    assert interpreter.is_symlink()
    resolved = resolve_runtime_python(interpreter)
    assert resolved == interpreter

    script = tmp_path / "check.py"
    script.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.prefix)\n",
        encoding="utf-8",
    )
    output = tmp_path / "prefix.txt"
    receipt = run_local_command(
        command=["python", str(script), str(output)], cwd=tmp_path,
        output_root=tmp_path / "process", timeout_seconds=10, runtime_python=resolved,
    )
    assert receipt["state"] == "passed"
    assert Path(output.read_text(encoding="utf-8")) == environment
