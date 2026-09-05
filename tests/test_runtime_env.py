from pathlib import Path

from wmloop.runtime_env import runtime_subprocess_env


def test_symlinked_virtualenv_python_keeps_prefix_and_cuda_wheel_libs(tmp_path: Path) -> None:
    prefix = tmp_path / "venv"
    python = prefix / "bin" / "python"
    (prefix / "bin").mkdir(parents=True)
    package_lib = prefix / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusparselt" / "lib"
    package_lib.mkdir(parents=True)
    python.touch()
    env = runtime_subprocess_env(python, base={"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    assert env["CONDA_PREFIX"] == str(prefix)
    assert env["PATH"].split(":")[0] == str(prefix / "bin")
    assert env["LD_LIBRARY_PATH"].split(":")[:2] == [str(package_lib), str(prefix / "lib")]

