from pathlib import Path

import pytest

from installer import build


def test_find_gcc_discovers_candidate_outside_path(tmp_path: Path) -> None:
    compiler = tmp_path / "ucrt64" / "bin" / "gcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()

    assert build.find_gcc(path_env="", candidates=[compiler]) == compiler


def test_ensure_gcc_adds_discovered_compiler_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / "mingw64" / "bin" / "gcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setenv("PATH", "existing-path")

    selected = build.ensure_gcc_available(
        path_env="",
        candidates=[compiler],
        platform="win32",
    )

    assert selected == compiler
    assert build.os.environ["PATH"].split(build.os.pathsep)[0] == str(compiler.parent)


def test_missing_windows_gcc_reports_install_commands() -> None:
    with pytest.raises(RuntimeError) as error:
        build.ensure_gcc_available(
            path_env="",
            candidates=[],
            platform="win32",
        )

    message = str(error.value)
    assert "winget install --id MSYS2.MSYS2 -e" in message
    assert "mingw-w64-ucrt-x86_64-gcc" in message
    assert "python installer/build.py" in message
