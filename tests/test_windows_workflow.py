from pathlib import Path


def test_windows_workflow_builds_and_smokes_frozen_executable() -> None:
    workflow = Path(".github/workflows/windows-frozen-smoke.yml")
    assert workflow.is_file()

    contents = workflow.read_text(encoding="utf-8")
    assert "runs-on: windows-latest" in contents
    assert "submodules: recursive" in contents
    assert "python installer/build.py --skip-install" in contents
    assert r".\dist\V2TTS.exe --smoke-test" in contents
