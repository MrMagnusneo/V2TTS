from pathlib import Path


def test_windows_workflow_builds_and_smokes_frozen_executable() -> None:
    workflow = Path(".github/workflows/windows-frozen-smoke.yml")
    assert workflow.is_file()

    contents = workflow.read_text(encoding="utf-8")
    assert "runs-on: windows-latest" in contents
    assert "submodules: recursive" in contents
    assert "python installer/build.py --skip-install" in contents
    assert r".\dist\V2TTS.exe --smoke-test" in contents


def test_windows_build_installs_onnx_asr_runtime() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "onnx-asr[cpu,hub]>=0.12,<0.13" in requirements
    assert '"onnx-asr[cpu,hub]>=0.12,<0.13"' in project


def test_windows_build_pins_and_collects_sherpa_without_models() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    spec = Path("installer/V2TTS.spec").read_text(encoding="utf-8")

    assert "sherpa-onnx==1.13.5" in requirements
    assert '"sherpa-onnx==1.13.5"' in project
    assert 'collect_submodules("sherpa_onnx")' in spec
    assert 'collect_data_files("sherpa_onnx")' in spec
    assert 'collect_dynamic_libs("sherpa_onnx")' in spec
    assert 'copy_metadata("sherpa-onnx")' in spec
    assert "sherpa-onnx-streaming-t-one" not in spec
    assert "sherpa-onnx-streaming-zipformer" not in spec


def test_pyinstaller_collects_onnx_asr_without_model_weights() -> None:
    spec = Path("installer/V2TTS.spec").read_text(encoding="utf-8")

    assert 'collect_submodules("onnx_asr")' in spec
    assert 'collect_data_files("onnx_asr")' in spec
    assert 'copy_metadata("onnx-asr")' in spec
    assert 'collect_dynamic_libs("onnxruntime")' in spec
    assert "gigaam-v3-e2e-rnnt" not in spec


def test_windows_frozen_build_matches_supported_python_313() -> None:
    workflow = Path(".github/workflows/windows-frozen-smoke.yml")
    contents = workflow.read_text(encoding="utf-8")

    assert 'python-version: "3.13"' in contents
