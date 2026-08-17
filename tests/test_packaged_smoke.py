from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import main
import smoke_test


def test_multiprocessing_smoke_round_trip() -> None:
    assert smoke_test.run_multiprocessing_smoke() is None


def test_smoke_synthesizes_and_decodes_both_tts_engines(tmp_path: Path) -> None:
    soundfile = MagicMock()
    soundfile.read.return_value = (np.array([0.1, -0.1]), 22050)

    with (
        patch(
            "smoke_test.synthesize_text",
            side_effect=lambda *args, **kwargs: kwargs["manual_model"],
        ) as synthesize,
        patch("smoke_test.get_soundfile", return_value=soundfile),
    ):
        assert smoke_test.run_packaged_smoke(tts_root=tmp_path) == 0

    assert [call.kwargs["manual_model"] for call in synthesize.call_args_list] == [
        "ru_tts",
        "sam",
    ]
    assert all(call.kwargs["auto_select"] is False for call in synthesize.call_args_list)
    assert all(call.kwargs["tts_root"] == str(tmp_path) for call in synthesize.call_args_list)
    assert soundfile.read.call_count == 2


def test_smoke_rejects_empty_audio(tmp_path: Path) -> None:
    soundfile = MagicMock()
    soundfile.read.return_value = (np.array([]), 22050)

    with (
        patch(
            "smoke_test.synthesize_text",
            side_effect=lambda *args, **kwargs: kwargs["manual_model"],
        ),
        patch("smoke_test.get_soundfile", return_value=soundfile),
        pytest.raises(RuntimeError, match="empty audio"),
    ):
        smoke_test.run_packaged_smoke(tts_root=tmp_path)


def test_main_smoke_mode_does_not_create_gui() -> None:
    with (
        patch("main.multiprocessing.freeze_support") as freeze_support,
        patch("main.check_runtime_dependencies") as check_dependencies,
        patch("smoke_test.run_packaged_smoke", return_value=0) as run_smoke,
        patch("main.AppController") as app_controller,
    ):
        assert main.main(["--smoke-test"]) == 0

    check_dependencies.assert_called_once_with()
    freeze_support.assert_called_once_with()
    run_smoke.assert_called_once_with()
    app_controller.assert_not_called()


def test_main_restores_standard_streams_for_windowed_executable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    def dependency_check() -> None:
        sys.stdout.write("")
        sys.stderr.write("")

    with (
        patch("main.multiprocessing.freeze_support"),
        patch("main.check_runtime_dependencies", side_effect=dependency_check),
        patch("smoke_test.run_packaged_smoke", return_value=0),
    ):
        assert main.main(["--smoke-test"]) == 0
