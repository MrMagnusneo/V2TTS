import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pytest
from stt import (
    GigaAMTranscriber,
    WhisperTranscriber,
    create_transcriber,
    default_compute_type,
    _ensure_gigaam_model_files,
)
from stt_profiles import STTSelection

def test_resample_audio_invalid_rates():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Sample rates must be positive."):
        WhisperTranscriber._resample_audio(audio, -16000, 16000)

    with pytest.raises(ValueError, match="Sample rates must be positive."):
        WhisperTranscriber._resample_audio(audio, 16000, 0)

def test_resample_audio_same_rate():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 16000)
    assert result is audio
    np.testing.assert_array_equal(result, audio)

def test_resample_audio_empty_array():
    audio = np.array([], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 8000)
    assert result is audio
    np.testing.assert_array_equal(result, audio)

def test_resample_audio_upsample():
    audio = np.array([0.0, 1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 1, 2)
    expected = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

def test_resample_audio_downsample():
    audio = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 4, 2)
    expected = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

def test_resample_audio_edge_case_small_dst_len():
    audio = np.array([1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 1)
    assert len(result) == 1
    np.testing.assert_array_equal(result, audio)


def test_whisper_factory_passes_explicit_russian_and_download_root(tmp_path):
    with patch("stt.WhisperModel") as model_cls, patch(
        "stt.stt_model_dir", return_value=tmp_path / "medium"
    ):
        transcriber = create_transcriber(
            STTSelection("ru", "whisper", "medium", "cpu")
        )

    assert transcriber.language == "ru"
    assert model_cls.call_args.kwargs["download_root"] == str(tmp_path / "medium")


def test_gigaam_uses_local_directory_and_cpu_provider(tmp_path):
    load_model = MagicMock(return_value=MagicMock())

    with patch("stt._load_onnx_asr_model", load_model):
        GigaAMTranscriber("gigaam-v3-e2e-rnnt", "cpu", tmp_path)

    load_model.assert_called_once_with(
        "gigaam-v3-e2e-rnnt",
        tmp_path,
        providers=["CPUExecutionProvider"],
    )


def test_gigaam_converts_pcm16_and_returns_text(tmp_path):
    model = MagicMock()
    model.recognize.return_value = "Привет, мир."
    with patch("stt._load_onnx_asr_model", return_value=model):
        transcriber = GigaAMTranscriber(
            "gigaam-v3-e2e-rnnt",
            "cpu",
            tmp_path,
        )

    pcm = np.array([0, 16384, -16384], dtype=np.int16)

    assert transcriber.transcribe_pcm16(pcm, 44100) == "Привет, мир."
    waveform = model.recognize.call_args.args[0]
    np.testing.assert_allclose(waveform, [0.0, 0.5, -0.5])
    assert model.recognize.call_args.kwargs == {"sample_rate": 44100}


def test_gigaam_reports_cpu_when_cuda_provider_is_unavailable(tmp_path):
    warnings = []
    load_model = MagicMock(return_value=MagicMock())
    with patch(
        "stt._available_onnx_providers",
        return_value={"CPUExecutionProvider"},
    ), patch("stt._load_onnx_asr_model", load_model):
        transcriber = GigaAMTranscriber(
            "gigaam-v3-e2e-rnnt",
            "cuda",
            tmp_path,
            warnings.append,
        )

    assert transcriber.actual_device == "cpu"
    assert "CUDAExecutionProvider" in transcriber.fallback_reason
    assert warnings == [
        "STT CUDA unavailable; using CPU: "
        "CUDAExecutionProvider is not available"
    ]
    load_model.assert_called_once_with(
        "gigaam-v3-e2e-rnnt",
        tmp_path,
        providers=["CPUExecutionProvider"],
    )


def test_gigaam_cuda_uses_cuda_only_provider(tmp_path):
    session = FakeOnnxSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
    load_model = MagicMock(return_value=FakeGigaAMModel(session))
    with patch(
        "stt._available_onnx_providers",
        return_value={"CUDAExecutionProvider", "CPUExecutionProvider"},
    ), patch("stt._load_onnx_asr_model", load_model):
        transcriber = GigaAMTranscriber(
            "gigaam-v3-e2e-rnnt", "cuda", tmp_path
        )

    assert transcriber.actual_device == "cuda"
    load_model.assert_called_once_with(
        "gigaam-v3-e2e-rnnt",
        tmp_path,
        providers=["CUDAExecutionProvider"],
    )
    assert session.fallback_disabled


def test_gigaam_cuda_load_failure_retries_on_cpu(tmp_path):
    warnings = []
    load_model = MagicMock(side_effect=[RuntimeError("CUDA failed"), MagicMock()])
    with patch(
        "stt._available_onnx_providers",
        return_value={"CUDAExecutionProvider", "CPUExecutionProvider"},
    ), patch("stt._load_onnx_asr_model", load_model):
        transcriber = GigaAMTranscriber(
            "gigaam-v3-e2e-rnnt", "cuda", tmp_path, warnings.append
        )

    assert transcriber.actual_device == "cpu"
    assert load_model.call_args_list[0].kwargs["providers"] == [
        "CUDAExecutionProvider"
    ]
    assert load_model.call_args_list[1].kwargs["providers"] == [
        "CPUExecutionProvider"
    ]


class FakeOnnxSession:
    def __init__(self, providers):
        self.providers = providers
        self.fallback_disabled = False

    def get_providers(self):
        return self.providers

    def disable_fallback(self):
        self.fallback_disabled = True


class FakeGigaAMCore:
    def __init__(self, session):
        self._encoder = session


class FakeGigaAMModel:
    def __init__(self, session):
        self.asr = FakeGigaAMCore(session)


def test_gigaam_silent_cpu_provider_is_reported_as_cpu(tmp_path):
    cpu_session = FakeOnnxSession(["CPUExecutionProvider"])
    cuda_model = FakeGigaAMModel(cpu_session)
    cpu_model = MagicMock()
    load_model = MagicMock(side_effect=[cuda_model, cpu_model])

    with patch(
        "stt._available_onnx_providers",
        return_value={"CUDAExecutionProvider", "CPUExecutionProvider"},
    ), patch("stt._load_onnx_asr_model", load_model):
        transcriber = GigaAMTranscriber(
            "gigaam-v3-e2e-rnnt", "cuda", tmp_path
        )

    assert transcriber.actual_device == "cpu"
    assert "did not activate CUDAExecutionProvider" in transcriber.fallback_reason


def test_partial_gigaam_directory_is_resumed_without_deleting_cache(tmp_path):
    model_dir = tmp_path / "gigaam"
    model_dir.mkdir()
    partial = model_dir / "partial.onnx"
    partial.write_bytes(b"incomplete")

    def ensure_model(_name, path):
        assert partial.exists()

    with (
        patch("stt.stt_model_dir", return_value=model_dir),
        patch("stt._ensure_gigaam_model_files", side_effect=ensure_model) as ensure,
        patch("stt._load_onnx_asr_model", return_value=MagicMock()),
    ):
        create_transcriber(
            STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
        )

    ensure.assert_called_once_with("gigaam-v3-e2e-rnnt", model_dir)
    assert partial.exists()
    assert (model_dir / ".v2tts-complete").is_file()


def test_gigaam_download_is_marked_complete_before_session_load(tmp_path):
    model_dir = tmp_path / "gigaam"

    def ensure_model(_name, path):
        path.mkdir(parents=True)

    def fail_session(*args, **kwargs):
        assert (model_dir / ".v2tts-complete").is_file()
        raise RuntimeError("session init failed")

    with (
        patch("stt.stt_model_dir", return_value=model_dir),
        patch("stt._ensure_gigaam_model_files", side_effect=ensure_model),
        patch("stt._load_onnx_asr_model", side_effect=fail_session),
        pytest.raises(RuntimeError, match="session init failed"),
    ):
        create_transcriber(
            STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
        )


def test_gigaam_file_resolver_is_online_and_resumable(tmp_path):
    resolver = MagicMock()
    with patch("stt._create_asr_resolver", return_value=resolver) as create_resolver:
        _ensure_gigaam_model_files("gigaam-v3-e2e-rnnt", tmp_path)

    create_resolver.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", tmp_path
    )
    resolver.resolve_model.assert_called_once_with()


def test_factory_cuda_fallback_exposes_reason_and_warning(tmp_path):
    warnings = []
    with patch("stt.WhisperModel") as model_cls, patch(
        "stt.stt_model_dir", return_value=tmp_path / "medium"
    ):
        model_cls.side_effect = [
            RuntimeError("cublas64_12.dll missing"),
            MagicMock(),
        ]
        transcriber = create_transcriber(
            STTSelection("en", "whisper", "medium", "cuda"),
            warnings.append,
        )

    assert transcriber.actual_device == "cpu"
    assert "cublas64_12.dll missing" in transcriber.fallback_reason
    assert warnings == [
        "STT CUDA unavailable; using CPU: cublas64_12.dll missing"
    ]


def test_model_load_error_names_model_and_directory(tmp_path):
    model_dir = tmp_path / "gigaam-model"
    with patch(
        "stt._load_onnx_asr_model",
        side_effect=OSError("network down"),
    ), patch("stt.stt_model_dir", return_value=model_dir):
        with pytest.raises(RuntimeError) as error:
            create_transcriber(
                STTSelection(
                    "ru",
                    "gigaam",
                    "gigaam-v3-e2e-rnnt",
                    "cpu",
                )
            )

    message = str(error.value)
    assert "gigaam-v3-e2e-rnnt" in message
    assert str(model_dir) in message

class TestWhisperTranscriber(unittest.TestCase):
    @patch('stt.WhisperModel')
    def test_cuda_fallback(self, mock_whisper_model):
        def side_effect(model_size, **kwargs):
            if kwargs.get('device') == 'cuda':
                raise Exception("No CUDA runtime")
            return MagicMock()

        mock_whisper_model.side_effect = side_effect
        transcriber = WhisperTranscriber(device="cuda")

        self.assertEqual(transcriber.actual_device, 'cpu')
        self.assertEqual(transcriber.compute_type, 'int8')
        self.assertEqual(mock_whisper_model.call_count, 2)

        first_call = mock_whisper_model.call_args_list[0]
        self.assertEqual(first_call.args[0], "medium")
        self.assertEqual(first_call.kwargs['device'], 'cuda')
        self.assertEqual(first_call.kwargs['compute_type'], 'float16')

        second_call = mock_whisper_model.call_args_list[1]
        self.assertEqual(second_call.args[0], "medium")
        self.assertEqual(second_call.kwargs['device'], 'cpu')
        self.assertEqual(second_call.kwargs['compute_type'], 'int8')
        self.assertEqual(second_call.kwargs['num_workers'], 1)
        self.assertEqual(second_call.kwargs['cpu_threads'], max(1, os.cpu_count() or 1))

    @patch('stt.WhisperModel')
    def test_cpu_exception_no_fallback(self, mock_whisper_model):
        mock_whisper_model.side_effect = Exception("CPU initialization failed")
        with self.assertRaises(Exception) as context:
            WhisperTranscriber(device="cpu")
        self.assertTrue("CPU initialization failed" in str(context.exception))
        self.assertEqual(mock_whisper_model.call_count, 1)

class TestDefaultComputeType(unittest.TestCase):
    def test_cuda_device(self):
        self.assertEqual(default_compute_type('cuda'), 'float16')

    def test_cpu_device(self):
        self.assertEqual(default_compute_type('cpu'), 'int8')

    def test_fallback_devices(self):
        self.assertEqual(default_compute_type('mps'), 'int8')
        self.assertEqual(default_compute_type('unknown'), 'int8')
        self.assertEqual(default_compute_type(''), 'int8')

if __name__ == '__main__':
    unittest.main()
