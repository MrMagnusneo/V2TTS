import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pytest
from stt import WhisperTranscriber, default_compute_type

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
