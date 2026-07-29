import unittest
import numpy as np
from stt import WhisperTranscriber

class TestWhisperTranscriber(unittest.TestCase):
    def test_resample_audio_downsample(self):
        # Downsample from 44100 to 16000
        src_rate = 44100
        dst_rate = 16000
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
        resampled = WhisperTranscriber._resample_audio(audio, src_rate, dst_rate)
        expected_len = max(1, int(round(len(audio) * dst_rate / src_rate)))
        self.assertEqual(len(resampled), expected_len)
        self.assertEqual(resampled.dtype, np.float32)

    def test_resample_audio_upsample(self):
        # Upsample from 8000 to 16000
        src_rate = 8000
        dst_rate = 16000
        audio = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        resampled = WhisperTranscriber._resample_audio(audio, src_rate, dst_rate)
        expected_len = max(1, int(round(len(audio) * dst_rate / src_rate)))
        self.assertEqual(len(resampled), expected_len)
        self.assertEqual(resampled.dtype, np.float32)

    def test_resample_audio_equal_rates(self):
        # Equal sample rates should return the original audio
        src_rate = 16000
        dst_rate = 16000
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        resampled = WhisperTranscriber._resample_audio(audio, src_rate, dst_rate)
        np.testing.assert_array_equal(resampled, audio)

    def test_resample_audio_empty_array(self):
        # Empty array should return original array
        src_rate = 16000
        dst_rate = 8000
        audio = np.array([], dtype=np.float32)
        resampled = WhisperTranscriber._resample_audio(audio, src_rate, dst_rate)
        np.testing.assert_array_equal(resampled, audio)

    def test_resample_audio_negative_src_rate(self):
        # Negative src_rate should raise ValueError
        audio = np.array([0.1], dtype=np.float32)
        with self.assertRaises(ValueError):
            WhisperTranscriber._resample_audio(audio, -16000, 16000)

    def test_resample_audio_zero_src_rate(self):
        # Zero src_rate should raise ValueError
        audio = np.array([0.1], dtype=np.float32)
        with self.assertRaises(ValueError):
            WhisperTranscriber._resample_audio(audio, 0, 16000)

    def test_resample_audio_negative_dst_rate(self):
        # Negative dst_rate should raise ValueError
        audio = np.array([0.1], dtype=np.float32)
        with self.assertRaises(ValueError):
            WhisperTranscriber._resample_audio(audio, 16000, -16000)

    def test_resample_audio_zero_dst_rate(self):
        # Zero dst_rate should raise ValueError
        audio = np.array([0.1], dtype=np.float32)
        with self.assertRaises(ValueError):
            WhisperTranscriber._resample_audio(audio, 16000, 0)
from stt import default_compute_type
from unittest.mock import patch, MagicMock
import os
from stt import WhisperTranscriber, default_compute_type

class TestWhisperTranscriber(unittest.TestCase):
    @patch('stt.WhisperModel')
    def test_cuda_fallback(self, mock_whisper_model):
        # We want to mock WhisperModel so that if it's called with device="cuda",
        # it raises an Exception, simulating a missing CUDA runtime.
        # But wait, when the fallback happens, it calls WhisperModel AGAIN with device="cpu".
        # We need the second call to succeed.
        def side_effect(model_size, **kwargs):
            if kwargs.get('device') == 'cuda':
                raise Exception("No CUDA runtime")
            return MagicMock()

        mock_whisper_model.side_effect = side_effect

        transcriber = WhisperTranscriber(device="cuda")

        # Check that it fell back to actual_device='cpu'
        self.assertEqual(transcriber.actual_device, 'cpu')
        self.assertEqual(transcriber.compute_type, 'int8')

        # Check the calls made to WhisperModel
        self.assertEqual(mock_whisper_model.call_count, 2)

        # First call should have device="cuda"
        first_call = mock_whisper_model.call_args_list[0]
        self.assertEqual(first_call.args[0], "medium")
        self.assertEqual(first_call.kwargs['device'], 'cuda')
        self.assertEqual(first_call.kwargs['compute_type'], 'float16')

        # Second call should have device="cpu", compute_type="int8"
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
