import unittest
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
