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

if __name__ == '__main__':
    unittest.main()
