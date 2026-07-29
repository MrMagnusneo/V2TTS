import unittest
from unittest.mock import MagicMock
from audio_queue import SpeechLoopRunner

class TestSelectInputSampleRate(unittest.TestCase):
    def setUp(self):
        self.sd_mock = MagicMock()

    def test_first_preferred_rate_succeeds(self):
        # sd.check_input_settings succeeds immediately
        self.sd_mock.check_input_settings.return_value = None
        self.sd_mock.query_devices.return_value = {"default_samplerate": 16000.0}

        rate = SpeechLoopRunner._select_input_sample_rate(self.sd_mock, device_index=1)

        self.assertEqual(rate, 16000)
        self.sd_mock.check_input_settings.assert_called_once_with(
            device=1, channels=1, samplerate=16000, dtype="float32"
        )
        self.sd_mock.query_devices.assert_called_once_with(1, "input")

    def test_fallback_to_second_preferred_rate(self):
        # Fail for 16000, succeed for 32000
        def check_input_settings_side_effect(device, channels, samplerate, dtype):
            if samplerate == 16000:
                raise Exception("Unsupported sample rate")
            return None

        self.sd_mock.check_input_settings.side_effect = check_input_settings_side_effect
        self.sd_mock.query_devices.return_value = {"default_samplerate": 16000.0}

        rate = SpeechLoopRunner._select_input_sample_rate(self.sd_mock, device_index=1)

        self.assertEqual(rate, 32000)
        self.assertEqual(self.sd_mock.check_input_settings.call_count, 3)
        self.sd_mock.query_devices.assert_called_once_with(1, "input")

    def test_exhaust_preferred_rates_valid_fallback(self):
        # Fail for all preferred rates
        def check_input_settings_side_effect(device, channels, samplerate, dtype):
            if samplerate == 22050:
                return None
            raise Exception("Unsupported sample rate")

        self.sd_mock.check_input_settings.side_effect = check_input_settings_side_effect
        self.sd_mock.query_devices.return_value = {"default_samplerate": 22050.0}

        rate = SpeechLoopRunner._select_input_sample_rate(self.sd_mock, device_index=1)

        self.assertEqual(rate, 22050)
        self.assertEqual(self.sd_mock.check_input_settings.call_count, 1)
        self.sd_mock.query_devices.assert_called_once_with(1, "input")

    def test_exhaust_preferred_rates_missing_fallback(self):
        # Fail for all preferred rates
        self.sd_mock.check_input_settings.side_effect = Exception("Unsupported sample rate")

        # query_devices returns dict without default_samplerate
        self.sd_mock.query_devices.return_value = {}

        rate = SpeechLoopRunner._select_input_sample_rate(self.sd_mock, device_index=1)

        self.assertEqual(rate, 16000)
        self.assertEqual(self.sd_mock.check_input_settings.call_count, 5)
        self.sd_mock.query_devices.assert_called_once_with(1, "input")

    def test_exhaust_preferred_rates_invalid_fallback(self):
        # Fail for all preferred rates
        self.sd_mock.check_input_settings.side_effect = Exception("Unsupported sample rate")

        # query_devices returns zero or negative default_samplerate
        self.sd_mock.query_devices.return_value = {"default_samplerate": -1.0}

        rate = SpeechLoopRunner._select_input_sample_rate(self.sd_mock, device_index=1)

        self.assertEqual(rate, 16000)
        self.assertEqual(self.sd_mock.check_input_settings.call_count, 5)
        self.sd_mock.query_devices.assert_called_once_with(1, "input")

if __name__ == '__main__':
    unittest.main()
