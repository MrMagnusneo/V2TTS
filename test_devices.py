import unittest
from devices import parse_index_from_label

class TestDevices(unittest.TestCase):
    def test_parse_index_from_label_valid(self):
        self.assertEqual(parse_index_from_label("[12] Device name"), 12)
        self.assertEqual(parse_index_from_label("[0] Default Device"), 0)

    def test_parse_index_from_label_missing_brackets(self):
        with self.assertRaises(ValueError):
            parse_index_from_label("Device name")
        with self.assertRaises(ValueError):
            parse_index_from_label("12] Device name")
        with self.assertRaises(ValueError):
            parse_index_from_label("[12 Device name")

    def test_parse_index_from_label_empty_brackets(self):
        with self.assertRaises(ValueError):
            parse_index_from_label("[] Device name")

    def test_parse_index_from_label_reversed_brackets(self):
        with self.assertRaises(ValueError):
            parse_index_from_label("]12[ Device name")

    def test_parse_index_from_label_invalid_integer(self):
        with self.assertRaises(ValueError):
            parse_index_from_label("[abc] Device name")

if __name__ == '__main__':
    unittest.main()
from unittest.mock import patch, MagicMock
from devices import list_audio_devices, AudioDevice

class TestListAudioDevices(unittest.TestCase):
    @patch('devices.get_sounddevice')
    def test_list_audio_devices(self, mock_get_sounddevice):
        # Setup mock sounddevice
        mock_sd = MagicMock()
        mock_get_sounddevice.return_value = mock_sd

        # Synthetic devices data
        synthetic_devices = [
            {
                'name': 'Built-in Microphone',
                'max_input_channels': 2,
                'max_output_channels': 0,
            },
            {
                'name': 'Built-in Output',
                'max_input_channels': 0,
                'max_output_channels': 2,
            },
            {
                'name': 'External Headset',
                'max_input_channels': 1,
                'max_output_channels': 2,
            }
        ]

        # Mock query_devices to return our synthetic list
        mock_sd.query_devices.return_value = synthetic_devices

        # Call the function
        result = list_audio_devices()

        # Assertions
        self.assertEqual(len(result), 3)

        self.assertEqual(result[0].index, 0)
        self.assertEqual(result[0].name, 'Built-in Microphone')
        self.assertEqual(result[0].max_input_channels, 2)
        self.assertEqual(result[0].max_output_channels, 0)

        self.assertEqual(result[1].index, 1)
        self.assertEqual(result[1].name, 'Built-in Output')
        self.assertEqual(result[1].max_input_channels, 0)
        self.assertEqual(result[1].max_output_channels, 2)

        self.assertEqual(result[2].index, 2)
        self.assertEqual(result[2].name, 'External Headset')
        self.assertEqual(result[2].max_input_channels, 1)
        self.assertEqual(result[2].max_output_channels, 2)

if __name__ == '__main__':
    unittest.main()
