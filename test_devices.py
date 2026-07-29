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
