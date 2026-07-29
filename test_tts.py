import unittest
from tts import choose_tts_engine

class TestChooseTtsEngine(unittest.TestCase):
    def test_auto_select_false(self):
        self.assertEqual(choose_tts_engine("hello", auto_select=False, manual_model="custom"), "custom")
        self.assertEqual(choose_tts_engine("привет", auto_select=False, manual_model="other"), "other")

    def test_cyrillic_text(self):
        self.assertEqual(choose_tts_engine("Привет"), "silero")
        self.assertEqual(choose_tts_engine("Hello Привет"), "silero")

    def test_latin_text(self):
        self.assertEqual(choose_tts_engine("Hello"), "sam")

    def test_edge_cases(self):
        # Empty string
        self.assertEqual(choose_tts_engine(""), "sam")
        # Special characters
        self.assertEqual(choose_tts_engine("!@#"), "sam")
        # Numbers
        self.assertEqual(choose_tts_engine("12345"), "sam")
        # Whitespace only
        self.assertEqual(choose_tts_engine("   "), "sam")

if __name__ == '__main__':
    unittest.main()
