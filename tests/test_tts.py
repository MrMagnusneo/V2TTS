from pathlib import Path
import unittest

from tts import TTS_MODELS, choose_tts_engine

def test_choose_tts_engine_manual_override():
    assert choose_tts_engine("text", auto_select=False, manual_model="ru_tts") == "ru_tts"
    assert choose_tts_engine("текст", auto_select=False, manual_model="sam") == "sam"

def test_choose_tts_engine_happy_path():
    assert choose_tts_engine("Привет мир") == "ru_tts"
    assert choose_tts_engine("Hello world") == "sam"
    assert choose_tts_engine("Привет world") == "ru_tts"


def test_only_packaged_tts_models_are_offered():
    assert TTS_MODELS == ["ru_tts", "sam"]


def test_silero_is_not_a_runtime_dependency():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    main_source = Path("main.py").read_text(encoding="utf-8")

    assert "silero-tts" not in requirements
    assert "silero_tts" not in main_source

def test_choose_tts_engine_edge_cases():
    assert choose_tts_engine("") == "sam"
    assert choose_tts_engine("!@#") == "sam"
    assert choose_tts_engine("123") == "sam"
    assert choose_tts_engine("   ") == "sam"

class TestChooseTTSEngine(unittest.TestCase):
    def test_auto_select_false_returns_manual_model(self):
        self.assertEqual(choose_tts_engine("test", auto_select=False, manual_model="some_model"), "some_model")
        self.assertEqual(choose_tts_engine("test", auto_select=False, manual_model="another_model"), "another_model")

    def test_cyrillic_text_returns_ru_tts(self):
        self.assertEqual(choose_tts_engine("Привет мир", auto_select=True), "ru_tts")
        self.assertEqual(choose_tts_engine("mixed text с кириллицей", auto_select=True), "ru_tts")

    def test_latin_text_returns_sam(self):
        self.assertEqual(choose_tts_engine("Hello world", auto_select=True), "sam")
        self.assertEqual(choose_tts_engine("mixed text with numbers 123", auto_select=True), "sam")

    def test_non_alphabetic_text_returns_sam_default(self):
        self.assertEqual(choose_tts_engine("12345", auto_select=True), "sam")
        self.assertEqual(choose_tts_engine("!@#$%", auto_select=True), "sam")
        self.assertEqual(choose_tts_engine("", auto_select=True), "sam")

if __name__ == '__main__':
    unittest.main()
