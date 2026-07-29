import pytest
from tts import choose_tts_engine

def test_choose_tts_engine_manual_override():
    assert choose_tts_engine("text", auto_select=False, manual_model="ru_tts") == "ru_tts"
    assert choose_tts_engine("текст", auto_select=False, manual_model="sam") == "sam"

def test_choose_tts_engine_happy_path():
    assert choose_tts_engine("Привет мир") == "silero"
    assert choose_tts_engine("Hello world") == "sam"
    assert choose_tts_engine("Привет world") == "silero"

def test_choose_tts_engine_edge_cases():
    assert choose_tts_engine("") == "sam"
    assert choose_tts_engine("!@#") == "sam"
    assert choose_tts_engine("123") == "sam"
    assert choose_tts_engine("   ") == "sam"
