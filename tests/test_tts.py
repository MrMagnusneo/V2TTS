import configparser
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
import wave

import pytest

import tts
from tts import TTS_MODELS, choose_tts_engine

def test_choose_tts_engine_manual_override():
    assert choose_tts_engine("text", auto_select=False, manual_model="ru_tts") == "ru_tts"
    assert choose_tts_engine("текст", auto_select=False, manual_model="sam") == "sam"

def test_choose_tts_engine_happy_path():
    assert choose_tts_engine("Привет мир") == "ru_tts"
    assert choose_tts_engine("Hello world") == "sam"
    assert choose_tts_engine("Привет world") == "ru_tts"


def test_all_packaged_tts_models_are_offered():
    assert TTS_MODELS == ["ru_tts", "sam", "dectalk", "silero", "coqui"]


def test_resolve_tts_paths_includes_every_vendored_engine(tmp_path: Path):
    paths = tts.resolve_tts_paths(str(tmp_path))

    assert paths.dectalk_python_root == tmp_path / "dectalk-python"
    assert paths.silero_tts_root == tmp_path / "silero-tts-wrapper"
    assert paths.coqui_tts_root == tmp_path / "coqui-tts-wrapper"


def test_every_selectable_vendored_engine_is_a_submodule():
    config = configparser.ConfigParser()
    config.read(".gitmodules", encoding="utf-8")

    expected = {
        "tts/ru_tts-python": "https://github.com/MrMagnusneo/ru_tts-python",
        "tts/sam-python": "https://github.com/MrMagnusneo/sam-python",
        "tts/dectalk-python": "https://github.com/MrMagnusneo/dectalk-python",
        "tts/silero-tts-wrapper": "https://github.com/MrMagnusneo/silero-tts-wrapper",
        "tts/coqui-tts-wrapper": "https://github.com/MrMagnusneo/coqui-tts-wrapper",
    }
    configured = {
        config[section]["path"]: config[section]["url"]
        for section in config.sections()
    }

    assert configured == expected
    assert (Path("tts/silero-tts-wrapper") / "silero_tts").is_dir()
    assert (Path("tts/coqui-tts-wrapper") / "coqui_tts").is_dir()


def test_dectalk_manual_selection_produces_valid_wav(tmp_path: Path):
    output = tmp_path / "dectalk.wav"

    selected = tts.synthesize_text(
        "Hello from DECtalk",
        str(output),
        auto_select=False,
        manual_model="dectalk",
    )

    assert selected == "dectalk"
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() > 0


@pytest.mark.parametrize(
    ("model", "module_name", "class_name", "cache_name"),
    [
        ("silero", "silero_tts", "SileroTTSEngine", "_SILERO_TTS_ENGINES"),
        ("coqui", "coqui_tts", "CoquiTTSEngine", "_COQUI_TTS_ENGINES"),
    ],
)
def test_downloaded_model_engines_support_file_like_wav_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    module_name: str,
    class_name: str,
    cache_name: str,
):
    created = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            created.append(self)

        def synthesize_to_file(self, text, out_path):
            Path(out_path).write_bytes(b"RIFF-vendored-" + text.encode())

    fake_module = ModuleType(module_name)
    setattr(fake_module, class_name, FakeEngine)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    getattr(tts, cache_name).clear()
    output = io.BytesIO()

    first = tts.synthesize_text(
        "hello",
        output,
        auto_select=False,
        manual_model=model,
        tts_root=str(tmp_path),
    )
    second = tts.synthesize_text(
        "again",
        io.BytesIO(),
        auto_select=False,
        manual_model=model,
        tts_root=str(tmp_path),
    )

    assert first == second == model
    assert output.getvalue() == b"RIFF-vendored-hello"
    assert len(created) == 1

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
