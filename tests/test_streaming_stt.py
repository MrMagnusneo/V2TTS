import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from streaming_models import STREAMING_MODEL_PROFILES
from streaming_stt import (
    SherpaStreamingTranscriber,
    StableTextCommitter,
    StreamingResult,
)


def test_streaming_result_is_an_explicit_online_contract() -> None:
    assert StreamingResult("привет", True).text == "привет"
    assert StreamingResult("привет", True).endpoint is True


def test_revised_tail_is_not_spoken_and_committed_words_do_not_repeat() -> None:
    committer = StableTextCommitter()
    assert committer.observe("я хочу проверить потоковую") == []
    assert committer.observe("я хочу проверить потоковое распознавание") == []
    assert committer.observe("я хочу проверить потоковое распознавание речи") == []
    assert committer.observe(
        "я хочу проверить потоковое распознавание речи сейчас"
    ) == ["я хочу проверить потоковое распознавание речи"]
    assert committer.finish(
        "я хочу проверить потоковое распознавание речи сейчас"
    ) == ["сейчас"]


def test_clause_boundary_commits_after_three_stable_words() -> None:
    committer = StableTextCommitter()
    assert committer.observe("when this works, we continue") == []
    assert committer.observe("when this works, we continue testing") == [
        "when this works,"
    ]


def test_six_word_target_and_eight_word_cap_preserve_order() -> None:
    committer = StableTextCommitter()
    text = "one two three four five six seven eight nine ten eleven twelve"
    assert committer.observe(text) == []
    assert committer.observe(text) == [
        "one two three four five six",
        "seven eight nine ten eleven twelve",
    ]


def test_endpoint_flushes_every_word_even_past_eight_word_cap() -> None:
    committer = StableTextCommitter()
    text = "one two three four five six seven eight nine ten"
    assert committer.finish(text) == [
        "one two three four five six",
        "seven eight nine ten",
    ]


def test_pause_flushes_short_latest_tail_only_once() -> None:
    committer = StableTextCommitter()
    committer.observe("короткая фраза")
    committer.observe("короткая фраза")
    assert committer.flush_pause() == ["короткая фраза"]
    assert committer.flush_pause() == []


def test_finish_flushes_unstable_tail_and_resets_utterance() -> None:
    committer = StableTextCommitter()
    committer.observe("hello new")
    assert committer.finish("hello new world") == ["hello new world"]
    assert committer.observe("другая реплика") == []
    assert committer.finish("") == ["другая реплика"]


def test_empty_and_punctuation_only_hypotheses_are_never_spoken() -> None:
    committer = StableTextCommitter()
    committer.observe("полезный текст")
    assert committer.observe("") == []
    assert committer.flush_pause() == ["полезный текст"]
    assert committer.finish("... !!!") == []


def test_repeated_words_are_preserved_once_by_position() -> None:
    committer = StableTextCommitter()
    text = "да да это действительно работает сейчас"
    committer.observe(text)
    assert committer.observe(text) == [text]
    assert committer.finish(text) == []


def test_whitespace_is_normalized_without_changing_punctuation_or_case() -> None:
    committer = StableTextCommitter()
    committer.observe("Hello,   WORLD!  It's fine")
    assert committer.finish("Hello,   WORLD!  It's fine") == [
        "Hello, WORLD! It's fine"
    ]


@pytest.mark.parametrize(
    ("profile_id", "factory_name"),
    [
        ("sherpa_streaming_ru_t_one", "from_t_one_ctc"),
        ("sherpa_streaming_en_zipformer_20m", "from_transducer"),
    ],
)
def test_sherpa_adapter_uses_profile_factory_and_decodes_ready_frames(
    tmp_path: Path,
    profile_id: str,
    factory_name: str,
) -> None:
    profile = STREAMING_MODEL_PROFILES[profile_id]
    for relative in profile.required_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    online_recognizer = MagicMock()
    native_recognizer = MagicMock()
    native_stream = MagicMock()
    native_recognizer.create_stream.return_value = native_stream
    native_recognizer.is_ready.side_effect = [True, False]
    native_recognizer.get_result.return_value = SimpleNamespace(text=" привет ")
    native_recognizer.is_endpoint.return_value = True
    online_recognizer.from_t_one_ctc.return_value = native_recognizer
    online_recognizer.from_transducer.return_value = native_recognizer
    sherpa_module = SimpleNamespace(OnlineRecognizer=online_recognizer)

    transcriber = SherpaStreamingTranscriber(
        profile,
        tmp_path,
        sherpa_module=sherpa_module,
        num_threads=2,
    )
    result = transcriber.accept_audio(np.zeros(480, np.float32), 16000)

    assert result == StreamingResult("привет", True)
    getattr(online_recognizer, factory_name).assert_called_once()
    native_stream.accept_waveform.assert_called_once()
    native_recognizer.decode_stream.assert_called_once_with(native_stream)


def test_sherpa_adapter_uses_profile_endpoint_rules(tmp_path: Path) -> None:
    profile = dataclasses.replace(
        STREAMING_MODEL_PROFILES["sherpa_streaming_ru_t_one"],
        rule1_min_trailing_silence=1.1,
        rule2_min_trailing_silence=0.7,
        rule3_min_utterance_length=42.0,
    )
    for relative in profile.required_files:
        (tmp_path / relative).write_bytes(b"fixture")
    factory = MagicMock()
    factory.return_value.create_stream.return_value = MagicMock()
    sherpa_module = SimpleNamespace(
        OnlineRecognizer=SimpleNamespace(from_t_one_ctc=factory)
    )

    SherpaStreamingTranscriber(profile, tmp_path, sherpa_module=sherpa_module)

    assert factory.call_args.kwargs["rule1_min_trailing_silence"] == 1.1
    assert factory.call_args.kwargs["rule2_min_trailing_silence"] == 0.7
    assert factory.call_args.kwargs["rule3_min_utterance_length"] == 42.0


def test_sherpa_adapter_accepts_device_rate_then_finishes_and_resets(
    tmp_path: Path,
) -> None:
    profile = STREAMING_MODEL_PROFILES["sherpa_streaming_ru_t_one"]
    for relative in profile.required_files:
        (tmp_path / relative).write_bytes(b"fixture")
    online_recognizer = MagicMock()
    native_recognizer = MagicMock()
    first_stream = MagicMock()
    second_stream = MagicMock()
    native_recognizer.create_stream.side_effect = [first_stream, second_stream]
    native_recognizer.is_ready.side_effect = [False, False, False]
    native_recognizer.get_result.return_value = SimpleNamespace(text="готово")
    online_recognizer.from_t_one_ctc.return_value = native_recognizer
    transcriber = SherpaStreamingTranscriber(
        profile,
        tmp_path,
        sherpa_module=SimpleNamespace(OnlineRecognizer=online_recognizer),
    )

    transcriber.accept_audio(np.zeros(1, np.float32), 44100)
    first_stream.accept_waveform.assert_called_once()
    assert first_stream.accept_waveform.call_args.args[0] == 44100
    assert transcriber.finish() == StreamingResult("готово", True)
    first_stream.input_finished.assert_called_once_with()
    transcriber.reset()
    transcriber.accept_audio(np.zeros(1, np.float32), 16000)
    second_stream.accept_waveform.assert_called_once()
