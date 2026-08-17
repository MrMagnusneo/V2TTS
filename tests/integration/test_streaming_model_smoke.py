import os
import threading
from pathlib import Path

import pytest

from audio_backend import get_soundfile
from streaming_models import (
    STREAMING_MODEL_PROFILES,
    ensure_streaming_model,
    is_streaming_model_ready,
    streaming_model_dir,
)
from streaming_stt import SherpaStreamingTranscriber


@pytest.mark.integration
def test_real_streaming_model_produces_text() -> None:
    if os.getenv("V2TTS_STREAMING_MODEL_TEST") != "1":
        pytest.skip("set V2TTS_STREAMING_MODEL_TEST=1")

    profile_id = os.getenv(
        "V2TTS_STREAMING_PROFILE",
        "sherpa_streaming_ru_t_one",
    )
    try:
        profile = STREAMING_MODEL_PROFILES[profile_id]
    except KeyError:
        pytest.fail(f"unknown streaming profile: {profile_id}")

    if is_streaming_model_ready(profile_id):
        model_dir = streaming_model_dir(profile_id)
    elif os.getenv("V2TTS_ALLOW_MODEL_DOWNLOAD") == "1":
        model_dir = ensure_streaming_model(
            profile_id,
            stop_event=threading.Event(),
            on_progress=lambda *_: None,
        )
    else:
        pytest.skip("install the model or set V2TTS_ALLOW_MODEL_DOWNLOAD=1")

    wav_value = os.getenv("V2TTS_STREAMING_TEST_WAV")
    if not wav_value:
        pytest.skip("set V2TTS_STREAMING_TEST_WAV to a speech WAV")
    wav_path = Path(wav_value)
    if not wav_path.is_file():
        pytest.fail(f"speech WAV does not exist: {wav_path}")

    soundfile = get_soundfile()
    audio, sample_rate = soundfile.read(wav_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, 0]

    transcriber = SherpaStreamingTranscriber(profile, model_dir)
    transcript = ""
    try:
        frame_size = max(1, int(sample_rate * 0.03))
        for offset in range(0, len(audio), frame_size):
            result = transcriber.accept_audio(
                audio[offset : offset + frame_size],
                sample_rate,
            )
            transcript = result.text or transcript
        transcript = transcriber.finish().text or transcript
    finally:
        transcriber.close()

    assert transcript.strip()
