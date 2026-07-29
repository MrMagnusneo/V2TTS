import numpy as np
import pytest
from stt import WhisperTranscriber


def test_resample_audio_invalid_rates():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Sample rates must be positive."):
        WhisperTranscriber._resample_audio(audio, -16000, 16000)

    with pytest.raises(ValueError, match="Sample rates must be positive."):
        WhisperTranscriber._resample_audio(audio, 16000, 0)


def test_resample_audio_same_rate():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 16000)

    # Check that identity is maintained (the exact same array is returned)
    assert result is audio
    np.testing.assert_array_equal(result, audio)


def test_resample_audio_empty_array():
    audio = np.array([], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 8000)

    # Check that identity is maintained
    assert result is audio
    np.testing.assert_array_equal(result, audio)


def test_resample_audio_upsample():
    audio = np.array([0.0, 1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 1, 2)

    # src_len = 2, dst_len = 4
    # src_x = [0.0, 0.5]
    # dst_x = [0.0, 0.25, 0.5, 0.75]
    # np.interp(dst_x, src_x, audio) -> [0.0, 0.5, 1.0, 1.0]
    expected = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_resample_audio_downsample():
    audio = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 4, 2)

    # src_len = 4, dst_len = 2
    # src_x = [0.0, 0.25, 0.5, 0.75]
    # dst_x = [0.0, 0.5]
    # np.interp([0.0, 0.5], src_x, audio) -> [0.0, 1.0]
    expected = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_resample_audio_edge_case_small_dst_len():
    # If src_len * dst_rate / src_rate < 0.5, dst_len becomes 0
    # This triggers max(1, 0) -> 1
    audio = np.array([1.0], dtype=np.float32)
    result = WhisperTranscriber._resample_audio(audio, 16000, 1) # dst_len = round(1 * 1 / 16000) = 0

    assert len(result) == 1
    np.testing.assert_array_equal(result, audio)
