from __future__ import annotations


def get_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except OSError as exc:
        raise RuntimeError(
            "PortAudio library not found. Install PortAudio system package "
            "(Linux: libportaudio2/portaudio19-dev, Windows: bundled with python-sounddevice)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import sounddevice: {exc}") from exc
    return sd


def get_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Failed to import soundfile: {exc}") from exc
    return sf

