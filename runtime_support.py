import os
import sys
from typing import TextIO


_fallback_streams: list[TextIO] = []


def ensure_standard_streams() -> None:
    """Provide writable streams in PyInstaller's windowed mode."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, name, stream)
        _fallback_streams.append(stream)
