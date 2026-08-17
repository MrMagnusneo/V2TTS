from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from streaming_models import StreamingModelProfile


_CLAUSE_END = re.compile(r"[,;:!?…]|[.—–-]$")


@dataclass(frozen=True)
class StreamingResult:
    text: str
    endpoint: bool


class OnlineTranscriber(Protocol):
    def accept_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult: ...

    def finish(self) -> StreamingResult: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class SherpaStreamingTranscriber:
    def __init__(
        self,
        profile: StreamingModelProfile,
        model_dir: Path,
        *,
        sherpa_module=None,
        num_threads: int | None = None,
    ) -> None:
        if sherpa_module is None:
            import sherpa_onnx as sherpa_module

        self.profile = profile
        self.model_dir = Path(model_dir)
        self._num_threads = num_threads or max(
            1,
            min(4, os.cpu_count() or 1),
        )
        common = {
            "tokens": str(self.model_dir / "tokens.txt"),
            "num_threads": self._num_threads,
            "sample_rate": profile.sample_rate,
            "feature_dim": 80,
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": profile.rule1_min_trailing_silence,
            "rule2_min_trailing_silence": profile.rule2_min_trailing_silence,
            "rule3_min_utterance_length": profile.rule3_min_utterance_length,
            "decoding_method": "greedy_search",
        }
        if profile.architecture == "t_one_ctc":
            self._recognizer = sherpa_module.OnlineRecognizer.from_t_one_ctc(
                model=str(self.model_dir / "model.onnx"),
                **common,
            )
        elif profile.architecture == "transducer":
            self._recognizer = sherpa_module.OnlineRecognizer.from_transducer(
                encoder=str(
                    self.model_dir / "encoder-epoch-99-avg-1.int8.onnx"
                ),
                decoder=str(
                    self.model_dir / "decoder-epoch-99-avg-1.int8.onnx"
                ),
                joiner=str(
                    self.model_dir / "joiner-epoch-99-avg-1.int8.onnx"
                ),
                **common,
            )
        else:
            raise ValueError(
                f"Unsupported streaming architecture: {profile.architecture}"
            )
        self._stream = self._recognizer.create_stream()

    def accept_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Streaming audio must be mono")
        self._stream.accept_waveform(sample_rate, audio)
        self._decode_ready()
        return StreamingResult(
            self._result_text(),
            bool(self._recognizer.is_endpoint(self._stream)),
        )

    def finish(self) -> StreamingResult:
        self._stream.input_finished()
        self._decode_ready()
        return StreamingResult(self._result_text(), True)

    def reset(self) -> None:
        self._stream = self._recognizer.create_stream()

    def close(self) -> None:
        self._stream = None

    def _decode_ready(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _result_text(self) -> str:
        result = self._recognizer.get_result(self._stream)
        return str(getattr(result, "text", result)).strip()


def _tokens(text: str) -> list[str]:
    return text.split()


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def _contains_word(tokens: list[str]) -> bool:
    return any(any(character.isalnum() for character in token) for token in tokens)


class StableTextCommitter:
    """Turn revisable online hypotheses into one-shot speakable chunks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: list[str] = []
        self._latest: list[str] = []
        self._stable_length = 0
        self._committed_length = 0

    def observe(self, text: str) -> list[str]:
        current = _tokens(text)
        if not current:
            return []
        common = _common_prefix_length(self._previous, current)
        self._latest = current
        self._stable_length = max(self._committed_length, common)
        self._previous = current
        return self._take_ready(force=False)

    def flush_pause(self) -> list[str]:
        self._stable_length = len(self._latest)
        return self._take_ready(force=True)

    def finish(self, text: str) -> list[str]:
        current = _tokens(text)
        if current:
            self._latest = current
        self._stable_length = len(self._latest)
        chunks = self._take_ready(force=True)
        self.reset()
        return chunks

    def _take_ready(self, *, force: bool) -> list[str]:
        chunks: list[str] = []
        available_end = min(self._stable_length, len(self._latest))

        while self._committed_length < available_end:
            pending = self._latest[self._committed_length : available_end]
            take = self._next_boundary(pending, force=force)
            if take == 0:
                break
            selected = pending[:take]
            self._committed_length += take
            if _contains_word(selected):
                chunks.append(" ".join(selected))

        return chunks

    @staticmethod
    def _next_boundary(tokens: list[str], *, force: bool) -> int:
        for index, token in enumerate(tokens, start=1):
            if index >= 3 and _CLAUSE_END.search(token):
                return min(index, 8)
        if len(tokens) >= 6:
            return 6
        if force and tokens:
            return min(len(tokens), 8)
        return 0
