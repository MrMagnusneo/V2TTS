import dataclasses
import hashlib
import io
import tarfile
import threading
import time
from pathlib import Path

import pytest

from streaming_models import (
    STREAMING_MODEL_PROFILES,
    ModelDownloadCancelled,
    StreamingModelProfile,
    _extract_safely,
    ensure_streaming_model,
    install_streaming_model_archive,
    is_streaming_model_ready,
)


def _archive(files: dict[str, bytes], root: str = "fixture") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _profile(payload: bytes, required=("model.onnx", "tokens.txt")):
    return StreamingModelProfile(
        profile_id="fixture",
        language="ru",
        architecture="t_one_ctc",
        url="https://invalid.example/model.tar.bz2",
        archive_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        archive_root="fixture",
        required_files=required,
        sample_rate=16000,
    )


def test_manifest_pins_official_models() -> None:
    ru = STREAMING_MODEL_PROFILES["sherpa_streaming_ru_t_one"]
    en = STREAMING_MODEL_PROFILES["sherpa_streaming_en_zipformer_20m"]
    assert (ru.archive_size, ru.sha256, ru.required_files) == (
        128468156,
        "b9c907450e99a6e5049e279bf18368a17db0bdc5e63b7fa978943138debbe3ae",
        ("model.onnx", "tokens.txt"),
    )
    assert ru.sample_rate == 8000
    assert (en.archive_size, en.sha256, en.required_files) == (
        127887156,
        "9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9",
        (
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ),
    )
    assert en.sample_rate == 16000


def test_install_marks_only_a_verified_complete_model(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _archive({"model.onnx": b"model", "tokens.txt": b"tokens"})
    profile = _profile(payload)
    monkeypatch.setitem(STREAMING_MODEL_PROFILES, profile.profile_id, profile)

    installed = install_streaming_model_archive(
        profile,
        io.BytesIO(payload),
        root=tmp_path,
        on_progress=lambda *_: None,
    )

    assert (installed / "model.onnx").read_bytes() == b"model"
    assert (installed / ".v2tts-complete").is_file()
    assert is_streaming_model_ready("fixture", root=tmp_path)
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize(
    "changed, message",
    [
        ({"archive_size": 1}, "size"),
        ({"sha256": "0" * 64}, "SHA-256"),
    ],
)
def test_install_rejects_wrong_archive_metadata(
    tmp_path: Path, changed: dict, message: str
) -> None:
    payload = _archive({"model.onnx": b"model", "tokens.txt": b"tokens"})
    profile = dataclasses.replace(_profile(payload), **changed)

    with pytest.raises(ValueError, match=message):
        install_streaming_model_archive(
            profile, io.BytesIO(payload), root=tmp_path, on_progress=lambda *_: None
        )

    assert not list(tmp_path.rglob(".v2tts-complete"))


def test_install_rejects_missing_file_and_path_traversal(tmp_path: Path) -> None:
    missing = _archive({"tokens.txt": b"tokens"})
    with pytest.raises(ValueError, match="model.onnx"):
        install_streaming_model_archive(
            _profile(missing),
            io.BytesIO(missing),
            root=tmp_path,
            on_progress=lambda *_: None,
        )

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        info = tarfile.TarInfo("fixture/../../escape.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    traversal = output.getvalue()
    with pytest.raises(ValueError, match="unsafe archive member"):
        install_streaming_model_archive(
            _profile(traversal, required=()),
            io.BytesIO(traversal),
            root=tmp_path,
            on_progress=lambda *_: None,
        )
    assert not (tmp_path.parent / "escape.txt").exists()


def test_ensure_download_is_cancellable_and_preserves_ready_model(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _archive({"model.onnx": b"model", "tokens.txt": b"tokens"})
    profile = _profile(payload)
    monkeypatch.setitem(STREAMING_MODEL_PROFILES, profile.profile_id, profile)

    class CancelAfterFirstRead(io.BytesIO):
        def read(self, size=-1):
            data = super().read(1 if size != 0 else size)
            stop.set()
            return data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    stop = threading.Event()
    monkeypatch.setattr(
        "streaming_models.urllib.request.urlopen",
        lambda *args, **kwargs: CancelAfterFirstRead(payload),
    )
    with pytest.raises(ModelDownloadCancelled):
        ensure_streaming_model(
            "fixture", stop_event=stop, on_progress=lambda *_: None, root=tmp_path
        )
    assert not list(tmp_path.rglob(".v2tts-complete"))

    stop.clear()
    install_streaming_model_archive(
        profile, io.BytesIO(payload), root=tmp_path, on_progress=lambda *_: None
    )
    monkeypatch.setattr(
        "streaming_models.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("ready model was downloaded again"),
    )
    assert ensure_streaming_model(
        "fixture", stop_event=stop, on_progress=lambda *_: None, root=tmp_path
    ).is_dir()


def test_install_cancels_during_extraction_and_removes_workspace(
    tmp_path: Path,
) -> None:
    payload = _archive(
        {
            "model.onnx": b"model" * 100,
            "tokens.txt": b"tokens",
        }
    )
    profile = _profile(payload)
    class CancelDuringExtraction:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 5

    stop = CancelDuringExtraction()

    with pytest.raises(ModelDownloadCancelled):
        install_streaming_model_archive(
            profile,
            io.BytesIO(payload),
            root=tmp_path,
            on_progress=lambda *_: None,
            stop_event=stop,
        )

    assert not list(tmp_path.glob(".fixture.partial-*"))
    assert not list(tmp_path.rglob(".v2tts-complete"))


def test_ensure_removes_abandoned_workspaces_before_new_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = _archive({"model.onnx": b"model", "tokens.txt": b"tokens"})
    profile = _profile(payload)
    monkeypatch.setitem(STREAMING_MODEL_PROFILES, profile.profile_id, profile)
    abandoned_download = tmp_path / ".fixture.download-abandoned.partial"
    abandoned_download.write_bytes(b"partial")
    abandoned_extract = tmp_path / ".fixture.partial-abandoned"
    abandoned_extract.mkdir()
    (abandoned_extract / "partial").write_bytes(b"partial")
    monkeypatch.setattr(
        "streaming_models.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    ensure_streaming_model(
        "fixture",
        stop_event=threading.Event(),
        on_progress=lambda *_: None,
        root=tmp_path,
    )

    assert not abandoned_download.exists()
    assert not abandoned_extract.exists()


def test_concurrent_ensure_serializes_one_profile_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = _archive({"model.onnx": b"model", "tokens.txt": b"tokens"})
    profile = _profile(payload)
    monkeypatch.setitem(STREAMING_MODEL_PROFILES, profile.profile_id, profile)
    first_read = threading.Event()
    release = threading.Event()
    calls = []

    class BlockingResponse(io.BytesIO):
        def __init__(self, data: bytes, block: bool) -> None:
            super().__init__(data)
            self.block = block

        def read(self, size=-1):
            if self.block:
                self.block = False
                first_read.set()
                assert release.wait(timeout=2)
            return super().read(size)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(*_args, **_kwargs):
        calls.append(object())
        return BlockingResponse(payload, block=len(calls) == 1)

    monkeypatch.setattr("streaming_models.urllib.request.urlopen", urlopen)
    results = []
    errors = []

    def install() -> None:
        try:
            results.append(
                ensure_streaming_model(
                    "fixture",
                    stop_event=threading.Event(),
                    on_progress=lambda *_: None,
                    root=tmp_path,
                )
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=install)
    second = threading.Thread(target=install)
    first.start()
    assert first_read.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    calls_while_first_is_active = len(calls)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert calls_while_first_is_active == 1
    assert len(calls) == 1
    assert len(results) == 2
    assert errors == []


def test_extraction_traversal_checks_cancellation_incrementally(
    tmp_path: Path,
    monkeypatch,
) -> None:
    member = tarfile.TarInfo("fixture")
    member.type = tarfile.DIRTYPE

    class StopDuringTraversal:
        cancelled = False

        def is_set(self) -> bool:
            return self.cancelled

    stop = StopDuringTraversal()

    class FakeArchive:
        materialized = False
        yielded = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def getmembers(self):
            self.materialized = True
            return [member]

        def next(self):
            if self.yielded:
                return None
            self.yielded = True
            stop.cancelled = True
            return member

    archive = FakeArchive()
    monkeypatch.setattr("streaming_models.tarfile.open", lambda *_args, **_kwargs: archive)

    with pytest.raises(ModelDownloadCancelled):
        _extract_safely(tmp_path / "model.tar.bz2", tmp_path / "out", stop)

    assert archive.materialized is False
