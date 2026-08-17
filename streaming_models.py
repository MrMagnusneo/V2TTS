from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from stt_profiles import user_data_root


COMPLETE_MARKER = ".v2tts-complete"
ProgressCallback = Callable[[int, int], None]


class ModelDownloadCancelled(RuntimeError):
    pass


def _check_cancelled(stop_event) -> None:
    if stop_event is not None and stop_event.is_set():
        raise ModelDownloadCancelled("Streaming model download cancelled")


@contextmanager
def _profile_install_lock(parent: Path, profile_id: str, stop_event):
    lock_path = parent / f".{profile_id}.install.lock"
    with lock_path.open("a+b") as lock_file:
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        acquired = False
        try:
            while not acquired:
                _check_cancelled(stop_event)
                try:
                    if os.name == "nt":
                        import msvcrt

                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(
                            lock_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    acquired = True
                except (BlockingIOError, OSError):
                    time.sleep(0.05)
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class StreamingModelProfile:
    profile_id: str
    language: str
    architecture: str
    url: str
    archive_size: int
    sha256: str
    archive_root: str
    required_files: tuple[str, ...]
    sample_rate: int
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 300.0


STREAMING_MODEL_PROFILES: dict[str, StreamingModelProfile] = {
    "sherpa_streaming_ru_t_one": StreamingModelProfile(
        profile_id="sherpa_streaming_ru_t_one",
        language="ru",
        architecture="t_one_ctc",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/sherpa-onnx-streaming-t-one-russian-"
            "2025-09-08.tar.bz2"
        ),
        archive_size=128468156,
        sha256=(
            "b9c907450e99a6e5049e279bf18368a17db0bdc5e63b7fa978943138debbe3ae"
        ),
        archive_root="sherpa-onnx-streaming-t-one-russian-2025-09-08",
        required_files=("model.onnx", "tokens.txt"),
        sample_rate=8000,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300.0,
    ),
    "sherpa_streaming_en_zipformer_20m": StreamingModelProfile(
        profile_id="sherpa_streaming_en_zipformer_20m",
        language="en",
        architecture="transducer",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/sherpa-onnx-streaming-zipformer-en-20M-"
            "2023-02-17.tar.bz2"
        ),
        archive_size=127887156,
        sha256=(
            "9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9"
        ),
        archive_root="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        required_files=(
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ),
        sample_rate=16000,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300.0,
    ),
}


def streaming_model_dir(
    profile_id: str,
    root: Path | None = None,
) -> Path:
    base = (
        Path(root)
        if root is not None
        else user_data_root() / "models" / "sherpa-onnx"
    )
    return base / profile_id


def _validate_installed(
    profile: StreamingModelProfile,
    destination: Path,
) -> bool:
    return (
        (destination / COMPLETE_MARKER).is_file()
        and all((destination / name).is_file() for name in profile.required_files)
    )


def is_streaming_model_ready(
    profile_id: str,
    root: Path | None = None,
) -> bool:
    profile = STREAMING_MODEL_PROFILES[profile_id]
    return _validate_installed(profile, streaming_model_dir(profile_id, root))


def _copy_and_verify(
    profile: StreamingModelProfile,
    source: BinaryIO,
    archive_path: Path,
    on_progress: ProgressCallback,
    stop_event=None,
) -> None:
    digest = hashlib.sha256()
    completed = 0
    with archive_path.open("wb") as output:
        while True:
            _check_cancelled(stop_event)
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            completed += len(chunk)
            on_progress(completed, profile.archive_size)
        output.flush()
        os.fsync(output.fileno())

    if completed != profile.archive_size:
        raise ValueError(
            f"Streaming model archive size mismatch: expected "
            f"{profile.archive_size}, got {completed}"
        )
    actual_digest = digest.hexdigest()
    if actual_digest != profile.sha256:
        raise ValueError(
            f"Streaming model archive SHA-256 mismatch: expected "
            f"{profile.sha256}, got {actual_digest}"
        )


def _extract_safely(
    archive_path: Path,
    extraction_root: Path,
    stop_event=None,
) -> None:
    resolved_root = extraction_root.resolve()
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        while True:
            _check_cancelled(stop_event)
            member = archive.next()
            if member is None:
                break
            if member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
            target = (extraction_root / member.name).resolve()
            if not target.is_relative_to(resolved_root):
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not extract archive member: {member.name}")
            with source, target.open("wb") as output:
                while True:
                    _check_cancelled(stop_event)
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)


def _replace_directory_atomically(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.invalid")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def install_streaming_model_archive(
    profile: StreamingModelProfile,
    source: BinaryIO,
    *,
    root: Path | None = None,
    on_progress: ProgressCallback,
    stop_event=None,
) -> Path:
    destination = streaming_model_dir(profile.profile_id, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{profile.profile_id}.partial-",
            dir=destination.parent,
        )
    )
    archive_path = work_dir / "model.tar.bz2.partial"
    extraction_root = work_dir / "extracted"
    try:
        _copy_and_verify(
            profile,
            source,
            archive_path,
            on_progress,
            stop_event,
        )
        _check_cancelled(stop_event)
        extraction_root.mkdir()
        _extract_safely(archive_path, extraction_root, stop_event)
        model_root = extraction_root / profile.archive_root
        missing = [
            name for name in profile.required_files if not (model_root / name).is_file()
        ]
        if missing:
            raise ValueError(
                "Streaming model archive is missing required files: "
                + ", ".join(missing)
            )
        (model_root / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        _replace_directory_atomically(model_root, destination)
        return destination
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _cleanup_abandoned_workspaces(parent: Path, profile_id: str) -> None:
    for path in parent.glob(f".{profile_id}.download-*.partial"):
        path.unlink(missing_ok=True)
    for path in parent.glob(f".{profile_id}.partial-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def ensure_streaming_model(
    profile_id: str,
    stop_event,
    on_progress: ProgressCallback,
    root: Path | None = None,
) -> Path:
    profile = STREAMING_MODEL_PROFILES[profile_id]
    destination = streaming_model_dir(profile_id, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _profile_install_lock(destination.parent, profile_id, stop_event):
        if _validate_installed(profile, destination):
            return destination
        _cleanup_abandoned_workspaces(destination.parent, profile_id)

        descriptor, download_name = tempfile.mkstemp(
            prefix=f".{profile_id}.download-",
            suffix=".partial",
            dir=destination.parent,
        )
        os.close(descriptor)
        download_path = Path(download_name)
        completed = 0
        try:
            with urllib.request.urlopen(profile.url, timeout=30) as response:
                with download_path.open("wb") as output:
                    while True:
                        _check_cancelled(stop_event)
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        completed += len(chunk)
                        on_progress(completed, profile.archive_size)
                    output.flush()
                    os.fsync(output.fileno())
            _check_cancelled(stop_event)
            with download_path.open("rb") as source:
                return install_streaming_model_archive(
                    profile,
                    source,
                    root=root,
                    on_progress=on_progress,
                    stop_event=stop_event,
                )
        finally:
            download_path.unlink(missing_ok=True)
