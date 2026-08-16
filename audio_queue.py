import multiprocessing
import queue
import threading
import uuid
from typing import Callable, Optional

from pipeline import RunConfig, pipeline_process_main, prepare_audio_for_output


EventCallback = Callable[[str, str, str], None]
ScheduleCallback = Callable[[int, Callable[[], None]], object]


def _schedule_with_timer(
    delay_ms: int,
    callback: Callable[[], None],
) -> threading.Timer:
    timer = threading.Timer(delay_ms / 1000.0, callback)
    timer.daemon = True
    timer.start()
    return timer


class SpeechLoopRunner:
    def __init__(
        self,
        config: RunConfig,
        on_event: EventCallback,
        *,
        on_stopped: Optional[Callable[[], None]] = None,
        schedule: Optional[ScheduleCallback] = None,
        mp_context=None,
    ):
        self.config = config
        self.on_event = on_event
        self.on_stopped = on_stopped or (lambda: None)
        self._schedule = schedule or _schedule_with_timer
        self._context = mp_context or multiprocessing.get_context("spawn")

        self._process = None
        self._cancel_event = None
        self._event_queue = None
        self._active_run_id: Optional[str] = None
        self._event_run_id: Optional[str] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def active_run_id(self) -> Optional[str]:
        with self._lock:
            return self._active_run_id

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return

            run_id = uuid.uuid4().hex
            cancel_event = self._context.Event()
            event_queue = self._context.Queue()
            process = self._context.Process(
                target=pipeline_process_main,
                args=(run_id, self.config, cancel_event, event_queue),
                name=f"V2TTS-pipeline-{run_id[:8]}",
            )
            self._active_run_id = run_id
            self._event_run_id = run_id
            self._cancel_event = cancel_event
            self._event_queue = event_queue
            self._process = process

        try:
            process.start()
        except Exception:
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._cancel_event = None
                    self._event_queue = None
                    self._active_run_id = None
                    self._event_run_id = None
            try:
                event_queue.close()
            except Exception:
                pass
            raise

        monitor = threading.Thread(
            target=self._monitor_worker,
            args=(process, event_queue, run_id),
            name=f"V2TTS-monitor-{run_id[:8]}",
            daemon=True,
        )
        self._monitor_thread = monitor
        monitor.start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            cancel_event = self._cancel_event
            if process is None:
                return
            self._active_run_id = None
            self._event_run_id = None

        if cancel_event is not None:
            cancel_event.set()
        self._schedule(300, lambda: self._terminate_if_alive(process))

    def is_running(self) -> bool:
        with self._lock:
            process = self._process
        return bool(process is not None and process.is_alive())

    def accepts_events_from(self, run_id: str) -> bool:
        with self._lock:
            return run_id == self._event_run_id

    def _terminate_if_alive(self, process) -> None:
        with self._lock:
            is_current = process is self._process
        if not is_current or not process.is_alive():
            return
        process.terminate()
        self._schedule(1000, lambda: self._kill_if_alive(process))

    def _kill_if_alive(self, process) -> None:
        with self._lock:
            is_current = process is self._process
        if is_current and process.is_alive():
            process.kill()

    def _dispatch_event(self, event) -> None:
        run_id, kind, payload = event
        with self._lock:
            if run_id != self._event_run_id:
                return

        if kind in {"status", "text", "error", "warning", "state"}:
            self.on_event(run_id, kind, payload)

    def _monitor_worker(self, process, event_queue, run_id: str) -> None:
        while True:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            else:
                self._dispatch_event(event)

            if not process.is_alive():
                break

        while True:
            try:
                self._dispatch_event(event_queue.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError):
                break

        process.join()
        self._report_abnormal_exit(process, run_id)
        self._finish_worker(process, event_queue)

    def _report_abnormal_exit(self, process, run_id: str) -> None:
        exitcode = process.exitcode
        if exitcode in {None, 0}:
            return
        self._dispatch_event(
            (
                run_id,
                "error",
                f"Pipeline worker exited unexpectedly with code {exitcode}",
            )
        )

    def _finish_worker(self, process, event_queue) -> None:
        with self._lock:
            if process is not self._process:
                return
            self._process = None
            self._cancel_event = None
            self._event_queue = None
            self._active_run_id = None
            self._monitor_thread = None

        try:
            event_queue.close()
            event_queue.join_thread()
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass
        self.on_stopped()


__all__ = ["RunConfig", "SpeechLoopRunner", "prepare_audio_for_output"]
