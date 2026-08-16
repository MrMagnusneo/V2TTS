import threading
import queue
from unittest.mock import MagicMock, patch

import audio_queue
from audio_queue import SpeechLoopRunner
from gui import AppGUI
from pipeline import RunConfig
from stt_profiles import STTSelection


class Scheduler:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, delay_ms, callback):
        self.calls.append((delay_ms, callback))


def config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=7,
        stt=STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )


def build_runner(context, scheduler, events) -> SpeechLoopRunner:
    return SpeechLoopRunner(
        config(),
        on_event=lambda run_id, kind, message: events.append(
            (run_id, kind, message)
        ),
        on_stopped=lambda: events.append(("stopped", "")),
        schedule=scheduler,
        mp_context=context,
    )


def attach_live_process(runner: SpeechLoopRunner):
    process = MagicMock()
    process.is_alive.return_value = True
    cancel_event = MagicMock()
    runner._process = process
    runner._cancel_event = cancel_event
    runner._active_run_id = "run-1"
    runner._event_run_id = "run-1"
    return process, cancel_event


def test_stop_never_calls_portaudio_and_schedules_escalation() -> None:
    scheduler = Scheduler()
    events = []
    runner = build_runner(MagicMock(), scheduler, events)
    _, cancel_event = attach_live_process(runner)

    with patch.object(
        audio_queue,
        "get_sounddevice",
        create=True,
    ) as get_sounddevice:
        runner.stop()

    get_sounddevice.assert_not_called()
    cancel_event.set.assert_called_once()
    assert runner.active_run_id is None
    assert scheduler.calls[0][0] == 300
    assert events == []


def test_stale_run_events_are_discarded() -> None:
    events = []
    runner = build_runner(MagicMock(), Scheduler(), events)
    runner._active_run_id = "new-run"

    runner._dispatch_event(("old-run", "text", "must not appear"))
    runner._dispatch_event(("old-run", "error", "must not appear"))

    assert events == []


def test_current_run_state_event_is_dispatched() -> None:
    events = []
    runner = build_runner(MagicMock(), Scheduler(), events)
    runner._active_run_id = "run-1"
    runner._event_run_id = "run-1"

    runner._dispatch_event(("run-1", "state", "listening"))

    assert events == [("run-1", "state", "listening")]


def test_abnormal_worker_exit_is_reported_for_current_run() -> None:
    events = []
    runner = build_runner(MagicMock(), Scheduler(), events)
    process, _ = attach_live_process(runner)
    process.exitcode = -1073741819

    runner._report_abnormal_exit(process, "run-1")

    assert events == [
        (
            "run-1",
            "error",
            "Pipeline worker exited unexpectedly with code -1073741819",
        )
    ]


def test_abnormal_worker_exit_is_suppressed_after_stop() -> None:
    events = []
    runner = build_runner(MagicMock(), Scheduler(), events)
    process, _ = attach_live_process(runner)
    process.exitcode = 1

    runner.stop()
    runner._report_abnormal_exit(process, "run-1")

    assert events == []


def test_dispatch_stop_race_keeps_run_id_on_late_delivery() -> None:
    entered = threading.Event()
    release = threading.Event()
    events = []

    def delayed_delivery(run_id, kind, message):
        entered.set()
        assert release.wait(timeout=1)
        events.append((run_id, kind, message))

    runner = SpeechLoopRunner(
        config(),
        on_event=delayed_delivery,
        schedule=Scheduler(),
        mp_context=MagicMock(),
    )
    attach_live_process(runner)
    dispatch = threading.Thread(
        target=runner._dispatch_event,
        args=(("run-1", "text", "late result"),),
    )

    dispatch.start()
    assert entered.wait(timeout=1)
    runner.stop()
    release.set()
    dispatch.join(timeout=1)

    assert events == [("run-1", "text", "late result")]
    assert runner.active_run_id is None


def test_completed_run_events_remain_deliverable_until_ui_stops_runner() -> None:
    runner = build_runner(MagicMock(), Scheduler(), [])
    process, _ = attach_live_process(runner)
    event_queue = MagicMock()

    runner._finish_worker(process, event_queue)

    assert runner.active_run_id is None
    assert runner.accepts_events_from("run-1")


def test_fatal_error_is_rendered_after_worker_finishes() -> None:
    runner = build_runner(MagicMock(), Scheduler(), [])
    process, _ = attach_live_process(runner)
    runner._finish_worker(process, MagicMock())
    gui = object.__new__(AppGUI)
    gui.ui_queue = queue.Queue()
    gui.is_run_current = runner.accepts_events_from
    gui.on_worker_stopped = MagicMock(return_value=False)
    gui.root = MagicMock()
    gui._append_log = MagicMock()
    gui.ui_queue.put(("run-1", "error", "model load failed"))
    gui.ui_queue.put((None, "worker_stopped", ""))

    with patch("gui.messagebox.showerror") as showerror:
        gui._poll_ui_queue()

    gui._append_log.assert_called_once_with("ERROR: model load failed\n")
    showerror.assert_called_once_with("Runtime error", "model load failed")


def test_escalation_terminates_then_kills_only_child() -> None:
    scheduler = Scheduler()
    runner = build_runner(MagicMock(), scheduler, [])
    process, _ = attach_live_process(runner)

    runner.stop()
    scheduler.calls.pop(0)[1]()

    process.terminate.assert_called_once()
    assert scheduler.calls[0][0] == 1000

    scheduler.calls.pop(0)[1]()
    process.kill.assert_called_once()


def test_start_does_not_create_second_live_worker() -> None:
    context = MagicMock()
    runner = build_runner(context, Scheduler(), [])
    attach_live_process(runner)

    runner.start()

    context.Process.assert_not_called()
