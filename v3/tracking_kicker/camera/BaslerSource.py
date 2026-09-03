"""Process-isolated Basler/pypylon image source.

The parent process owns only a shared-memory ring and the source interface.
All pypylon calls run in :mod:`BaslerCaptureProcess`. This is important for
USB transport failures because pypylon can block while stopping a damaged
grab stream.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from ..Objects.Frames import Frame
from .BaslerCaptureProcess import run_basler_capture_process
from .Camera import Camera


class BaslerSource:
    """Acquire Basler frames through a killable child process."""

    RING_SLOTS = 4
    SLOT_BYTES_PER_PIXEL = 8
    STARTUP_TIMEOUT_SECONDS = 10.0
    # Give pylon/the USB transport a moment to release a failed stream before
    # the replacement process tries to open the same device again.
    RESTART_BACKOFF_SECONDS = 2.0
    FRAME_STALL_SECONDS = 1.0

    def __init__(self, cfg: Camera):
        if cfg is None:
            raise ValueError("Eine Kamerakonfiguration ist erforderlich")
        if importlib.util.find_spec("pypylon") is None:
            raise ModuleNotFoundError(
                "pypylon ist nicht installiert; Basler-Kamera kann nicht verwendet werden"
            )

        self.cfg = cfg
        self.fps = float(getattr(cfg, "force_fps", 0) or 0)
        # pypylon and OpenCV both own native state and worker threads.  A fork
        # can inherit that state in an unsafe half-initialized condition once
        # the analysis workers start.  Spawn gives the capture process a clean
        # interpreter on every platform.
        start_methods = mp.get_all_start_methods()
        method = "spawn" if "spawn" in start_methods else start_methods[0]
        self._ctx = mp.get_context(method)
        self._process_lock = threading.RLock()
        self._release_lock = threading.Lock()
        self._closed = False
        self._process = None
        self._original_affinity = None
        self._next_restart_time = 0.0
        self._last_frame_time = time.monotonic()
        self._last_sequence = 0
        self._last_error = None
        self._last_error_time = 0.0
        self._read_count = 0
        self._read_failures = 0
        self._read_log_time = time.perf_counter()

        width = max(1, int(cfg.width))
        height = max(1, int(cfg.height))
        self._slot_bytes = width * height * self.SLOT_BYTES_PER_PIXEL
        self._shared_buffer = self._ctx.RawArray(
            "B", self._slot_bytes * self.RING_SLOTS
        )
        self._slot_locks = [self._ctx.Lock() for _ in range(self.RING_SLOTS)]
        # Metadata is read/written while the corresponding slot lock is held.
        self._slot_sequences = self._ctx.Array(
            "q", [0] * self.RING_SLOTS, lock=False
        )
        self._slot_timestamps = self._ctx.Array(
            "d", [0.0] * self.RING_SLOTS, lock=False
        )
        self._slot_ndims = self._ctx.Array("i", [0] * self.RING_SLOTS, lock=False)
        self._slot_shapes0 = self._ctx.Array("i", [0] * self.RING_SLOTS, lock=False)
        self._slot_shapes1 = self._ctx.Array("i", [0] * self.RING_SLOTS, lock=False)
        self._slot_shapes2 = self._ctx.Array("i", [0] * self.RING_SLOTS, lock=False)
        self._slot_dtype_codes = self._ctx.Array(
            "i", [0] * self.RING_SLOTS, lock=False
        )
        self._latest_sequence = self._ctx.Value("q", 0)
        self._stop_event = self._ctx.Event()
        self._ready_event = self._ctx.Event()
        self._startup_state = self._ctx.Value("i", 0)
        self._fatal_state = self._ctx.Value("i", 0)
        self._fps_value = self._ctx.Value("d", self.fps)
        self._events = self._ctx.Queue(maxsize=32)

        try:
            self._start_process(wait=True)
        except Exception:
            self._closed = True
            self._stop_process()
            self._close_event_queue()
            self._restore_host_affinity()
            raise

    # ------------------------------------------------------------------
    # Child lifecycle
    # ------------------------------------------------------------------

    def _prepare_capture_cpu(self):
        """Temporarily add the reserved CPU while spawning the child."""
        if not getattr(self.cfg, "reserve_capture_cpu", True):
            return None
        capture_cpu = getattr(self.cfg, "capture_cpu", None)
        if capture_cpu is None or not hasattr(os, "sched_getaffinity"):
            return None
        try:
            capture_cpu = int(capture_cpu)
            current = set(os.sched_getaffinity(0))
            expanded = current | {capture_cpu}
            if expanded != current:
                os.sched_setaffinity(0, expanded)
            if self._original_affinity is None:
                self._original_affinity = set(current)
            return current, capture_cpu
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _restore_parent_after_spawn(affinity_state) -> None:
        if affinity_state is None:
            return
        current, capture_cpu = affinity_state
        remaining = current - {capture_cpu}
        if not remaining:
            return
        try:
            os.sched_setaffinity(0, remaining)
        except OSError:
            pass

    def _restore_host_affinity(self) -> None:
        if self._original_affinity is None or not hasattr(os, "sched_setaffinity"):
            return
        try:
            os.sched_setaffinity(0, self._original_affinity)
        except OSError:
            pass

    def _start_process(self, *, wait: bool) -> None:
        with self._process_lock:
            if self._closed:
                return
            if self._process is not None and self._process.is_alive():
                return

            self._ready_event.clear()
            self._startup_state.value = 0
            self._fatal_state.value = 0
            process = self._ctx.Process(
                target=run_basler_capture_process,
                args=(
                    self.cfg,
                    self._shared_buffer,
                    self._slot_bytes,
                    self._slot_locks,
                    self._slot_sequences,
                    self._slot_timestamps,
                    self._slot_ndims,
                    self._slot_shapes0,
                    self._slot_shapes1,
                    self._slot_shapes2,
                    self._slot_dtype_codes,
                    self._latest_sequence,
                    self._stop_event,
                    self._ready_event,
                    self._startup_state,
                    self._fatal_state,
                    self._fps_value,
                    self._events,
                ),
                name="basler-capture-process",
                daemon=True,
            )
            affinity_state = self._prepare_capture_cpu()
            try:
                process.start()
            except Exception:
                if affinity_state is not None:
                    try:
                        os.sched_setaffinity(0, affinity_state[0])
                    except OSError:
                        pass
                raise
            self._restore_parent_after_spawn(affinity_state)
            self._process = process

        if not wait:
            return

        deadline = time.monotonic() + self.STARTUP_TIMEOUT_SECONDS
        while not self._ready_event.wait(timeout=0.1):
            process = self._process
            if process is None or not process.is_alive():
                break
            if time.monotonic() >= deadline:
                break

        self._drain_events()
        if self._startup_state.value != 1:
            error = self._last_error or "Basler-Prozess wurde nicht bereit"
            self._stop_process()
            raise RuntimeError(error)

        self._last_frame_time = time.monotonic()

    def _stop_process(
        self,
        timeout: float = 1.5,
        *,
        restart: bool = False,
    ) -> bool:
        """Stop the capture process without allowing pylon to block forever.

        A normal shutdown gets a chance to execute the child's
        ``StopGrabbing``/``Close`` cleanup.  If the transport layer is stuck,
        the bounded join expires and the parent terminates the child.  During
        recovery the shared stop event is cleared only after the old process
        is gone, so the replacement process can use the same event safely.

        Returns whether the process had to be force-terminated.
        """
        with self._process_lock:
            process = self._process
            self._process = None
        if process is None:
            if restart:
                self._stop_event.clear()
            return False

        forced = False
        try:
            if process.is_alive():
                self._stop_event.set()
            process.join(timeout=max(0.0, float(timeout)))
            if process.is_alive():
                forced = True
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                else:
                    process.terminate()
                process.join(timeout=0.5)
        finally:
            try:
                process.close()
            except (OSError, ValueError):
                pass
            if restart:
                self._stop_event.clear()
        if forced:
            print(
                "[CAM-RECOVER] Alter Capture-Prozess reagierte nicht; "
                "er wurde beendet"
            )
        return forced

    def _restart_process(self, reason: str) -> None:
        if self._closed:
            return
        now = time.monotonic()
        if now < self._next_restart_time:
            return

        print(f"[CAM-RECOVER] Basler-Prozess wird neu gestartet: {reason}")
        self._stop_process(restart=True)
        self._last_sequence = int(self._latest_sequence.value)
        self._last_frame_time = now
        self._next_restart_time = now + self.RESTART_BACKOFF_SECONDS
        self._start_process(wait=False)

    # ------------------------------------------------------------------
    # Shared-memory frame handoff
    # ------------------------------------------------------------------

    def _read_latest_frame(self):
        latest = int(self._latest_sequence.value)
        if latest <= self._last_sequence:
            return None

        first = max(self._last_sequence + 1, latest - self.RING_SLOTS + 1)
        for sequence in range(latest, first - 1, -1):
            slot = sequence % self.RING_SLOTS
            with self._slot_locks[slot]:
                if int(self._slot_sequences[slot]) != sequence:
                    continue

                ndim = int(self._slot_ndims[slot])
                shape0 = int(self._slot_shapes0[slot])
                shape1 = int(self._slot_shapes1[slot])
                shape2 = int(self._slot_shapes2[slot])
                dtype_code = int(self._slot_dtype_codes[slot])
                if ndim == 2 and shape0 > 0 and shape1 > 0:
                    shape = (shape0, shape1)
                elif ndim == 3 and shape0 > 0 and shape1 > 0 and shape2 > 0:
                    shape = (shape0, shape1, shape2)
                else:
                    self._last_sequence = sequence
                    return None

                if dtype_code == 1:
                    dtype = np.uint8
                elif dtype_code == 2:
                    dtype = np.uint16
                else:
                    self._last_sequence = sequence
                    return None

                required = int(np.prod(shape)) * np.dtype(dtype).itemsize
                if required > self._slot_bytes:
                    self._last_sequence = sequence
                    return None

                image = np.ndarray(
                    shape,
                    dtype=dtype,
                    buffer=self._shared_buffer,
                    offset=slot * self._slot_bytes,
                ).copy()
                timestamp = float(self._slot_timestamps[slot])

            self._last_sequence = sequence
            frame = Frame()
            frame.frame = image
            frame.timestamp = timestamp
            return frame

        # The child captured more frames than the ring can retain. Return the
        # newest one on the next call instead of replaying stale images.
        self._last_sequence = latest - 1
        return None

    # ------------------------------------------------------------------
    # Source interface
    # ------------------------------------------------------------------

    def _log_error(self, message: str) -> None:
        now = time.perf_counter()
        if message != self._last_error or now - self._last_error_time >= 1.0:
            print(f"[CAM-ERR] {message}")
            self._last_error = message
            self._last_error_time = now

    def _drain_events(self) -> None:
        while True:
            try:
                kind, message = self._events.get_nowait()
            except (queue.Empty, EOFError, OSError):
                return
            self._log_error(message)
            if kind == "fatal":
                self._fatal_state.value = 1

    def read(self):
        if self._closed:
            return False, None

        self._drain_events()
        process = self._process
        startup_state = int(self._startup_state.value)
        now = time.monotonic()

        if startup_state < 0 or process is None or not process.is_alive():
            self._restart_process("Capture-Prozess beendet")
            self._read_failures += 1
            return False, None
        if startup_state != 1:
            self._read_failures += 1
            return False, None

        frame = self._read_latest_frame()
        if frame is not None:
            self._read_count += 1
            self._read_failures = 0
            self._last_frame_time = now
        else:
            self._read_failures += 1
            if now - self._last_frame_time >= self.FRAME_STALL_SECONDS:
                reason = (
                    "dauerhafter Grab-/Read-Fehler"
                    if self._fatal_state.value
                    else "keine neuen Frames"
                )
                self._restart_process(reason)

        if now - self._read_log_time >= 1.0:
            elapsed = now - self._read_log_time
            print(
                f"[CAM-RAW] geliefert={self._read_count} "
                f"fps={self._read_count / elapsed:.1f} "
                f"fehlgeschlagen={self._read_failures}"
            )
            self._read_count = 0
            self._read_failures = 0
            self._read_log_time = now

        return (True, frame) if frame is not None else (False, None)

    def reset(self) -> None:
        """Keep the source interface; a live camera has no replay position."""

    def showFPS(self) -> str:
        value = float(self._fps_value.value)
        if value > 0:
            self.fps = value
        return f"{self.fps:.2f}"

    def get_temperature(self) -> Optional[str]:
        return None

    def getTemperatur(self, camera=None) -> Optional[str]:
        return self.get_temperature()

    def _close_event_queue(self) -> None:
        try:
            self._events.close()
            self._events.join_thread()
        except (AttributeError, OSError, ValueError):
            pass

    def release(self) -> None:
        with self._release_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()

        self._stop_process(timeout=1.5)
        self._close_event_queue()
        self._restore_host_affinity()
        print("[SHUTDOWN] Basler-Capture-Prozess beendet")

    close = release
