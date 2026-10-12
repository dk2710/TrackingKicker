"""Basler/pypylon image source.

Only this module knows about pypylon.  The rest of the application receives
plain :class:`~tracking_kicker.Objects.Frames.Frame` objects and can therefore
also run against the shared-memory :class:`camera.CameraProcessSource`.
"""

from __future__ import annotations

import ctypes
import gc
import multiprocessing
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np
from pypylon import pylon

from .Objects.Frames import Frame
from .camera.Camera import Camera


def _camera_capture_process_main(
    cfg: Camera,
    shared_memory_name: str,
    slot_bytes: int,
    slot_count: int,
    frame_sequence,
    frame_nbytes,
    frame_shape,
    last_frame_time,
    stop_event,
) -> None:
    """Own pypylon in a killable process and publish the latest frame."""
    shared = shared_memory.SharedMemory(name=shared_memory_name)
    source = None
    try:
        source = BaslerSource(cfg, _process_child=True)
        print("[CAM-PROC] Basler-Capture-Prozess gestartet", flush=True)

        while not stop_event.is_set():
            ok, frame = source.read()
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            image = np.ascontiguousarray(frame.frame)
            if image.dtype != np.uint8:
                image = image.astype(np.uint8, copy=False)
            raw = image.view(np.uint8).reshape(-1)
            nbytes = int(raw.nbytes)
            if nbytes > slot_bytes:
                print(
                    "[CAM-PROC] Frame ist groesser als der Shared-Memory-Slot: "
                    f"{nbytes} > {slot_bytes} Bytes",
                    flush=True,
                )
                continue

            next_sequence = int(frame_sequence.value) + 1
            slot = next_sequence % slot_count
            offset = slot * slot_bytes
            shared.buf[offset : offset + nbytes] = raw.tobytes()

            shape = tuple(int(value) for value in image.shape)
            frame_shape[0] = len(shape)
            for index in range(3):
                frame_shape[index + 1] = shape[index] if index < len(shape) else 0
            frame_nbytes.value = nbytes
            last_frame_time.value = time.monotonic()

            # Publish the sequence last. The parent uses this as the commit
            # marker and retries if a slot changes while it is being copied.
            frame_sequence.value = next_sequence
    except BaseException as exc:
        print(
            "[CAM-PROC] Capture-Prozess beendet: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        if source is not None:
            try:
                source.release()
            except BaseException as exc:
                print(
                    "[CAM-PROC] Kamera-Cleanup im Capture-Prozess fehlgeschlagen: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        try:
            shared.close()
        except Exception:
            pass


class BaslerSource:
    """Acquire frames from one Basler camera through pypylon.

    The source follows the pypylon user-allocated grab pattern used in the
    documentation: ``StartGrabbing`` is called once, each result is consumed
    with ``RetrieveResult`` and released in a ``finally`` block.  A serial
    number can be configured so a reconnect cannot accidentally select another
    USB camera.
    """

    RETRIEVE_TIMEOUT_MS = 100
    STARTUP_SETTLE_SECONDS = 2.0
    RECOVERY_FAILURE_THRESHOLD = 5
    RECOVERY_MIN_FAILURE_DURATION_SECONDS = 2.0
    RECOVERY_INTERVAL_SECONDS = 5.0
    DEVICE_ERROR_REPORT_LIMIT = 16
    PROCESS_SLOT_COUNT = 8
    PROCESS_START_TIMEOUT_SECONDS = 6.0
    PROCESS_STALL_TIMEOUT_SECONDS = 3.0
    PROCESS_STOP_TIMEOUT_SECONDS = 2.0

    def __init__(self, cfg: Camera, *, _process_child: bool = False):
        if cfg is None:
            raise ValueError("Eine Kamerakonfiguration ist erforderlich")

        self.cfg = cfg
        self._process_child = bool(_process_child)
        self._process_mode = not self._process_child
        self.camera: Optional[Any] = None
        self._camera_lock = threading.RLock()
        # pypylon's USB StopGrabbing() can block inside the transport layer.
        # Keep release() itself bounded so the application process can exit
        # and the kernel can tear down the USB handle.
        self._release_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._pending_cleanup: Optional[threading.Thread] = None
        self._closed = False
        self._stop_requested = False
        self._shutdown_incomplete = False

        self._process_context = None
        self._capture_process = None
        self._capture_stop_event = None
        self._capture_process_lock = threading.Lock()
        self._capture_process_started_at = 0.0
        self._last_process_restart = 0.0
        self._last_parent_sequence = 0
        self._shared = None
        self._shared_slot_bytes = 0
        self._shared_slot_count = self.PROCESS_SLOT_COUNT
        self._shared_sequence = None
        self._shared_frame_nbytes = None
        self._shared_frame_shape = None
        self._shared_last_frame_time = None

        self._read_count = 0
        self._read_failures = 0
        self._read_log_time = time.perf_counter()
        self._last_grab_error: Optional[str] = None
        self._last_grab_error_time = 0.0
        self._consecutive_grab_failures = 0
        self._first_grab_failure_time = 0.0
        self._last_recovery_time = 0.0
        self.fps = float(getattr(cfg, "force_fps", 0) or 0)

        try:
            if self._process_child:
                self.camera = self._open_camera()
            else:
                self._start_capture_process()
                self._wait_for_first_process_frame()
        except Exception as exc:
            # Open() may have succeeded before a feature or ROI failed.  The
            # partially initialized camera must still be closed here.
            print(
                "[CAM-ERR] BaslerSource konnte die Kamera nicht starten: "
                f"{type(exc).__name__}: {exc}"
            )
            if self._process_mode:
                self._release_capture_process()
            self.camera = None
            raise

    # ------------------------------------------------------------------
    # Isolated capture process
    # ------------------------------------------------------------------

    def _start_capture_process(self) -> None:
        """Create the shared-memory ring and start the pylon child process."""
        self._process_context = multiprocessing.get_context("spawn")
        self._shared_slot_bytes = max(
            1,
            int(self.cfg.width) * int(self.cfg.height) * 4,
        )
        shared_size = self._shared_slot_bytes * self._shared_slot_count
        self._shared = shared_memory.SharedMemory(create=True, size=shared_size)
        self._shared_sequence = self._process_context.Value(
            ctypes.c_ulonglong,
            0,
            lock=False,
        )
        self._shared_frame_nbytes = self._process_context.Value(
            ctypes.c_ulonglong,
            0,
            lock=False,
        )
        self._shared_last_frame_time = self._process_context.Value(
            ctypes.c_double,
            0.0,
            lock=False,
        )
        self._shared_frame_shape = self._process_context.Array(
            ctypes.c_int,
            [0, 0, 0, 0],
            lock=False,
        )
        self._last_parent_sequence = 0
        self._capture_stop_event = self._process_context.Event()
        self._spawn_capture_process()

    def _spawn_capture_process(self) -> None:
        if self._closed or self._shared is None:
            return

        self._capture_process_started_at = time.monotonic()
        self._capture_stop_event = self._process_context.Event()
        process = self._process_context.Process(
            target=_camera_capture_process_main,
            args=(
                self.cfg,
                self._shared.name,
                self._shared_slot_bytes,
                self._shared_slot_count,
                self._shared_sequence,
                self._shared_frame_nbytes,
                self._shared_frame_shape,
                self._shared_last_frame_time,
                self._capture_stop_event,
            ),
            name="basler-capture-process",
            daemon=True,
        )
        process.start()
        self._capture_process = process
        print(
            f"[CAM-PROC] Capture-Prozess gestartet (PID={process.pid})",
            flush=True,
        )

    def _wait_for_first_process_frame(self) -> None:
        deadline = time.monotonic() + self.PROCESS_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._shared_sequence is not None and self._shared_sequence.value > 0:
                return
            process = self._capture_process
            if process is None or not process.is_alive():
                break
            time.sleep(0.05)

        self._release_capture_process()
        raise RuntimeError(
            "Basler-Capture-Prozess lieferte innerhalb des Startzeitlimits "
            "kein Bild"
        )

    def _terminate_capture_process(self, process) -> None:
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)

    def _restart_capture_process(self, reason: str) -> None:
        if self._closed or self._stop_requested:
            return

        with self._capture_process_lock:
            now = time.monotonic()
            if now - self._last_process_restart < 1.0:
                return
            self._last_process_restart = now
            print(f"[CAM-PROC] Capture-Prozess wird neu gestartet: {reason}", flush=True)

            process = self._capture_process
            if self._capture_stop_event is not None:
                self._capture_stop_event.set()
            self._terminate_capture_process(process)
            self._capture_process = None
            self._last_parent_sequence = int(self._shared_sequence.value)
            self._spawn_capture_process()

    def _read_from_capture_process(self):
        if self._closed or self._stop_requested:
            return False, None

        process = self._capture_process
        sequence_value = self._shared_sequence
        if process is None or sequence_value is None:
            return False, None

        sequence = int(sequence_value.value)
        if sequence <= self._last_parent_sequence:
            now = time.monotonic()
            last_frame_time = float(self._shared_last_frame_time.value)
            process_stalled = (
                now - self._capture_process_started_at
                >= self.PROCESS_STALL_TIMEOUT_SECONDS
                and (
                    last_frame_time <= 0.0
                    or now - last_frame_time >= self.PROCESS_STALL_TIMEOUT_SECONDS
                )
            )
            if not process.is_alive():
                self._restart_capture_process("Prozess wurde beendet")
            elif process_stalled:
                self._restart_capture_process("kein neues Bild innerhalb des Zeitlimits")
            return False, None

        for _ in range(3):
            sequence_before = int(sequence_value.value)
            if sequence_before <= self._last_parent_sequence:
                return False, None

            slot = sequence_before % self._shared_slot_count
            nbytes = int(self._shared_frame_nbytes.value)
            ndim = int(self._shared_frame_shape[0])
            shape = tuple(
                int(self._shared_frame_shape[index + 1])
                for index in range(min(ndim, 3))
            )
            if nbytes <= 0 or not shape:
                return False, None

            offset = slot * self._shared_slot_bytes
            raw = np.frombuffer(
                self._shared.buf,
                dtype=np.uint8,
                count=nbytes,
                offset=offset,
            ).copy()
            sequence_after = int(sequence_value.value)
            if sequence_before != sequence_after:
                continue

            try:
                image = raw.reshape(shape)
            except ValueError:
                self._log_grab_error(
                    f"Shared-Memory-Frame hat ungueltige Form: "
                    f"shape={shape} bytes={nbytes}"
                )
                return False, None

            self._last_parent_sequence = sequence_before
            frame = Frame()
            frame.frame = image
            frame.timestamp = float(self._shared_last_frame_time.value)
            return True, frame

        return False, None

    def _release_capture_process(self) -> bool:
        with self._release_lock:
            if self._closed and self._capture_process is None:
                complete = not self._shutdown_incomplete
                self._close_shared_memory()
                return complete
            self._stop_requested = True
            self._closed = True
            process = self._capture_process
            self._capture_process = None
            if self._capture_stop_event is not None:
                self._capture_stop_event.set()

        if process is not None:
            process.join(timeout=self.PROCESS_STOP_TIMEOUT_SECONDS)
            if process.is_alive():
                print(
                    "[CAM-PROC] Capture-Prozess reagiert nicht; "
                    "er wird beendet",
                    flush=True,
                )
                self._terminate_capture_process(process)

        self._close_shared_memory()
        self._shutdown_incomplete = False
        print("[CAM-PROC] Shared-Memory und Capture-Prozess freigegeben", flush=True)
        return True

    def _close_shared_memory(self) -> None:
        shared = self._shared
        self._shared = None
        if shared is None:
            return
        try:
            shared.close()
        except Exception:
            pass
        try:
            shared.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[CAM-PROC] Shared-Memory konnte nicht entfernt werden: {exc}")

    # ------------------------------------------------------------------
    # Device discovery and configuration
    # ------------------------------------------------------------------

    def request_capture_stop(self) -> None:
        """Tell the capture side to stop before the camera is released."""
        self._stop_requested = True
        if self._process_mode and self._capture_stop_event is not None:
            self._capture_stop_event.set()
            print("[CAM-PROC] Stop für Capture-Prozess angefordert", flush=True)

    def _select_device_info(self):

        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if not devices:
            raise RuntimeError(
                "Keine Basler-Kamera gefunden (pypylon EnumerateDevices=0). "
                "USB-Verbindung, pylon-Treiber und Zugriffsrechte pruefen."
            )

        wanted_serial = str(getattr(self.cfg, "serial_number", "") or "").strip()
        if not wanted_serial:
            return factory, devices[0]

        for device_info in devices:
            if self._device_serial(device_info) == wanted_serial:
                return factory, device_info

        available = ", ".join(self._device_serial(info) for info in devices)
        raise RuntimeError(
            f"Basler-Kamera mit Seriennummer {wanted_serial!r} nicht gefunden. "
            f"Verfuegbare Seriennummern: {available or 'unbekannt'}"
        )

    @staticmethod
    def _device_serial(device_info) -> str:
        try:
            return str(device_info.GetSerialNumber())
        except Exception:
            return ""

    @staticmethod
    def _device_label(device_info) -> str:
        values = []
        for method_name in ("GetModelName", "GetSerialNumber", "GetDeviceClass"):
            try:
                value = getattr(device_info, method_name)()
            except Exception:
                continue
            if value:
                values.append(str(value))
        return " / ".join(values) or "unbekanntes Basler-Geraet"

    def _open_camera(self):
        print("_open_kamera Funktion wird gestartet")
        #factory, device_info = self._select_device_info()
        #print(f"versuche Kamera {self._device_label(device_info)} zu öffnen")
        # camera = pylon.InstantCamera(factory.CreateDevice(device_info))

        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        print(" versuche Kamera zu öffnen")

        try:
            camera.Open()
            self._configure_camera(camera)
            camera.StartGrabbing(self._grab_strategy())
            self._settle_stream(camera)
            #print(f"[CAMERA] Basler geoeffnet: {self._device_label(device_info)}")
            self._log_configuration(camera)
            self._log_device_error_report(camera, "Kamera-Start")
            return camera
        except Exception as exc:
            self._log_grab_error(
                "Basler-Kamera konnte nicht geoeffnet/konfiguriert werden: "
                f"{type(exc).__name__}: {exc}"
            )
            self._log_device_error_report(camera, "Kamera-Startfehler")
            self._stop_and_close(camera)
            raise

    def _settle_stream(self, camera) -> None:
        """Discard transient USB results before handing the source to workers."""
        deadline = time.perf_counter() + self.STARTUP_SETTLE_SECONDS
        while time.perf_counter() < deadline:
            result = None
            try:
                result = camera.RetrieveResult(
                    self.RETRIEVE_TIMEOUT_MS,
                    pylon.TimeoutHandling_Return,
                )
            except Exception:
                # The regular grab loop owns error reporting and recovery.
                # Startup is deliberately best-effort; keep settling so a
                # transient USB result does not reach the worker.
                time.sleep(0.01)
            finally:
                if result is not None:
                    try:
                        result.Release()
                    except Exception:
                        pass

    def _configure_camera(self, camera) -> None:
        """Apply all settings that must be made while the camera is open."""

        # The camera is operated by the user-triggered grab loop.  Explicitly
        # disabling trigger mode avoids inheriting a stale camera state from
        # pylon Viewer or an earlier process.
        self._set_feature(camera, "TriggerSelector", "FrameStart", required=False)
        self._set_feature(camera, "TriggerMode", "Off", required=True)
        self._set_feature(camera, "AcquisitionMode", "Continuous", required=True)

        self._set_feature(camera, "ExposureAuto", "Off", required=True)
        self._set_feature(camera, "GainAuto", "Off", required=True)
        self._set_feature(
            camera,
            "PixelFormat",
            getattr(self.cfg, "pixel_format", "BayerRG8"),
            required=True,
        )
        self._set_feature(camera, "ExposureTime", float(self.cfg.exposure_time), required=True)
        self._set_feature(camera, "Gain", float(self.cfg.gain), required=True)

        # Set offsets to zero before changing the ROI.  This avoids an invalid
        # intermediate ROI when the old offset is too large for the new size.
        self._set_feature(camera, "OffsetX", 0, required=True)
        self._set_feature(camera, "OffsetY", 0, required=True)
        self._set_feature(camera, "Width", int(self.cfg.width), required=True)
        self._set_feature(camera, "Height", int(self.cfg.height), required=True)
        self._set_feature(camera, "OffsetX", int(self.cfg.offset_x), required=True)
        self._set_feature(camera, "OffsetY", int(self.cfg.offset_y), required=True)

        max_buffers = max(1, int(getattr(self.cfg, "max_num_buffer", 256)))
        try:
            camera.MaxNumBuffer = max_buffers
            print(f"pylon MaxNumBuffer = {max_buffers}")
        except Exception as exc:
            print(f"[CAMERA-WARN] MaxNumBuffer nicht setzbar: {exc}")

        requested_fps = float(getattr(self.cfg, "force_fps", 0) or 0)
        if requested_fps > 0:
            # This feature is not present on every camera/transport layer.  A
            # missing optional frame-rate control must not prevent grabbing.
            self._set_feature(camera, "AcquisitionFrameRateEnable", True, required=False)
            self._set_feature(camera, "AcquisitionFrameRate", requested_fps, required=False)

    @staticmethod
    def _set_feature(camera, name: str, value, *, required: bool) -> bool:
        try:
            node = getattr(camera, name)
            setter = getattr(node, "SetValue", None)
            if callable(setter):
                setter(value)
            else:
                node.Value = value
            return True
        except Exception as exc:
            message = f"{name}={value!r} konnte nicht gesetzt werden: {exc}"
            if required:
                raise RuntimeError(message) from exc
            print(f"[CAMERA-WARN] {message}")
            return False

    @staticmethod
    def _get_feature(camera, name: str):
        try:
            node = getattr(camera, name)
            getter = getattr(node, "GetValue", None)
            return getter() if callable(getter) else node.Value
        except Exception:
            return None

    def _log_configuration(self, camera) -> None:
        model = self._get_feature(camera, "DeviceModelName")
        if model:
            print(f"pylon Modell: {model}")
        for name in (
            "PixelFormat",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "ExposureAuto",
            "ExposureTime",
            "GainAuto",
            "Gain",
            "AcquisitionMode",
            "TriggerMode",
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate",
            "BslResultingAcquisitionFrameRate",
            "ResultingFrameRate",
        ):
            value = self._get_feature(camera, name)
            if value is not None:
                print(f"pylon {name} = {value}")

    def _grab_strategy(self):
        if getattr(self.cfg, "grab_strategy_latest_only", False):
            return pylon.GrabStrategy_LatestImageOnly
        return pylon.GrabStrategy_OneByOne

    # ------------------------------------------------------------------
    # Frame acquisition and recovery
    # ------------------------------------------------------------------

    def _grab_raw_once(self):
        with self._camera_lock:
            if self._closed or self._stop_requested or self.camera is None:
                return False, None

            camera = self.camera
            grab_result = None
            try:
                if not camera.IsGrabbing():
                    self._report_grab_error(
                        camera,
                        "Kamera grabbt nicht (IsGrabbing=False)",
                    )
                    return False, None

                grab_result = camera.RetrieveResult(
                    self.RETRIEVE_TIMEOUT_MS,
                    pylon.TimeoutHandling_Return,
                )
                if grab_result is None:
                    self._report_grab_error(camera, "RetrieveResult liefert None")
                    return False, None

                if not grab_result.IsValid():
                    self._report_grab_error(
                        camera,
                        "Grab-Ergebnis ungueltig: "
                        f"code={self._grab_error_code(grab_result)} "
                        f"text={self._grab_error_text(grab_result)}"
                    )
                    return False, None

                if not grab_result.GrabSucceeded():
                    self._report_grab_error(
                        camera,
                        "Grab fehlgeschlagen: "
                        f"code={self._grab_error_code(grab_result)} "
                        f"text={self._grab_error_text(grab_result)}"
                    )
                    return False, None

                image = grab_result.GetArray()
                if image is None:
                    self._report_grab_error(
                        camera,
                        "Grab-Ergebnis enthaelt kein Bild",
                    )
                    return False, None

                frame = Frame()
                # pypylon's array is backed by the grab result.  Copy before
                # Release(), otherwise a later grab can overwrite the image.
                frame.frame = image.copy()
                frame.timestamp = time.perf_counter()
                return True, frame
            except Exception as exc:
                self._report_grab_error(
                    camera,
                    f"Basler-Grab fehlgeschlagen: {type(exc).__name__}: {exc}",
                )
                return False, None
            finally:
                if grab_result is not None:
                    try:
                        grab_result.Release()
                    except Exception as exc:
                        self._report_grab_error(
                            camera,
                            "Grab-Ergebnis konnte nicht freigegeben werden: "
                            f"{type(exc).__name__}: {exc}",
                        )

    @staticmethod
    def _grab_error_code(grab_result) -> str:
        try:
            return str(grab_result.GetErrorCode())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _grab_error_text(grab_result) -> str:
        try:
            return str(grab_result.GetErrorDescription())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _format_device_error_code(value) -> str:
        """Format a Basler BslErrorReportValue for a useful diagnostic log."""
        if isinstance(value, int):
            return f"{value} (0x{value:08X})"
        return str(value)

    def _log_device_error_report(self, camera, context: str) -> None:
        """Read the camera's optional Basler device-error report.

        Basler exposes the report through ``BslErrorPresent``,
        ``BslErrorReportValue`` and ``BslErrorReportNext``. These nodes are
        not available on every camera model, so the diagnostic is best effort
        and must never turn a normal grab error into a second application
        error.
        """
        if camera is None:
            return

        try:
            error_present = getattr(camera, "BslErrorPresent", None)
            report_value = getattr(camera, "BslErrorReportValue", None)
            report_next = getattr(camera, "BslErrorReportNext", None)
            if error_present is None or report_value is None:
                return

            is_readable = getattr(error_present, "IsReadable", None)
            if callable(is_readable) and not is_readable():
                return
            if not bool(error_present.Value):
                return

            self._log_grab_error(f"{context}: BslErrorPresent=True")
            for _ in range(self.DEVICE_ERROR_REPORT_LIMIT):
                value = report_value.Value
                self._log_grab_error(
                    f"{context}: BslErrorReportValue="
                    f"{self._format_device_error_code(value)}"
                )

                if report_next is None:
                    break
                is_executable = getattr(report_next, "IsExecutable", None)
                if callable(is_executable) and not is_executable():
                    break
                report_next.Execute()
        except Exception as exc:
            self._log_grab_error(
                f"{context}: Basler-Fehlerbericht konnte nicht gelesen werden: "
                f"{type(exc).__name__}: {exc}"
            )

    def _report_grab_error(self, camera, message: str) -> None:
        """Log the grab error and the camera-side diagnostic, if available."""
        self._log_grab_error(message)
        self._log_device_error_report(camera, "Basler-Geraet")

    def _log_grab_error(self, message: str) -> None:
        now = time.perf_counter()
        if message != self._last_grab_error or now - self._last_grab_error_time >= 1.0:
            print(f"[CAM-ERR] {message}")
            self._last_grab_error = message
            self._last_grab_error_time = now

    def read(self):
        if self._process_mode:
            ok, frame = self._read_from_capture_process()
            now = time.perf_counter()
            if ok and frame is not None:
                self._read_count += 1
            else:
                self._read_failures += 1

            if now - self._read_log_time >= 1.0:
                elapsed = now - self._read_log_time
                print(
                    f"[CAM-RAW-PARENT] geliefert={self._read_count} "
                    f"fps={self._read_count / elapsed:.1f} "
                    f"fehlgeschlagen={self._read_failures}",
                    flush=True,
                )
                self._read_count = 0
                self._read_failures = 0
                self._read_log_time = now
            return ok, frame

        ok, frame = self._grab_raw_once()
        now = time.perf_counter()

        if ok and frame is not None:
            self._read_count += 1
            self._consecutive_grab_failures = 0
            self._first_grab_failure_time = 0.0
        else:
            self._read_failures += 1
            self._consecutive_grab_failures += 1
            if self._consecutive_grab_failures == 1:
                self._first_grab_failure_time = now
            failure_duration = now - self._first_grab_failure_time
            if (
                self._consecutive_grab_failures >= self.RECOVERY_FAILURE_THRESHOLD
                and failure_duration >= self.RECOVERY_MIN_FAILURE_DURATION_SECONDS
            ):
                self._recover_grab_stream()

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

        return ok, frame

    def _recover_grab_stream(self) -> None:
        with self._camera_lock:
            if self._closed:
                return

            now = time.perf_counter()
            if now - self._last_recovery_time < self.RECOVERY_INTERVAL_SECONDS:
                return
            self._last_recovery_time = now

            print("[CAM-RECOVER] Kamera-Grab-Stream wird neu aufgebaut")
            old_camera = self.camera
            self.camera = None

        # StopGrabbing() can block in the USB transport layer.  Recovery is
        # called from the camera worker, so doing it inline would strand that
        # worker and make a later shutdown impossible.  Do not reconnect
        # while cleanup is still blocked: a second InstantCamera would only
        # compete for the same USB handle.
        if not self._cleanup_camera_bounded(old_camera):
            print(
                "[CAM-RECOVER] Altes Kamera-Handle blockiert; "
                "Reconnect wird uebersprungen"
            )
            return

        # Close() stops acquisition, while DestroyDevice() detaches the
        # transport-layer device.  The explicit detach is important on USB:
        # otherwise the old InstantCamera can keep the device node busy even
        # though pylon reports it as closed.
        del old_camera
        gc.collect()

        if self._closed:
            return

        try:
            new_camera = self._open_camera()
            with self._camera_lock:
                if self._closed:
                    discard_camera = True
                else:
                    self.camera = new_camera
                    discard_camera = False
            if discard_camera:
                self._cleanup_camera_bounded(new_camera)
                return
            self._consecutive_grab_failures = 0
            print("[CAM-RECOVER] Kamera neu verbunden")
        except Exception as exc:
            print(f"[CAM-RECOVER] Kamera-Reconnect fehlgeschlagen: {exc}")

    def _cleanup_camera_bounded(self, camera, timeout: float = 1.0) -> bool:
        """Run pypylon cleanup without blocking an application worker."""
        with self._cleanup_lock:
            if self._pending_cleanup is not None:
                if self._pending_cleanup.is_alive():
                    return False
                self._pending_cleanup = None

            if camera is None:
                return True

            cleanup = threading.Thread(
                target=self._stop_and_close,
                args=(camera,),
                name="basler-cleanup",
                daemon=True,
            )
            self._pending_cleanup = cleanup
            cleanup.start()
        cleanup.join(timeout=max(0.0, float(timeout)))
        finished = not cleanup.is_alive()
        if finished:
            with self._cleanup_lock:
                if self._pending_cleanup is cleanup:
                    self._pending_cleanup = None
        return finished

    @staticmethod
    def _stop_and_close(camera) -> None:
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                print("[CAMERA] StopGrabbing wird aufgerufen", flush=True)
                camera.StopGrabbing()
                print("[CAMERA] StopGrabbing abgeschlossen", flush=True)
        except Exception as exc:
            print(
                "[CAMERA] StopGrabbing fehlgeschlagen: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        try:
            if camera.IsOpen():
                camera.Close()
                print("[CAMERA] Close abgeschlossen", flush=True)
        except Exception as exc:
            print(
                "[CAMERA] Close fehlgeschlagen: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        try:
            attached = getattr(camera, "IsPylonDeviceAttached", None)
            if not callable(attached) or attached():
                destroy = getattr(camera, "DestroyDevice", None)
                if callable(destroy):
                    destroy()
                    print("[CAMERA] DestroyDevice abgeschlossen", flush=True)
        except Exception as exc:
            print(
                "[CAMERA] DestroyDevice fehlgeschlagen: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Source compatibility and shutdown
    # ------------------------------------------------------------------

    def showFPS(self) -> str:
        for name in ("BslResultingAcquisitionFrameRate", "ResultingFrameRate"):
            value = self._get_feature(self.camera, name) if self.camera is not None else None
            try:
                if value is not None and float(value) > 0:
                    self.fps = float(value)
                    break
            except (TypeError, ValueError):
                continue
        return f"{self.fps:.2f}"

    def reset(self) -> None:
        """Keep the source interface; a live camera has no replay position."""

    def get_temperature(self) -> Optional[str]:
        if self.camera is None:
            return None
        for name in ("DeviceTemperature", "BslTemperature"):
            value = self._get_feature(self.camera, name)
            if value is not None:
                return str(value)
        return None

    # Preserve the old method spelling for callers outside the application.
    def getTemperatur(self, camera=None) -> Optional[str]:
        return self.get_temperature()

    def release(self) -> bool:
        """Release the camera without allowing a stuck USB cleanup to hang app exit.

        On this USB camera/transport combination, ``StopGrabbing`` has been
        observed to block in the kernel after the first run.  Waiting for it
        in the application's main thread leaves the process alive and keeps
        the camera busy for the next start.  Marking the source closed first
        prevents new grabs; cleanup then runs in a daemon thread and is given
        only a short grace period.  If pylon remains stuck, process exit
        releases the USB file descriptor.
        """
        if self._process_mode:
            return self._release_capture_process()

        with self._release_lock:
            if self._closed:
                return not self._shutdown_incomplete
            self._stop_requested = True
            self._closed = True
            # The worker threads are stopped by KickerApp before release().
            # Do not wait on _camera_lock here: an in-flight RetrieveResult()
            # may be the operation that is holding it.
            camera = self.camera
            self.camera = None

        # A recovery cleanup may still own an old pylon camera.  Never report
        # a clean shutdown, and never start a second pylon cleanup, while that
        # native call is still blocked.
        with self._cleanup_lock:
            pending_cleanup = self._pending_cleanup
            if pending_cleanup is not None and not pending_cleanup.is_alive():
                self._pending_cleanup = None
                pending_cleanup = None

        if pending_cleanup is not None:
            pending_cleanup.join(timeout=1.0)
            if pending_cleanup.is_alive():
                self._shutdown_incomplete = True
                print(
                    "[SHUTDOWN] Vorheriger Basler-Cleanup blockiert; "
                    "Kamera kann in diesem Prozess nicht sicher weiter "
                    "verwendet werden",
                    flush=True,
                )
                return False
            with self._cleanup_lock:
                if self._pending_cleanup is pending_cleanup:
                    self._pending_cleanup = None

        cleanup = threading.Thread(
            target=self._release_camera_async,
            args=(camera,),
            name="basler-cleanup",
            daemon=True,
        )
        cleanup.start()
        cleanup.join(timeout=1.0)
        cleanup_completed = not cleanup.is_alive()
        if not cleanup_completed:
            self._shutdown_incomplete = True
            print(
                "[SHUTDOWN] Basler-Cleanup blockiert; "
                "Prozess wird kontrolliert beendet",
                flush=True,
            )
            return False

        self._shutdown_incomplete = False
        print("[SHUTDOWN] Basler sauber gestoppt und geschlossen", flush=True)
        return True

    def _release_camera_async(self, camera) -> bool:
        """Serialize cleanup with a possible in-flight grab/recovery."""
        if camera is None:
            return True

        # Avoid running StopGrabbing concurrently with RetrieveResult or the
        # recovery path.  The timeout is intentionally bounded as this is a
        # daemon cleanup thread.
        if not self._camera_lock.acquire(timeout=1.0):
            print(
                "[SHUTDOWN] Kamera-Lock nicht freigegeben; "
                "Cleanup abgebrochen",
                flush=True,
            )
            return False
        try:
            self._stop_and_close(camera)
            return True
        finally:
            self._camera_lock.release()

    close = release
