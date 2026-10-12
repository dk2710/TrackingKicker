"""The pylon owner for the TrackingKicker camera pipeline.

This process deliberately does only transport work: it retrieves valid raw
frames and publishes them to the shared-memory ring. JPEG encoding, ball
detection and all web UI work belong to the BallTuner process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pypylon import pylon

from ..tuner.settings import (
    BallDetectionSettings,
    CameraSettings,
    load_settings,
    save_settings,
)
from ..camera_frame_ipc import DEFAULT_NAME, DEFAULT_SLOTS, SharedFrameRing
from ..camera.pixel_format import resolve_pixel_format


class CameraProcess:
    RETRIEVE_TIMEOUT_MS = 1000
    PREVIEW_FPS_LIMIT = 30.0

    def __init__(
        self,
        settings: CameraSettings,
        *,
        settings_path: Path,
        frame_ipc_name: str = DEFAULT_NAME,
        frame_ipc_slots: int = DEFAULT_SLOTS,
        ball_settings: Optional[BallDetectionSettings] = None,
        publish_frames: bool = True,
    ) -> None:
        self.settings = settings
        self.settings_path = Path(settings_path)
        self.ball_settings = ball_settings or BallDetectionSettings()
        self.frame_ipc_name = str(frame_ipc_name)
        # The mailbox has three fixed slots: one can be written while the
        # consumer copies another, with one additional slot for rollover.
        self.frame_ipc_slots = DEFAULT_SLOTS
        self.publish_frames = bool(publish_frames)
        self.frame_ring = SharedFrameRing.create_or_replace(
            self.frame_ipc_name,
            width=int(settings.width),
            height=int(settings.height),
            slots=self.frame_ipc_slots,
        )
        self.camera = None
        self._camera_lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._publisher_thread: Optional[threading.Thread] = None
        # The capture thread owns the pylon result. It copies each valid
        # frame into a fresh numpy array before releasing that result. The
        # publisher only takes the newest pending array, so it never makes
        # the camera wait for shared-memory I/O.
        self._handoff_lock = threading.Lock()
        self._handoff_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_frame_timestamp = 0.0
        self._pending_frame_created_at = 0.0
        self._capture_handoff_dropped = 0
        self._frame_ring_lock = threading.RLock()
        self._capture_diag_lock = threading.Lock()
        self._capture_diag_started = time.monotonic()
        self._capture_diag_attempts = 0
        self._capture_diag_successes = 0
        self._capture_diag_failures = 0
        self._capture_diag_retrieve_ms = 0.0
        self._capture_diag_retrieve_max_ms = 0.0
        self._capture_diag_validate_ms = 0.0
        self._capture_diag_validate_max_ms = 0.0
        self._capture_diag_copy_ms = 0.0
        self._capture_diag_copy_max_ms = 0.0
        self._capture_diag_handoff_ms = 0.0
        self._capture_diag_handoff_max_ms = 0.0
        self._capture_diag_release_ms = 0.0
        self._capture_diag_release_max_ms = 0.0
        self._publisher_diag_lock = threading.Lock()
        self._publisher_diag_started = time.monotonic()
        self._publisher_diag_frames = 0
        self._publisher_diag_errors = 0
        self._publisher_diag_publish_ms = 0.0
        self._publisher_diag_publish_max_ms = 0.0
        self._publisher_diag_handoff_age_ms = 0.0
        self._publisher_diag_handoff_age_max_ms = 0.0
        self._state_lock = threading.Lock()
        self._camera_open = False
        self._camera_grabbing = False
        self._model = ""
        self._available_pixel_formats: tuple[str, ...] = ()
        self._resulting_fps: Optional[float] = None
        self._camera_fps = 0.0
        self._camera_frames_total = 0
        self._failed_grabs = 0
        self._payload_errors = 0
        self._dropped_frames = 0
        self._frame_ipc_frames = 0
        self._frame_ipc_dropped = 0
        self._frame_ipc_errors = 0
        self._reconnects = 0
        self._last_block_id: Optional[int] = None
        self._last_success_block_id: Optional[int] = None
        self._last_frame_time = 0.0
        self._last_error = ""
        self._last_error_log_time = 0.0
        self._last_failure_phase = ""
        self._last_result_diagnostic = ""
        self._last_failure_time = 0.0
        self._capture_aborted = False

        print(
            f"[CAMERA-PROCESS] Frame-IPC aktiv: {self.frame_ipc_name} "
            f"({settings.width}x{settings.height}, {self.frame_ipc_slots} Slots), "
            f"Publishing={'EIN' if self.publish_frames else 'AUS'}"
        )

    def start(self) -> None:
        self._stop_event.clear()
        self._handoff_event.clear()
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop,
            name="camera-process-publisher",
            daemon=True,
        )
        self._publisher_thread.start()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-process-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._handoff_event.set()
        self._close_camera()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=2.0)
        with self._frame_ring_lock:
            self.frame_ring.close(unlink=True)

    def reconnect(self) -> bool:
        with self._camera_lock:
            if self._stop_event.is_set():
                return False
            opened = self._open_camera_locked()
        if not opened:
            self._abort_capture(self._last_error or "Kamera konnte nicht geoeffnet werden")
            return False
        with self._state_lock:
            self._capture_aborted = False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="camera-process-capture",
                daemon=True,
            )
            self._thread.start()
        if self._publisher_thread is None or not self._publisher_thread.is_alive():
            self._publisher_thread = threading.Thread(
                target=self._publisher_loop,
                name="camera-process-publisher",
                daemon=True,
            )
            self._publisher_thread.start()
        return True

    def update_settings(self, values: Dict[str, Any]) -> bool:
        valid_names = {item.name for item in fields(CameraSettings)}
        candidate = CameraSettings(**vars(self.settings))
        for name, value in values.items():
            if name not in valid_names:
                continue
            current = getattr(candidate, name)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Ungueltiger Wert fuer {name}: {value!r}") from exc
            if name in {"width", "height", "max_num_buffer"} and value <= 0:
                raise ValueError(f"{name} muss groesser als 0 sein")
            if name == "force_fps" and value < 30:
                raise ValueError("force_fps muss mindestens 30 sein")
            if name in {
                "offset_x", "offset_y", "packet_size", "inter_packet_delay",
                "max_transfer_size", "num_max_queued_urbs",
                "device_link_throughput_limit",
            } and value < 0:
                raise ValueError(f"{name} darf nicht negativ sein")
            if name == "grab_strategy" and value != "latest_image_only":
                raise ValueError("Die Tracking-Konfiguration verwendet LatestImageOnly")
            setattr(candidate, name, value)

        shape_changed = (candidate.width, candidate.height) != (
            self.settings.width,
            self.settings.height,
        )
        if shape_changed:
            replacement = SharedFrameRing.create_or_replace(
                self.frame_ipc_name,
                width=int(candidate.width),
                height=int(candidate.height),
                slots=self.frame_ipc_slots,
            )
            with self._frame_ring_lock:
                old_ring = self.frame_ring
                self.frame_ring = replacement
                old_ring.close(unlink=False)
        self.settings = candidate
        save_settings(
            self.settings_path,
            self.settings,
            self.ball_settings,
            {
                "host": "127.0.0.1",
                "port": 5011,
                "ball_detection_enabled": False,
                "frame_ipc_name": self.frame_ipc_name,
                "frame_ipc_slots": self.frame_ipc_slots,
                "camera_control_port": 5011,
            },
        )
        return self.reconnect()

    # ------------------------------------------------------------------
    # pylon configuration and lifecycle
    # ------------------------------------------------------------------
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
    def _set_stream_parameter(camera, name: str, value: int) -> bool:
        if int(value or 0) <= 0:
            return False
        try:
            node = camera.GetStreamGrabberNodeMap().GetNode(name)
            if node is None:
                raise RuntimeError("Knoten nicht vorhanden")
            node.SetValue(int(value))
            print(f"pylon {name} = {int(value)}")
            return True
        except Exception as exc:
            print(f"[CAMERA-WARN] {name} nicht setzbar: {exc}")
            return False

    def _configure_camera(self, camera) -> None:
        settings = self.settings
        self._set_feature(camera, "ExposureAuto", settings.exposure_auto, required=True)
        self._set_feature(camera, "GainAuto", settings.gain_auto, required=True)
        pixel_format, available_formats = resolve_pixel_format(
            camera, settings.pixel_format
        )
        self._available_pixel_formats = available_formats
        if available_formats:
            print(
                "[CAMERA-PROCESS] Verfuegbare PixelFormate: "
                + ", ".join(available_formats)
            )
        self._set_feature(camera, "PixelFormat", pixel_format, required=True)
        settings.pixel_format = pixel_format
        self._set_feature(camera, "ExposureTime", float(settings.exposure_time), required=True)
        self._set_feature(camera, "Gain", float(settings.gain), required=True)
        self._set_feature(camera, "OffsetX", 0, required=True)
        self._set_feature(camera, "OffsetY", 0, required=True)
        self._set_feature(camera, "Width", int(settings.width), required=True)
        self._set_feature(camera, "Height", int(settings.height), required=True)
        self._set_feature(camera, "OffsetX", int(settings.offset_x), required=True)
        self._set_feature(camera, "OffsetY", int(settings.offset_y), required=True)
        if int(settings.max_num_buffer or 0) > 0:
            try:
                camera.MaxNumBuffer = int(settings.max_num_buffer)
            except Exception as exc:
                print(f"[CAMERA-WARN] MaxNumBuffer nicht setzbar: {exc}")
        self._set_stream_parameter(camera, "MaxTransferSize", settings.max_transfer_size)
        self._set_stream_parameter(camera, "NumMaxQueuedUrbs", settings.num_max_queued_urbs)
        if int(settings.device_link_throughput_limit or 0) > 0:
            self._set_feature(
                camera,
                "DeviceLinkThroughputLimit",
                int(settings.device_link_throughput_limit),
                required=False,
            )
        self._set_feature(
            camera,
            "AcquisitionFrameRateEnable",
            bool(settings.acquisition_frame_rate_enable),
            required=False,
        )
        if settings.acquisition_frame_rate_enable and settings.force_fps > 0:
            self._set_feature(camera, "AcquisitionFrameRate", settings.force_fps, required=False)

    def _open_camera_locked(self) -> bool:
        self._close_camera_locked()
        camera = None
        try:
            device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            camera = pylon.InstantCamera(device)
            # This USB camera can expose AcquisitionMode as read-only during
            # pylon's automatic acquire configuration needs this workaround
            # during reconnects as well as during the initial camera setup.
            camera.RegisterConfiguration(
                None,
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_None,
            )
            camera.Open()
            self._configure_camera(camera)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            with self._state_lock:
                self._camera_open = True
                self._camera_grabbing = True
                self._last_block_id = None
            self.camera = camera
            self._model = str(camera.GetDeviceInfo().GetModelName())
            self._resulting_fps = self._read_resulting_fps(camera)
            self._set_error("")
            print(f"[CAMERA-PROCESS] Basler geoeffnet: {self._model}")
            print(
                "[CAMERA-PROCESS] Transport: "
                f"LatestImageOnly, MaxTransferSize={self.settings.max_transfer_size}, "
                f"DeviceLinkThroughputLimit={self.settings.device_link_throughput_limit}, "
                f"usbfs_memory_mb wird vom Host vorgegeben"
            )
            if self._resulting_fps is not None:
                print(f"[CAMERA-PROCESS] Resultierende Kamerarate: {self._resulting_fps:.2f} FPS")
            pixel_color_filter = self._get_feature(camera, "PixelColorFilter")
            if pixel_color_filter is not None:
                print(f"[CAMERA-PROCESS] PixelColorFilter: {pixel_color_filter}")
            return True
        except Exception as exc:
            self._set_error(f"Kamera konnte nicht geoeffnet werden: {exc}")
            self._stop_and_close(camera)
            self.camera = None
            with self._state_lock:
                self._camera_open = False
                self._camera_grabbing = False
            return False

    def _close_camera(self) -> None:
        with self._camera_lock:
            self._close_camera_locked()

    def _close_camera_locked(self) -> None:
        camera = self.camera
        self.camera = None
        with self._state_lock:
            self._camera_open = False
            self._camera_grabbing = False
        self._stop_and_close(camera)

    def _abort_capture(self, reason: str) -> None:
        """Stop capture immediately and keep the API alive for diagnostics."""
        self._close_camera()
        with self._state_lock:
            self._capture_aborted = True
        self._set_error(f"CAMERA-PROCESS ABGEBROCHEN: {reason}")

    @staticmethod
    def _stop_and_close(camera) -> None:
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception as exc:
            print(f"[CAMERA-WARN] StopGrabbing fehlgeschlagen: {exc}")
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception as exc:
            print(f"[CAMERA-WARN] Close fehlgeschlagen: {exc}")

    # ------------------------------------------------------------------
    # Transport loop and status
    # ------------------------------------------------------------------
    def _handoff_frame(self, image: np.ndarray, timestamp: float) -> None:
        """Replace the pending frame without waiting for the publisher."""

        with self._handoff_lock:
            if self._pending_frame is not None:
                with self._state_lock:
                    self._capture_handoff_dropped += 1
            self._pending_frame = image
            self._pending_frame_timestamp = float(timestamp)
            self._pending_frame_created_at = time.monotonic()
            self._handoff_event.set()

    def _record_capture_diagnostic(
        self,
        *,
        success: bool,
        retrieve_ms: float,
        validate_ms: float,
        copy_ms: float,
        handoff_ms: float,
        release_ms: float,
    ) -> Optional[str]:
        now = time.monotonic()
        with self._capture_diag_lock:
            self._capture_diag_attempts += 1
            if success:
                self._capture_diag_successes += 1
            else:
                self._capture_diag_failures += 1
            self._capture_diag_retrieve_ms += retrieve_ms
            self._capture_diag_retrieve_max_ms = max(
                self._capture_diag_retrieve_max_ms, retrieve_ms
            )
            self._capture_diag_validate_ms += validate_ms
            self._capture_diag_validate_max_ms = max(
                self._capture_diag_validate_max_ms, validate_ms
            )
            self._capture_diag_copy_ms += copy_ms
            self._capture_diag_copy_max_ms = max(
                self._capture_diag_copy_max_ms, copy_ms
            )
            self._capture_diag_handoff_ms += handoff_ms
            self._capture_diag_handoff_max_ms = max(
                self._capture_diag_handoff_max_ms, handoff_ms
            )
            self._capture_diag_release_ms += release_ms
            self._capture_diag_release_max_ms = max(
                self._capture_diag_release_max_ms, release_ms
            )
            elapsed = now - self._capture_diag_started
            if elapsed < 1.0:
                return None
            attempts = self._capture_diag_attempts
            successes = self._capture_diag_successes
            failures = self._capture_diag_failures
            retrieve_avg = self._capture_diag_retrieve_ms / max(1, attempts)
            validate_avg = self._capture_diag_validate_ms / max(1, attempts)
            copy_avg = self._capture_diag_copy_ms / max(1, successes)
            handoff_avg = self._capture_diag_handoff_ms / max(1, successes)
            release_avg = self._capture_diag_release_ms / max(1, attempts)
            message = (
                "[CAMERA-DIAG] capture 1s: "
                f"attempts={attempts}, success={successes}, failures={failures}, "
                f"fps={successes / elapsed:.2f}, "
                f"retrieve_ms(avg/max)={retrieve_avg:.3f}/{self._capture_diag_retrieve_max_ms:.3f}, "
                f"validate_ms(avg/max)={validate_avg:.3f}/{self._capture_diag_validate_max_ms:.3f}, "
                f"copy_ms(avg/max)={copy_avg:.3f}/{self._capture_diag_copy_max_ms:.3f}, "
                f"handoff_ms(avg/max)={handoff_avg:.3f}/{self._capture_diag_handoff_max_ms:.3f}, "
                f"release_ms(avg/max)={release_avg:.3f}/{self._capture_diag_release_max_ms:.3f}, "
                f"handoff_dropped={self._capture_handoff_dropped}"
            )
            self._capture_diag_started = now
            self._capture_diag_attempts = 0
            self._capture_diag_successes = 0
            self._capture_diag_failures = 0
            self._capture_diag_retrieve_ms = 0.0
            self._capture_diag_retrieve_max_ms = 0.0
            self._capture_diag_validate_ms = 0.0
            self._capture_diag_validate_max_ms = 0.0
            self._capture_diag_copy_ms = 0.0
            self._capture_diag_copy_max_ms = 0.0
            self._capture_diag_handoff_ms = 0.0
            self._capture_diag_handoff_max_ms = 0.0
            self._capture_diag_release_ms = 0.0
            self._capture_diag_release_max_ms = 0.0
            return message

    def _record_publisher_diagnostic(
        self, *, publish_ms: float, handoff_age_ms: float, error: bool = False
    ) -> Optional[str]:
        now = time.monotonic()
        with self._publisher_diag_lock:
            if error:
                self._publisher_diag_errors += 1
            else:
                self._publisher_diag_frames += 1
            self._publisher_diag_publish_ms += publish_ms
            self._publisher_diag_publish_max_ms = max(
                self._publisher_diag_publish_max_ms, publish_ms
            )
            self._publisher_diag_handoff_age_ms += handoff_age_ms
            self._publisher_diag_handoff_age_max_ms = max(
                self._publisher_diag_handoff_age_max_ms, handoff_age_ms
            )
            elapsed = now - self._publisher_diag_started
            if elapsed < 1.0:
                return None
            frames = self._publisher_diag_frames
            errors = self._publisher_diag_errors
            publish_avg = self._publisher_diag_publish_ms / max(1, frames + errors)
            age_avg = self._publisher_diag_handoff_age_ms / max(1, frames + errors)
            message = (
                "[CAMERA-DIAG] publisher 1s: "
                f"frames={frames}, errors={errors}, fps={frames / elapsed:.2f}, "
                f"publish_ms(avg/max)={publish_avg:.3f}/{self._publisher_diag_publish_max_ms:.3f}, "
                f"handoff_age_ms(avg/max)={age_avg:.3f}/{self._publisher_diag_handoff_age_max_ms:.3f}"
            )
            self._publisher_diag_started = now
            self._publisher_diag_frames = 0
            self._publisher_diag_errors = 0
            self._publisher_diag_publish_ms = 0.0
            self._publisher_diag_publish_max_ms = 0.0
            self._publisher_diag_handoff_age_ms = 0.0
            self._publisher_diag_handoff_age_max_ms = 0.0
            return message

    def _publisher_loop(self) -> None:
        """Publish only the newest frame handed off by the capture thread."""

        while not self._stop_event.is_set():
            if not self._handoff_event.wait(0.1):
                continue
            with self._handoff_lock:
                image = self._pending_frame
                timestamp = self._pending_frame_timestamp
                created_at = self._pending_frame_created_at
                self._pending_frame = None
                self._pending_frame_timestamp = 0.0
                self._pending_frame_created_at = 0.0
                self._handoff_event.clear()
            if image is None or not self.publish_frames:
                continue
            handoff_age_ms = max(0.0, (time.monotonic() - created_at) * 1000.0)
            publish_started = time.perf_counter()
            try:
                with self._frame_ring_lock:
                    # Timestamp the frame immediately after the owned
                    # GetArray().copy(). Preserve that acquisition timestamp
                    # through the asynchronous publisher instead of stamping
                    # it after the handoff/publisher delay.
                    sequence = self.frame_ring.publish(image, timestamp)
                publish_ms = (time.perf_counter() - publish_started) * 1000.0
                with self._state_lock:
                    if sequence is None:
                        self._frame_ipc_dropped += 1
                    else:
                        self._frame_ipc_frames += 1
                diagnostic = self._record_publisher_diagnostic(
                    publish_ms=publish_ms,
                    handoff_age_ms=handoff_age_ms,
                )
                if diagnostic:
                    print(diagnostic)
            except Exception as exc:
                publish_ms = (time.perf_counter() - publish_started) * 1000.0
                with self._state_lock:
                    self._frame_ipc_errors += 1
                self._set_error(f"Frame-IPC fehlgeschlagen: {exc}")
                diagnostic = self._record_publisher_diagnostic(
                    publish_ms=publish_ms,
                    handoff_age_ms=handoff_age_ms,
                    error=True,
                )
                if diagnostic:
                    print(diagnostic)

    def _capture_loop(self) -> None:
        report_time = time.monotonic()
        report_frames = 0
        while not self._stop_event.is_set():
            image = None
            grab_failed = False
            capture_success = False
            retrieve_ms = 0.0
            validate_ms = 0.0
            copy_ms = 0.0
            handoff_ms = 0.0
            release_ms = 0.0
            retrieve_started = time.perf_counter()
            with self._camera_lock:
                cleanup_running = (
                    self._cleanup_thread is not None
                    and self._cleanup_thread.is_alive()
                )
                if self.camera is None and not cleanup_running:
                    opened = self._open_camera_locked()
                    if not opened:
                        self._abort_capture(
                            self._last_error or "Kamera konnte nicht geoeffnet werden"
                        )
                        break
                camera = self.camera
                if camera is not None:
                    result = None
                    try:
                        result = camera.RetrieveResult(
                            self.RETRIEVE_TIMEOUT_MS,
                            pylon.TimeoutHandling_Return,
                        )
                        retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0
                        if result is None:
                            raise RuntimeError(
                                "RetrieveResult liefert None "
                                "(kein Grab-Ergebnis für diesen Abruf)"
                            )
                        validate_started = time.perf_counter()
                        valid = self._safe_result_value(result, "IsValid")
                        succeeded = self._safe_result_value(result, "GrabSucceeded")
                        validate_ms = (time.perf_counter() - validate_started) * 1000.0
                        if valid is False:
                            raise RuntimeError(
                                "Payload data has been discarded; "
                                f"Grab-Ergebnis ungueltig; {self._result_diagnostic(result, valid, succeeded)}"
                            )
                        if succeeded is False:
                            raise RuntimeError(
                                "Grab fehlgeschlagen; "
                                f"{self._result_diagnostic(result, valid, succeeded)}"
                            )
                        if valid is None or succeeded is None:
                            raise RuntimeError(
                                "Grab-Status konnte nicht gelesen werden; "
                                f"{self._result_diagnostic(result, valid, succeeded)}"
                            )
                        # GetArray() is only valid while this pylon result is
                        # alive. Copy it before Release(); the publisher
                        # thread receives an independent latest-frame array.
                        copy_started = time.perf_counter()
                        image = result.GetArray().copy()
                        # The time belongs to the copied camera frame, not to
                        # a later shared-memory publication.
                        frame_timestamp = time.perf_counter()
                        copy_ms = (time.perf_counter() - copy_started) * 1000.0
                        self._record_block_id(result)
                        if self.publish_frames:
                            handoff_started = time.perf_counter()
                            self._handoff_frame(image, frame_timestamp)
                            handoff_ms = (time.perf_counter() - handoff_started) * 1000.0
                        with self._state_lock:
                            self._camera_frames_total += 1
                            self._last_frame_time = time.monotonic()
                            self._last_failure_phase = ""
                            self._last_result_diagnostic = ""
                        report_frames += 1
                        capture_success = True
                    except Exception as exc:
                        grab_failed = True
                        if retrieve_ms <= 0.0:
                            retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0
                        diagnostic = self._result_diagnostic(result) if result is not None else "kein Grab-Ergebnis"
                        timing = (
                            f"timing_ms(retrieve={retrieve_ms:.3f}, "
                            f"validate={validate_ms:.3f}, copy={copy_ms:.3f}, "
                            f"handoff={handoff_ms:.3f}, release={release_ms:.3f})"
                        )
                        with self._state_lock:
                            self._failed_grabs += 1
                            if self._looks_like_payload_error(str(exc)):
                                self._payload_errors += 1
                            self._last_failure_phase = "RetrieveResult"
                            self._last_result_diagnostic = diagnostic
                            self._last_failure_time = time.monotonic()
                        self._set_error(
                            f"RetrieveResult/Read operation failed: {exc}; {diagnostic}; {timing}"
                        )
                    finally:
                        if result is not None:
                            release_started = time.perf_counter()
                            try:
                                result.Release()
                                release_ms = (time.perf_counter() - release_started) * 1000.0
                            except Exception as exc:
                                release_ms = (time.perf_counter() - release_started) * 1000.0
                                with self._state_lock:
                                    self._last_failure_phase = "Grab.Release"
                                    self._last_result_diagnostic = str(exc)
                                    self._last_failure_time = time.monotonic()
                                self._set_error(f"Grab.Release fehlgeschlagen: {exc}")
                                grab_failed = True
                                capture_success = False
            diagnostic = self._record_capture_diagnostic(
                success=capture_success,
                retrieve_ms=retrieve_ms,
                validate_ms=validate_ms,
                copy_ms=copy_ms,
                handoff_ms=handoff_ms,
                release_ms=release_ms,
            )
            if diagnostic:
                print(diagnostic)
            if camera is None:
                self._abort_capture("Kamera ist nicht erreichbar")
                break
            if grab_failed:
                self._abort_capture(
                    self._last_error or "RetrieveResult/Read operation failed"
                )
                break
            if image is None:
                # A failed/timeout grab should not spin at full speed. A
                # successful grab, including one published to shared memory,
                # must immediately request the next frame.
                time.sleep(0.01)
            now = time.monotonic()
            if now - report_time >= 1.0:
                with self._state_lock:
                    self._camera_fps = report_frames / (now - report_time)
                report_frames = 0
                report_time = now

    @staticmethod
    def _safe_result_value(result, method_name: str):
        try:
            return bool(getattr(result, method_name)())
        except Exception:
            return None

    @classmethod
    def _result_diagnostic(cls, result, valid=None, succeeded=None) -> str:
        """Read every diagnostic field pylon exposes without masking the error."""
        if result is None:
            return "kein Grab-Ergebnis"
        values = {
            "valid": valid if valid is not None else cls._safe_result_value(result, "IsValid"),
            "succeeded": succeeded if succeeded is not None else cls._safe_result_value(result, "GrabSucceeded"),
            "error_code": cls._grab_error_code(result),
            "description": cls._grab_error_text(result),
        }
        for label, method_name in (
            ("block_id", "GetBlockID"),
            ("payload_size", "GetPayloadSize"),
            ("buffer_size", "GetBufferSize"),
        ):
            try:
                values[label] = getattr(result, method_name)()
            except Exception:
                values[label] = "unbekannt"
        return ", ".join(f"{key}={value}" for key, value in values.items())

    @staticmethod
    def _get_feature(camera, name: str):
        try:
            node = getattr(camera, name)
            getter = getattr(node, "GetValue", None)
            return getter() if callable(getter) else node.Value
        except Exception:
            return None

    @staticmethod
    def _read_resulting_fps(camera) -> Optional[float]:
        for name in ("BslResultingAcquisitionFrameRate", "ResultingFrameRate"):
            try:
                node = getattr(camera, name)
                value = node.GetValue() if hasattr(node, "GetValue") else node.Value
                return float(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _grab_error_code(result) -> str:
        try:
            return str(result.GetErrorCode())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _grab_error_text(result) -> str:
        try:
            return str(result.GetErrorDescription())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _looks_like_payload_error(message: str) -> bool:
        text = str(message).lower()
        return any(marker in text for marker in (
            "payload", "block id", "read operation failed", "transport",
        ))

    def _record_block_id(self, result) -> None:
        try:
            block_id = int(result.GetBlockID())
        except Exception:
            return
        if block_id <= 0:
            return
        with self._state_lock:
            if self._last_block_id is not None:
                # Block IDs are transport counters and may wrap or restart
                # after a USB reconnect. Only count a forward gap; never turn
                # a counter reset into an enormous unsigned drop count.
                gap = block_id - self._last_block_id - 1
                if 0 < gap <= 1000000:
                    self._dropped_frames += gap
            self._last_block_id = block_id
            self._last_success_block_id = block_id

    def _set_error(self, message: str) -> None:
        message = str(message)
        with self._state_lock:
            changed = message != self._last_error
            due = time.monotonic() - self._last_error_log_time >= 1.0
            self._last_error = message
            if changed or due:
                self._last_error_log_time = time.monotonic()
                should_log = bool(message)
            else:
                should_log = False
        if should_log:
            print(f"[CAMERA-PROCESS-ERR] {message}")

    def status(self) -> Dict[str, Any]:
        frame_ipc_status = self._frame_ipc_status()
        with self._state_lock:
            capture_thread_alive = bool(
                self._thread is not None and self._thread.is_alive()
            )
            publisher_thread_alive = bool(
                self._publisher_thread is not None
                and self._publisher_thread.is_alive()
            )
            return {
                "camera_open": self._camera_open,
                "grabbing": self._camera_grabbing,
                "model": self._model,
                "pixel_format": self.settings.pixel_format,
                "pixel_color_filter": self._get_feature(self.camera, "PixelColorFilter")
                if self.camera is not None
                else None,
                "available_pixel_formats": list(self._available_pixel_formats),
                "requested_fps": float(self.settings.force_fps),
                "resulting_fps": self._resulting_fps,
                "camera_fps": round(self._camera_fps, 2),
                "camera_frames_total": self._camera_frames_total,
                "failed_grabs": self._failed_grabs,
                "payload_errors": self._payload_errors,
                "dropped_frames": self._dropped_frames,
                "reconnects": self._reconnects,
                "last_error": self._last_error,
                "last_failure_phase": self._last_failure_phase,
                "last_result_diagnostic": self._last_result_diagnostic,
                "capture_aborted": self._capture_aborted,
                "capture_thread_alive": capture_thread_alive,
                "publisher_thread_alive": publisher_thread_alive,
                "capture_handoff_dropped": self._capture_handoff_dropped,
                "last_failure_age_seconds": (
                    None if not self._last_failure_time else
                    max(0.0, time.monotonic() - self._last_failure_time)
                ),
                "last_success_block_id": self._last_success_block_id,
                "last_frame_age_seconds": (
                    None if not self._last_frame_time else
                    max(0.0, time.monotonic() - self._last_frame_time)
                ),
                "settings": dict(vars(self.settings)),
                "frame_ipc": frame_ipc_status,
                "frame_publishing_enabled": self.publish_frames,
            }

    def _frame_ipc_status(self) -> Dict[str, Any]:
        with self._frame_ring_lock:
            return {
                "name": self.frame_ipc_name,
                "slots": self.frame_ipc_slots,
                "available": True,
                "frames_published": self._frame_ipc_frames,
                "frame_ipc_dropped": self._frame_ipc_dropped,
                "errors": self._frame_ipc_errors,
                **self.frame_ring.stats(),
            }
