"""Direct, high-rate Basler source for v4.

The pylon grab loop deliberately runs in its own thread. It does only the
work that must happen while a grab result is owned by pylon: retrieve the
result, copy the Bayer image, release the result, and hand the copy to the
detector. Colour conversion and ball detection never run in this thread.

This is an in-process handoff. It does not use multiprocessing, shared
memory, a replay writer, or a web-stream queue.
"""

from __future__ import annotations

from collections import deque
import os
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np
from pypylon import pylon

from v4.tracking_kicker.Objects.Frames import Frame
from v4.tracking_kicker.model.Camera import Camera


class BaslerSource:
    """Continuously drain the Basler transport and expose copied frames."""

    RETRIEVE_TIMEOUT_MS = 1000
    DEFAULT_MAX_NUM_BUFFER = 256
    DEFAULT_MAX_TRANSFER_SIZE = 4194304
    DEFAULT_MAX_BUFFER_SIZE = 6291456
    DEFAULT_NUM_MAX_QUEUED_URBS = 64
    DEFAULT_HANDOFF_QUEUE_SIZE = 8

    def __init__(self, cfg: Camera):
        if cfg is None:
            raise ValueError("Eine Kamerakonfiguration ist erforderlich")

        self.cfg = cfg
        self.camera: Optional[Any] = None
        self.fps = float(getattr(cfg, "force_fps", 0.0) or 0.0)
        self.last_error = ""
        self._closed = False
        self._release_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._frames_condition = threading.Condition()
        self._frames: deque[Frame] = deque()
        self._handler_lock = threading.Lock()
        self._frame_handler = None
        self._handoff_queue_size = max(
            1,
            int(
                getattr(cfg, "handoff_queue_size", self.DEFAULT_HANDOFF_QUEUE_SIZE)
                or self.DEFAULT_HANDOFF_QUEUE_SIZE
            ),
        )
        self._handoff_frame_stride = max(
            1,
            int(getattr(cfg, "handoff_frame_stride", 1) or 1),
        )
        self._capture_thread: Optional[threading.Thread] = None

        self._stats_lock = threading.Lock()
        self._capture_started = time.perf_counter()
        self._capture_frames = 0
        self._capture_failures = 0
        self._payload_errors = 0
        self._handoff_dropped = 0
        self._last_block_id: Optional[str] = None
        self._last_grab_error = ""
        self._last_grab_error_time = 0.0
        self._last_logged_grab_error = ""

        try:
            self.camera = self._open_camera()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="v4-basler-capture",
                daemon=True,
            )
            self._capture_thread.start()
        except Exception:
            self._closed = True
            camera = self.camera
            self.camera = None
            self._stop_and_close(camera)
            raise

    # ------------------------------------------------------------------
    # Camera discovery and configuration
    # ------------------------------------------------------------------

    def _open_camera(self):
        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if not devices:
            raise RuntimeError(
                "Keine Basler-Kamera gefunden (pypylon EnumerateDevices=0)"
            )

        wanted_serial = str(getattr(self.cfg, "serial_number", "") or "").strip()
        if wanted_serial:
            device_info = next(
                (
                    item
                    for item in devices
                    if str(item.GetSerialNumber()) == wanted_serial
                ),
                None,
            )
            if device_info is None:
                available = ", ".join(
                    str(item.GetSerialNumber()) for item in devices
                )
                raise RuntimeError(
                    f"Basler-Kamera mit Seriennummer {wanted_serial!r} nicht gefunden; "
                    f"verfügbar: {available or 'unbekannt'}"
                )
        else:
            device_info = devices[0]

        camera = pylon.InstantCamera(factory.CreateDevice(device_info))
        try:
            # Do not let pylon's default configuration overwrite the explicit
            # trigger/acquisition settings when the camera is opened.
            camera.RegisterConfiguration(
                None,
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_None,
            )
            camera.Open()
            self._configure_camera(camera)
            camera.StartGrabbing(self._grab_strategy())
            self._log_configuration(camera, device_info)
            return camera
        except Exception:
            self._stop_and_close(camera)
            raise

    def _configure_camera(self, camera) -> None:
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
        self._set_feature(
            camera, "ExposureTime", float(self.cfg.exposure_time), required=True
        )
        self._set_feature(camera, "Gain", float(self.cfg.gain), required=True)

        # Set offsets to zero before changing the ROI. This prevents an
        # invalid intermediate ROI when a previous application used another
        # image size.
        self._set_feature(camera, "OffsetX", 0, required=True)
        self._set_feature(camera, "OffsetY", 0, required=True)
        self._set_feature(camera, "Width", int(self.cfg.width), required=True)
        self._set_feature(camera, "Height", int(self.cfg.height), required=True)
        self._set_feature(camera, "OffsetX", int(self.cfg.offset_x), required=True)
        self._set_feature(camera, "OffsetY", int(self.cfg.offset_y), required=True)

        max_buffers = max(
            1,
            int(
                getattr(self.cfg, "max_num_buffer", 0)
                or self.DEFAULT_MAX_NUM_BUFFER
            ),
        )
        try:
            # MaxNumBuffer is an InstantCamera property in pypylon, rather
            # than a regular camera node on all supported pylon versions.
            setattr(camera, "MaxNumBuffer", max_buffers)
            print(f"pylon MaxNumBuffer = {max_buffers}", flush=True)
        except Exception as exc:
            print(f"[CAMERA-WARN] MaxNumBuffer nicht setzbar: {exc}", flush=True)

        throughput_mode = str(
            getattr(self.cfg, "device_link_throughput_limit_mode", "Off") or "Off"
        )
        self._set_feature(
            camera,
            "DeviceLinkThroughputLimitMode",
            throughput_mode,
            required=False,
        )
        throughput_limit = int(
            getattr(self.cfg, "device_link_throughput_limit", 0) or 0
        )
        if throughput_limit > 0 and throughput_mode.lower() == "on":
            self._set_feature(
                camera,
                "DeviceLinkThroughputLimit",
                throughput_limit,
                required=False,
            )

        # These are host-side USB3 Vision stream settings. The values follow
        # Basler's USB tuning guidance for payload/LibUSB errors; a user value
        # can still override them through config.json.
        max_transfer_size = int(
            getattr(self.cfg, "max_transfer_size", 0)
            or self.DEFAULT_MAX_TRANSFER_SIZE
        )
        max_buffer_size = int(
            getattr(self.cfg, "max_buffer_size", 0)
            or self.DEFAULT_MAX_BUFFER_SIZE
        )
        num_max_queued_urbs = int(
            getattr(self.cfg, "num_max_queued_urbs", 0)
            or self.DEFAULT_NUM_MAX_QUEUED_URBS
        )
        self._set_stream_parameter(camera, "MaxBufferSize", max_buffer_size)
        self._set_stream_parameter(camera, "MaxTransferSize", max_transfer_size)
        self._set_stream_parameter(camera, "NumMaxQueuedUrbs", num_max_queued_urbs)
        priority = int(getattr(self.cfg, "transfer_loop_thread_priority", 0) or 0)
        if priority > 0:
            self._set_stream_parameter(
                camera, "TransferLoopThreadPriority", priority
            )

        requested_fps = float(getattr(self.cfg, "force_fps", 0.0) or 0.0)
        rate_enabled = bool(getattr(self.cfg, "acquisition_frame_rate_enable", False))
        if rate_enabled and requested_fps > 0:
            self._set_feature(
                camera, "AcquisitionFrameRateEnable", True, required=False
            )
            self._set_feature(
                camera, "AcquisitionFrameRate", requested_fps, required=False
            )
        else:
            # Explicitly disable the feature. Leaving this untouched makes
            # the camera retain a previous 120 FPS setting across restarts,
            # even when the new config requests the camera's full rate.
            self._set_feature(
                camera, "AcquisitionFrameRateEnable", False, required=False
            )

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
            print(f"[CAMERA-WARN] {message}", flush=True)
            return False

    @staticmethod
    def _set_stream_parameter(camera, name: str, value: int) -> bool:
        if int(value or 0) <= 0:
            return False
        try:
            node = camera.GetStreamGrabberNodeMap().GetNode(name)
            if node is None:
                raise RuntimeError("Knoten nicht vorhanden")
            setter = getattr(node, "SetValue", None)
            if callable(setter):
                setter(int(value))
            else:
                node.Value = int(value)
            print(f"pylon {name} = {int(value)}", flush=True)
            return True
        except Exception as exc:
            print(f"[CAMERA-WARN] {name} nicht setzbar: {exc}", flush=True)
            return False

    @staticmethod
    def _get_feature(camera, name: str):
        try:
            node = getattr(camera, name)
            getter = getattr(node, "GetValue", None)
            return getter() if callable(getter) else node.Value
        except Exception:
            return None

    @staticmethod
    def _grab_strategy():
        # LatestImageOnly returns the newest complete result and immediately
        # recycles older completed buffers. This is the stable pylon strategy
        # for this USB3 camera at its full rate; the dedicated receiver still
        # drains the stream continuously, so normal operation does not wait
        # for the detector.
        return pylon.GrabStrategy_LatestImageOnly

    def _log_configuration(self, camera, device_info) -> None:
        print(
            f"✅ Basler geöffnet: {device_info.GetModelName()} "
            f"(S/N {device_info.GetSerialNumber()})",
            flush=True,
        )
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
            "DeviceLinkThroughputLimitMode",
            "DeviceLinkThroughputLimit",
            "BslResultingAcquisitionFrameRate",
            "ResultingFrameRate",
        ):
            value = self._get_feature(camera, name)
            if value is not None:
                print(f"pylon {name} = {value}", flush=True)
        for name in ("MaxTransferSize", "NumMaxQueuedUrbs"):
            try:
                value = camera.GetStreamGrabberNodeMap().GetNode(name).Value
            except Exception:
                value = None
            if value is not None:
                print(f"pylon {name} = {value}", flush=True)
        try:
            resulting = float(
                self._get_feature(camera, "BslResultingAcquisitionFrameRate")
                or self._get_feature(camera, "ResultingFrameRate")
                or 0.0
            )
            if resulting > 0:
                self.fps = resulting
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Dedicated capture thread and handoff
    # ------------------------------------------------------------------

    def set_frame_handler(self, handler) -> None:
        """Process a grab while its pylon buffer is still owned by pylon.

        The handler must finish before the current grab result is released
        and must not retain ``image`` after returning. This path avoids a
        second full raw-image copy when v4 is running the detector only.
        Passing ``None`` restores the copied-frame ``read()`` handoff.
        """
        with self._handler_lock:
            self._frame_handler = handler

    def _get_frame_handler(self):
        with self._handler_lock:
            return self._frame_handler

    def _capture_loop(self) -> None:
        camera = self.camera
        self._pin_capture_thread()
        try:
            while not self._closed and camera is not None and camera.IsGrabbing():
                grab = None
                try:
                    grab = camera.RetrieveResult(
                        self.RETRIEVE_TIMEOUT_MS,
                        pylon.TimeoutHandling_Return,
                    )
                    if grab is None:
                        if self._closed:
                            break
                        self._record_grab_failure("RetrieveResult lieferte None")
                        continue

                    if not grab.IsValid():
                        if self._closed:
                            break
                        self._record_grab_failure(
                            "Grab-Ergebnis ungültig: " + self._grab_diagnostic(grab),
                            payload=True,
                        )
                        continue

                    if not grab.GrabSucceeded():
                        if self._closed:
                            break
                        self._record_grab_failure(
                            "Grab fehlgeschlagen: " + self._grab_diagnostic(grab),
                            payload=True,
                        )
                        continue

                    image = grab.GetArray()
                    if image is None:
                        self._record_grab_failure("Grab-Ergebnis enthält kein Bild")
                        continue

                    timestamp = time.perf_counter()
                    with self._stats_lock:
                        self._capture_frames += 1
                        capture_number = self._capture_frames
                        self._last_block_id = self._safe_block_id(grab)
                        self.last_error = ""
                    handler = self._get_frame_handler()
                    if handler is not None:
                        # The handler owns only this callback duration. It
                        # runs before Release() so it may inspect pylon's
                        # buffer without another complete image copy.
                        handler(image, timestamp)
                    else:
                        if (
                            self._handoff_frame_stride > 1
                            and capture_number % self._handoff_frame_stride
                        ):
                            continue
                        frame = Frame()
                        # The array is backed by pylon's result buffer. It
                        # must be copied before Release() returns that buffer.
                        frame.frame = np.asarray(image).copy()
                        frame.timestamp = timestamp
                        self._handoff(frame)
                except Exception as exc:
                    if self._closed:
                        break
                    text = str(exc)
                    self._record_grab_failure(
                        f"RetrieveResult/Framekopie fehlgeschlagen: {text}",
                        payload="payload" in text.lower()
                        or "discard" in text.lower(),
                    )
                    if not self._closed:
                        time.sleep(0.001)
                finally:
                    if grab is not None:
                        try:
                            grab.Release()
                        except Exception as exc:
                            if not self._closed:
                                self._record_grab_failure(
                                    f"Grab-Ergebnis konnte nicht freigegeben werden: {exc}"
                                )
        except Exception as exc:
            self._record_grab_failure(f"Kamera-Capture-Thread abgebrochen: {exc}")
        finally:
            with self._frames_condition:
                self._frames_condition.notify_all()
            # The capture thread is the only thread that can be inside
            # RetrieveResult(), so it also owns normal pylon cleanup.
            self._stop_and_close(camera)

    def _pin_capture_thread(self) -> None:
        """Give USB reception a dedicated CPU on Linux when configured."""
        if not bool(getattr(self.cfg, "reserve_capture_cpu", True)):
            return
        if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
            return
        value = getattr(self.cfg, "capture_cpu", None)
        if value is None:
            return
        try:
            cpu = int(value)
            os.sched_setaffinity(threading.get_native_id(), {cpu})
            print(f"[CAMERA] Capture-Thread auf CPU {cpu} fixiert", flush=True)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[CAMERA-WARN] Capture-CPU nicht setzbar: {exc}", flush=True)

    def _handoff(self, frame: Frame) -> None:
        with self._frames_condition:
            if len(self._frames) >= self._handoff_queue_size:
                self._frames.popleft()
                with self._stats_lock:
                    self._handoff_dropped += 1
            self._frames.append(frame)
            self._frames_condition.notify()

    def read(self) -> Tuple[bool, Optional[Frame]]:
        deadline = time.monotonic() + 1.1
        with self._frames_condition:
            while not self._frames and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._frames_condition.wait(timeout=remaining)
            if self._frames:
                return True, self._frames.popleft()
        return False, None

    def _record_grab_failure(self, message: str, *, payload: bool = False) -> None:
        now = time.perf_counter()
        with self._stats_lock:
            self._capture_failures += 1
            if payload:
                self._payload_errors += 1
            self._last_grab_error = str(message)
            self.last_error = str(message)
            should_log = (
                message != self._last_logged_grab_error
                or now - self._last_grab_error_time >= 1.0
            )
            if should_log:
                self._last_logged_grab_error = str(message)
                self._last_grab_error_time = now
        if should_log:
            print(f"[CAMERA-ERR] {message}", flush=True)

    @staticmethod
    def _safe_block_id(grab) -> Optional[str]:
        try:
            return str(grab.GetBlockID())
        except Exception:
            return None

    @staticmethod
    def _grab_diagnostic(grab) -> str:
        values = []
        for name, method in (
            ("valid", "IsValid"),
            ("succeeded", "GrabSucceeded"),
            ("error_code", "GetErrorCode"),
            ("description", "GetErrorDescription"),
            ("block_id", "GetBlockID"),
            ("payload_size", "GetPayloadSize"),
            ("buffer_size", "GetBufferSize"),
        ):
            try:
                values.append(f"{name}={getattr(grab, method)()}")
            except Exception:
                values.append(f"{name}=unbekannt")
        return ", ".join(values)

    # ------------------------------------------------------------------
    # Status and shutdown
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        now = time.perf_counter()
        with self._stats_lock:
            elapsed = max(0.001, now - self._capture_started)
            return {
                "camera_frames_total": int(self._capture_frames),
                "capture_fps": float(self._capture_frames / elapsed),
                "failed_grabs": int(self._capture_failures),
                "payload_errors": int(self._payload_errors),
                "handoff_dropped": int(self._handoff_dropped),
                "handoff_queued": len(self._frames),
                "handoff_frame_stride": int(self._handoff_frame_stride),
                "last_block_id": self._last_block_id,
                "last_error": str(self._last_grab_error or ""),
                "resulting_fps": float(self.fps),
                "capture_thread_alive": bool(
                    self._capture_thread is not None
                    and self._capture_thread.is_alive()
                ),
            }

    def showFPS(self) -> str:
        try:
            for name in ("BslResultingAcquisitionFrameRate", "ResultingFrameRate"):
                value = self._get_feature(self.camera, name)
                if value is not None and float(value) > 0:
                    self.fps = float(value)
                    break
        except (TypeError, ValueError):
            pass
        return f"{self.fps:.2f}"

    def reset(self) -> None:
        return

    def getTemperatur(self, camera=None) -> str:
        for name in ("DeviceTemperature", "BslTemperature"):
            value = self._get_feature(self.camera, name)
            if value is not None:
                return str(value)
        return "unbekannt"

    def release(self) -> None:
        with self._release_lock:
            if self._closed:
                return
            self._closed = True
            with self._frames_condition:
                self._frames_condition.notify_all()
            capture_thread = self._capture_thread

        camera = self.camera
        cleanup_thread = threading.Thread(
            target=self._stop_and_close,
            args=(camera,),
            name="v4-basler-cleanup",
            daemon=True,
        )
        cleanup_thread.start()
        if capture_thread is not None and capture_thread.is_alive():
            capture_thread.join(timeout=2.0)
        cleanup_thread.join(timeout=2.0)
        if (
            (capture_thread is not None and capture_thread.is_alive())
            or cleanup_thread.is_alive()
        ):
            print(
                "[SHUTDOWN] pylon-Cleanup blockiert; "
                "daemonisierte Cleanup-Threads laufen weiter",
                flush=True,
            )
        else:
            print("[SHUTDOWN] Basler sauber gestoppt und geschlossen", flush=True)
        final = self.stats()
        print(
            "[CAMERA] Capture final: "
            f"frames={final['camera_frames_total']}, "
            f"fps={final['capture_fps']:.2f}, "
            f"failed_grabs={final['failed_grabs']}, "
            f"payload_errors={final['payload_errors']}, "
            f"handoff_dropped={final['handoff_dropped']}",
            flush=True,
        )
        self.camera = None

    def _stop_and_close(self, camera) -> None:
        if camera is None:
            return
        with self._cleanup_lock:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            except Exception as exc:
                print(
                    f"[CAMERA-WARN] StopGrabbing fehlgeschlagen: {exc}",
                    flush=True,
                )
            try:
                if camera.IsOpen():
                    camera.Close()
            except Exception as exc:
                print(
                    f"[CAMERA-WARN] Kamera schließen fehlgeschlagen: {exc}",
                    flush=True,
                )
