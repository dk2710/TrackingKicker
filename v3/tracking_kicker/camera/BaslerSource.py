"""Basler/pypylon image source.

Only this module knows about pypylon.  The rest of the application receives
plain :class:`~tracking_kicker.Objects.Frames.Frame` objects and can therefore
also run against :mod:`.RawReplaySource`.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, Optional

from pypylon import pylon

from ..Objects.Frames import Frame
from .Camera import Camera


class BaslerSource:
    """Acquire frames from one Basler camera through pypylon.

    The source follows the pypylon user-allocated grab pattern used in the
    documentation: ``StartGrabbing`` is called once, each result is consumed
    with ``RetrieveResult`` and released in a ``finally`` block.  A serial
    number can be configured so a reconnect cannot accidentally select another
    USB camera.
    """

    RETRIEVE_TIMEOUT_MS = 100
    RECOVERY_FAILURE_THRESHOLD = 5
    RECOVERY_INTERVAL_SECONDS = 5.0

    def __init__(self, cfg: Camera):
        if cfg is None:
            raise ValueError("Eine Kamerakonfiguration ist erforderlich")

        self.cfg = cfg
        self.camera: Optional[Any] = None
        self._camera_lock = threading.RLock()
        self._closed = False

        self._read_count = 0
        self._read_failures = 0
        self._read_log_time = time.perf_counter()
        self._last_grab_error: Optional[str] = None
        self._last_grab_error_time = 0.0
        self._consecutive_grab_failures = 0
        self._last_recovery_time = 0.0
        self.fps = float(getattr(cfg, "force_fps", 0) or 0)

        try:
            self.camera = self._open_camera()
        except Exception:
            # Open() may have succeeded before a feature or ROI failed.  The
            # partially initialized camera must still be closed here.
            self.camera = None
            raise

    # ------------------------------------------------------------------
    # Device discovery and configuration
    # ------------------------------------------------------------------

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
        factory, device_info = self._select_device_info()
        camera = pylon.InstantCamera(factory.CreateDevice(device_info))

        try:
            camera.Open()
            self._configure_camera(camera)
            camera.StartGrabbing(self._grab_strategy())
            print(f"[CAMERA] Basler geoeffnet: {self._device_label(device_info)}")
            self._log_configuration(camera)
            return camera
        except Exception:
            self._stop_and_close(camera)
            raise

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
            if self._closed or self.camera is None:
                return False, None

            camera = self.camera
            grab_result = None
            try:
                if not camera.IsGrabbing():
                    self._log_grab_error("Kamera grabbt nicht (IsGrabbing=False)")
                    return False, None

                grab_result = camera.RetrieveResult(
                    self.RETRIEVE_TIMEOUT_MS,
                    pylon.TimeoutHandling_Return,
                )
                if grab_result is None:
                    self._log_grab_error("RetrieveResult liefert None")
                    return False, None

                if not grab_result.IsValid():
                    self._log_grab_error(
                        "Grab-Ergebnis ungueltig: "
                        f"code={self._grab_error_code(grab_result)} "
                        f"text={self._grab_error_text(grab_result)}"
                    )
                    return False, None

                if not grab_result.GrabSucceeded():
                    self._log_grab_error(
                        "Grab fehlgeschlagen: "
                        f"code={self._grab_error_code(grab_result)} "
                        f"text={self._grab_error_text(grab_result)}"
                    )
                    return False, None

                image = grab_result.GetArray()
                if image is None:
                    self._log_grab_error("Grab-Ergebnis enthaelt kein Bild")
                    return False, None

                frame = Frame()
                # pypylon's array is backed by the grab result.  Copy before
                # Release(), otherwise a later grab can overwrite the image.
                frame.frame = image.copy()
                frame.timestamp = time.perf_counter()
                return True, frame
            except Exception as exc:
                self._log_grab_error(f"Basler-Grab fehlgeschlagen: {exc}")
                return False, None
            finally:
                if grab_result is not None:
                    try:
                        grab_result.Release()
                    except Exception as exc:
                        self._log_grab_error(f"Grab-Ergebnis konnte nicht freigegeben werden: {exc}")

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

    def _log_grab_error(self, message: str) -> None:
        now = time.perf_counter()
        if message != self._last_grab_error or now - self._last_grab_error_time >= 1.0:
            print(f"[CAM-ERR] {message}")
            self._last_grab_error = message
            self._last_grab_error_time = now

    def read(self):
        ok, frame = self._grab_raw_once()
        now = time.perf_counter()

        if ok and frame is not None:
            self._read_count += 1
            self._consecutive_grab_failures = 0
        else:
            self._read_failures += 1
            self._consecutive_grab_failures += 1
            if self._consecutive_grab_failures >= self.RECOVERY_FAILURE_THRESHOLD:
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
            self._stop_and_close(old_camera)
            # Close() stops acquisition, while DestroyDevice() detaches the
            # transport-layer device.  The explicit detach is important on
            # USB: otherwise the old InstantCamera can keep the device node
            # busy even though pylon reports it as closed.
            del old_camera
            gc.collect()

            if self._closed:
                return

            try:
                self.camera = self._open_camera()
                self._consecutive_grab_failures = 0
                print("[CAM-RECOVER] Kamera neu verbunden")
            except Exception as exc:
                print(f"[CAM-RECOVER] Kamera-Reconnect fehlgeschlagen: {exc}")

    @staticmethod
    def _stop_and_close(camera) -> None:
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception as exc:
            print(f"[CAMERA] StopGrabbing fehlgeschlagen: {exc}")
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception as exc:
            print(f"[CAMERA] Close fehlgeschlagen: {exc}")
        try:
            attached = getattr(camera, "IsPylonDeviceAttached", None)
            if not callable(attached) or attached():
                destroy = getattr(camera, "DestroyDevice", None)
                if callable(destroy):
                    destroy()
        except Exception as exc:
            print(f"[CAMERA] DestroyDevice fehlgeschlagen: {exc}")

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

    def release(self) -> None:
        with self._camera_lock:
            if self._closed:
                return
            self._closed = True
            camera = self.camera
            self.camera = None
            self._stop_and_close(camera)
            print("[SHUTDOWN] Basler sauber gestoppt und geschlossen")

    close = release
