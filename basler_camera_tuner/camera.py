from __future__ import annotations

import threading
import time
from dataclasses import fields
from typing import Any, Dict, Optional

import cv2
import numpy as np
from pypylon import pylon

from .settings import BallDetectionSettings, CameraSettings


class BaslerCamera:
    """Small, independent Basler preview and settings controller.

    The camera is accessed by one capture thread. Settings are applied by
    stopping and reopening that stream, so the same values are also used after
    a transport error and reconnect.
    """

    RECOVERY_FAILURE_THRESHOLD = 5
    RETRY_SECONDS = 2.0

    def __init__(
        self,
        settings: CameraSettings,
        ball_settings: Optional[BallDetectionSettings] = None,
    ):
        self.settings = settings
        self.ball_settings = ball_settings or BallDetectionSettings()
        self.camera = None
        self._camera_lock = threading.RLock()
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._latest_jpeg: Optional[bytes] = None
        self._latest_ball_jpeg: Optional[bytes] = None
        self._latest_mask_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_number = 0
        self._received_frames = 0
        self._failed_grabs = 0
        self._measured_fps = 0.0
        self._last_error = ""
        self._model = ""
        self._last_frame_time = 0.0
        self._last_ball = None

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="basler-tuner-capture",
            daemon=True,
        )
        self._capture_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_camera()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)

    def reconnect(self) -> bool:
        """Apply current settings by reopening the camera stream."""
        with self._camera_lock:
            if self._stop_event.is_set():
                return False
            return self._open_camera_locked()

    def update_settings(self, values: Dict[str, Any]) -> bool:
        """Validate settings, store them, and restart the stream."""
        for name, value in values.items():
            if not hasattr(self.settings, name):
                continue
            current = getattr(self.settings, name)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"Ungueltiger Wert fuer {name}: {value!r}")

            if name in {"width", "height", "max_num_buffer"} and value <= 0:
                raise ValueError(f"{name} muss groesser als 0 sein")
            if name == "force_fps" and value < 0:
                raise ValueError("force_fps darf nicht negativ sein")
            if name in {"offset_x", "offset_y", "packet_size", "inter_packet_delay"} and value < 0:
                raise ValueError(f"{name} darf nicht negativ sein")
            if name == "grab_strategy" and value not in {
                "latest_image_only",
                "one_by_one",
            }:
                raise ValueError("grab_strategy muss latest_image_only oder one_by_one sein")

            setattr(self.settings, name, value)

        return self.reconnect()

    def update_ball_settings(self, values: Dict[str, Any]) -> None:
        """Update the live color/contour preview without reopening the camera."""
        valid_names = {item.name for item in fields(BallDetectionSettings)}
        updated = dict(vars(self.ball_settings))
        for name, value in values.items():
            if name not in valid_names:
                continue
            current = updated[name]
            try:
                if isinstance(current, int):
                    value = int(value)
                else:
                    value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Ungueltiger Wert fuer {name}: {value!r}")
            updated[name] = value

        candidate = BallDetectionSettings(**updated)

        integer_ranges = {
            "hsv_low_h": (0, 179), "hsv_high_h": (0, 179),
            "hsv_low_s": (0, 255), "hsv_high_s": (0, 255),
            "hsv_low_v": (0, 255), "hsv_high_v": (0, 255),
        }
        for name, (minimum, maximum) in integer_ranges.items():
            value = getattr(candidate, name)
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen")

        if candidate.hsv_low_h > candidate.hsv_high_h:
            raise ValueError("hsv_low_h darf nicht groesser als hsv_high_h sein")
        if candidate.hsv_low_s > candidate.hsv_high_s:
            raise ValueError("hsv_low_s darf nicht groesser als hsv_high_s sein")
        if candidate.hsv_low_v > candidate.hsv_high_v:
            raise ValueError("hsv_low_v darf nicht groesser als hsv_high_v sein")
        if candidate.area_min < 0 or candidate.area_max <= candidate.area_min:
            raise ValueError("area_max muss groesser als area_min sein")
        if candidate.radius_min_px < 0 or candidate.radius_max_px <= candidate.radius_min_px:
            raise ValueError("radius_max_px muss groesser als radius_min_px sein")
        if not 0.05 <= candidate.detection_scale <= 1.0:
            raise ValueError("detection_scale muss zwischen 0.05 und 1.0 liegen")
        if not 0.0 <= candidate.circularity_min <= 1.0:
            raise ValueError("circularity_min muss zwischen 0 und 1 liegen")
        if not 0.0 <= candidate.solidity_min <= 1.0:
            raise ValueError("solidity_min muss zwischen 0 und 1 liegen")

        self.ball_settings = candidate

    # ------------------------------------------------------------------
    # Configuration: this is deliberately explicit so every reconnect uses
    # exactly the values visible in the tuning UI/settings.json.
    # ------------------------------------------------------------------
    def _configure_reconnected_camera(self, camera) -> None:
        """Apply all editable settings to a newly opened Basler camera."""
        settings = self.settings

        self.try_set(camera, "ExposureAuto", settings.exposure_auto)
        self.try_set(camera, "PixelFormat", settings.pixel_format)
        self.try_set(camera, "GainAuto", settings.gain_auto)
        self.try_set(camera, "ExposureTime", settings.exposure_time)
        self.try_set(camera, "Gain", settings.gain)

        # Width/height are set before offsets because Basler ROI nodes usually
        # validate the offset against the currently selected ROI size.
        self.try_set(camera, "Width", settings.width)
        self.try_set(camera, "Height", settings.height)
        self.try_set(camera, "OffsetX", settings.offset_x)
        self.try_set(camera, "OffsetY", settings.offset_y)

        try:
            camera.MaxNumBuffer = settings.max_num_buffer
        except Exception as exc:
            self._set_error(f"MaxNumBuffer nicht setzbar: {exc}")

        self.try_set(
            camera,
            "AcquisitionFrameRateEnable",
            settings.force_fps > 0,
        )
        if settings.force_fps > 0:
            self.try_set(camera, "AcquisitionFrameRate", settings.force_fps)

        # These nodes exist on GigE cameras. They are optional so the tuner
        # also works with USB or cameras whose transport layer has no manual
        # packet controls.
        if settings.packet_size > 0:
            self.try_set(camera, "GevSCPSPacketSize", settings.packet_size)
        if settings.inter_packet_delay > 0:
            self.try_set(camera, "GevSCPD", settings.inter_packet_delay)

    def _grab_strategy(self):
        if self.settings.grab_strategy == "one_by_one":
            return pylon.GrabStrategy_OneByOne
        return pylon.GrabStrategy_LatestImageOnly

    def _open_camera_locked(self) -> bool:
        self._close_camera_locked()
        camera = None
        try:
            device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            camera = pylon.InstantCamera(device)
            camera.Open()
            self._configure_reconnected_camera(camera)
            camera.StartGrabbing(self._grab_strategy())
            self.camera = camera
            self._model = str(camera.GetDeviceInfo().GetModelName())
            self._last_error = ""
            print(f"[TUNER] Basler geoeffnet: {self._model}")
            return True
        except Exception as exc:
            self._set_error(f"Kamera konnte nicht geoeffnet werden: {exc}")
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()
            except Exception:
                pass
            self.camera = None
            return False

    def _close_camera(self) -> None:
        with self._camera_lock:
            self._close_camera_locked()

    def _close_camera_locked(self) -> None:
        camera = self.camera
        self.camera = None
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Capture and preview data
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        failures = 0
        report_time = time.monotonic()
        report_frames = 0

        while not self._stop_event.is_set():
            with self._camera_lock:
                if self.camera is None:
                    opened = self._open_camera_locked()
                    if not opened:
                        # Do not spin when no device is connected.
                        pass
                camera = self.camera

                grab = None
                if camera is not None:
                    try:
                        grab = camera.RetrieveResult(
                            100,
                            pylon.TimeoutHandling_Return,
                        )
                        if grab is None:
                            raise RuntimeError("RetrieveResult liefert None")
                        if not grab.IsValid() or not grab.GrabSucceeded():
                            code = self._grab_error_code(grab)
                            message = self._grab_error_text(grab)
                            raise RuntimeError(f"Grab {code}: {message}")

                        image = grab.GetArray().copy()
                        failures = 0
                    except Exception as exc:
                        failures += 1
                        self._failed_grabs += 1
                        self._set_error(str(exc))
                        image = None
                    finally:
                        if grab is not None:
                            try:
                                grab.Release()
                            except Exception:
                                pass
                else:
                    image = None

            if image is not None:
                self._publish_frame(image)
                report_frames += 1
            elif failures >= self.RECOVERY_FAILURE_THRESHOLD:
                self._recover_after_failure()
                failures = 0
            else:
                time.sleep(0.02)

            now = time.monotonic()
            if now - report_time >= 1.0:
                self._measured_fps = report_frames / (now - report_time)
                report_frames = 0
                report_time = now

    def _recover_after_failure(self) -> None:
        if self._stop_event.is_set():
            return
        print("[TUNER] Streamfehler, Kamera wird neu verbunden")
        with self._camera_lock:
            self._open_camera_locked()
        time.sleep(self.RETRY_SECONDS)

    def _publish_frame(self, image: np.ndarray) -> None:
        pixel_format = self.settings.pixel_format.lower()
        if image.ndim == 2:
            if pixel_format.startswith("bayer"):
                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        ball_preview, mask_preview, ball = self._create_ball_preview(image)
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        preview_ok, ball_encoded = cv2.imencode(".jpg", ball_preview)
        mask_ok, mask_encoded = cv2.imencode(".jpg", mask_preview)
        if not ok or not preview_ok or not mask_ok:
            return

        now = time.monotonic()
        with self._latest_lock:
            self._latest_frame = image
            self._latest_jpeg = encoded.tobytes()
            self._latest_ball_jpeg = ball_encoded.tobytes()
            self._latest_mask_jpeg = mask_encoded.tobytes()
            self._last_ball = ball
            self._frame_number += 1
            self._received_frames += 1
            self._last_frame_time = now

    def latest_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_jpeg

    def latest_ball_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_ball_jpeg

    def latest_mask_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_mask_jpeg

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._latest_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._camera_lock:
            camera = self.camera
            is_open = bool(camera is not None and camera.IsOpen())
            is_grabbing = bool(camera is not None and camera.IsGrabbing())
        with self._latest_lock:
            frame_number = self._frame_number
            last_frame_age = (
                None
                if self._last_frame_time == 0.0
                else max(0.0, time.monotonic() - self._last_frame_time)
            )
        return {
            "camera_open": is_open,
            "grabbing": is_grabbing,
            "model": self._model,
            "received_frames": self._received_frames,
            "frame_number": frame_number,
            "failed_grabs": self._failed_grabs,
            "measured_fps": round(self._measured_fps, 2),
            "last_frame_age_seconds": last_frame_age,
            "last_error": self._last_error,
            "settings": dict(vars(self.settings)),
            "ball_detection": dict(vars(self.ball_settings)),
            "detected_ball": self._last_ball,
        }

    def _create_ball_preview(self, image: np.ndarray):
        settings = self.ball_settings
        scale = settings.detection_scale
        detection_image = image
        if 0.0 < scale < 1.0:
            detection_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(
            cv2.GaussianBlur(detection_image, (5, 5), 0),
            cv2.COLOR_BGR2HSV,
        )
        lower = np.array(
            [settings.hsv_low_h, settings.hsv_low_s, settings.hsv_low_v],
            dtype=np.uint8,
        )
        upper = np.array(
            [settings.hsv_high_h, settings.hsv_high_s, settings.hsv_high_v],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates = []
        area_scale = scale * scale
        for contour in contours:
            area = cv2.contourArea(contour)
            if not settings.area_min * area_scale <= area <= settings.area_max * area_scale:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not settings.radius_min_px * scale < radius < settings.radius_max_px * scale:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
            if circularity < settings.circularity_min:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < settings.solidity_min:
                continue
            candidates.append((area, x / scale, y / scale, radius / scale))

        preview = image.copy()
        selected_ball = None
        if candidates:
            _, x, y, radius = max(candidates, key=lambda candidate: candidate[0])
            selected_ball = [round(x), round(y), round(radius)]
            cv2.circle(
                preview,
                (int(round(x)), int(round(y))),
                max(2, int(round(radius))),
                (0, 255, 255),
                3,
            )
            cv2.putText(
                preview,
                f"BALL x={int(x)} y={int(y)} r={int(radius)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            preview,
            f"Kandidaten: {len(candidates)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return preview, mask_bgr, selected_ball

    def _set_error(self, message: str) -> None:
        self._last_error = str(message)
        print(f"[TUNER] {self._last_error}")

    @staticmethod
    def _grab_error_code(grab) -> str:
        try:
            return str(grab.GetErrorCode())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _grab_error_text(grab) -> str:
        try:
            return str(grab.GetErrorDescription())
        except Exception:
            return "unbekannt"

    @staticmethod
    def try_set(camera, name: str, value: Any) -> bool:
        try:
            node = getattr(camera, name)
            node.Value = value
            print(f"[TUNER] {name} = {value}")
            return True
        except Exception as exc:
            print(f"[TUNER] {name} nicht setzbar: {exc}")
            return False
