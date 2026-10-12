"""Browser tuner for frames produced by the TrackingKicker camera process."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import fields
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from ..camera_frame_ipc import DEFAULT_NAME, DEFAULT_SLOTS, SharedFrameRing
from ..Objects.Frames import Frame
from ..camera.conversion import to_bgr
from ..detection import BallDetector

from .settings import BallDetectionSettings, CameraSettings


class BaslerCamera:
    """Website-side tuner that never opens pypylon or the USB camera."""

    PREVIEW_FPS_LIMIT = 30.0

    def __init__(
        self,
        settings: CameraSettings,
        ball_settings: Optional[BallDetectionSettings] = None,
        *,
        ball_detection_enabled: bool = False,
        frame_ipc_name: str = DEFAULT_NAME,
        frame_ipc_slots: int = DEFAULT_SLOTS,
        control_url: str = "http://127.0.0.1:5011",
    ) -> None:
        # Keep OpenCV's work bounded in the website process. The historical
        # tracker used one OpenCV worker because multi-threaded blur/contour
        # processing competed with the high-rate camera transport.
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
        self.settings = settings
        self.ball_settings = ball_settings or BallDetectionSettings()
        self._ball_detection_enabled = bool(ball_detection_enabled)
        self._frame_ipc_name = str(frame_ipc_name)
        self._frame_ipc_slots = DEFAULT_SLOTS
        self._control_url = str(control_url).rstrip("/")
        self._detector_lock = threading.RLock()
        self._detector = self._build_detector()
        self._ring: Optional[SharedFrameRing] = None
        self._ring_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._detection_queue: Queue = Queue(maxsize=8)
        self._capture_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
        self._preview_thread: Optional[threading.Thread] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_sequence = 0
        self._raw_timestamp = 0.0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ball_jpeg: Optional[bytes] = None
        self._latest_mask_jpeg: Optional[bytes] = None
        self._latest_detection_mask: Optional[np.ndarray] = None
        self._latest_detection_sequence = 0
        self._last_ball = None
        self._latest_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._frame_number = 0
        self._received_frames = 0
        self._preview_fps = 0.0
        self._detection_attempts = 0
        self._detection_hits = 0
        self._detection_errors = 0
        self._detection_queue_dropped = 0
        self._detection_time_ms = 0.0
        self._detection_last_ms = 0.0
        self._detection_started = time.monotonic()
        self._dropped_frames = 0
        self._ipc_errors = 0
        self._last_error = ""
        self._last_error_time = 0.0
        self._remote_status: Dict[str, Any] = {}
        self._remote_last_error = ""

    def _build_detector(self) -> BallDetector:
        settings = self.ball_settings
        return BallDetector(
            hsv_low=(
                int(settings.hsv_low_h),
                int(settings.hsv_low_s),
                int(settings.hsv_low_v),
            ),
            hsv_high=(
                int(settings.hsv_high_h),
                int(settings.hsv_high_s),
                int(settings.hsv_high_v),
            ),
            area_min=float(settings.area_min),
            area_max=float(settings.area_max),
            detection_scale=float(settings.detection_scale),
            expected_fps=float(self.settings.force_fps),
            radius_min_px=float(settings.radius_min_px),
            radius_max_px=float(settings.radius_max_px),
            circularity_min=float(settings.circularity_min),
            solidity_min=float(settings.solidity_min),
            pixel_format=str(self.settings.pixel_format),
        )

    # ------------------------------------------------------------------
    # CameraProcess control API
    # ------------------------------------------------------------------
    def _request(self, path: str, *, method: str = "GET", payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self._control_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=0.8) as response:
            return json.loads(response.read().decode("utf-8"))

    def _read_remote_status(self) -> Dict[str, Any]:
        try:
            remote = self._request("/api/status")
            with self._stats_lock:
                self._remote_status = remote
            remote_error = str(remote.get("last_error", "") or "")
            if remote_error and remote_error != self._remote_last_error:
                self._remote_last_error = remote_error
                self._set_error(f"CameraProcess: {remote_error}")
            elif not remote_error and self._remote_last_error:
                self._remote_last_error = ""
                with self._stats_lock:
                    if self._last_error.startswith("CameraProcess:"):
                        self._last_error = ""
                        self._last_error_time = 0.0
            return remote
        except (OSError, ValueError, HTTPError, URLError) as exc:
            self._set_error(f"CameraProcess-Verbindung fehlgeschlagen: {exc}")
            with self._stats_lock:
                return dict(self._remote_status)

    def update_settings(self, values: Dict[str, Any]) -> bool:
        if "grab_strategy" in values and values["grab_strategy"] != "latest_image_only":
            raise ValueError("Die Tracking-Konfiguration verwendet LatestImageOnly")
        result = self._request("/api/settings", method="POST", payload={"camera": values})
        if isinstance(result.get("settings"), dict):
            updated = CameraSettings(**result["settings"])
            if updated != self.settings:
                self.settings = updated
                with self._detector_lock:
                    self._detector = self._build_detector()
        return bool(result.get("camera_open", False))

    def reconnect(self) -> bool:
        try:
            result = self._request("/api/reconnect", method="POST", payload={})
            return bool(result.get("camera_open", False))
        except (OSError, ValueError, HTTPError, URLError) as exc:
            self._set_error(f"Kamera-Reconnect fehlgeschlagen: {exc}")
            return False

    # ------------------------------------------------------------------
    # Consumer lifecycle and shared-memory reading
    # ------------------------------------------------------------------
    def start(self) -> None:
        # The CameraProcess is authoritative for acquisition settings. This
        # also makes the UI reflect settings changed outside the tuner.
        self._sync_remote_settings()
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._consume_loop, name="ball-tuner-shared-memory", daemon=True
        )
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="ball-tuner-preview", daemon=True
        )
        self._capture_thread.start()
        self._detection_thread = threading.Thread(
            target=self._detection_loop, name="ball-tuner-detection", daemon=True
        )
        self._detection_thread.start()
        self._preview_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (
            self._capture_thread,
            self._detection_thread,
            self._preview_thread,
        ):
            if thread is not None:
                thread.join(timeout=2.0)
        with self._ring_lock:
            if self._ring is not None:
                self._ring.close()
                self._ring = None

    def _attach_ring(self) -> Optional[SharedFrameRing]:
        try:
            ring = SharedFrameRing.open(self._frame_ipc_name)
            expected = (int(self.settings.height), int(self.settings.width))
            actual = (ring.height, ring.width)
            if actual != expected:
                ring.close()
                raise RuntimeError(f"Framegroesse {actual} passt nicht zu {expected}")
            print(
                f"[BALL-TUNER] Shared Memory verbunden: {self._frame_ipc_name} "
                f"({ring.width}x{ring.height})"
            )
            return ring
        except Exception as exc:
            self._set_error(f"Shared-Memory-Verbindung fehlgeschlagen: {exc}")
            return None

    def _sync_remote_settings(self) -> None:
        remote = self._read_remote_status()
        values = remote.get("settings")
        if not isinstance(values, dict):
            return
        try:
            self.settings = CameraSettings(**values)
        except (TypeError, ValueError) as exc:
            self._set_error(f"Kameraeinstellungen des CameraProcess ungueltig: {exc}")

    def _consume_loop(self) -> None:
        last_sequence = 0
        while not self._stop_event.is_set():
            with self._ring_lock:
                ring = self._ring
            if ring is None:
                ring = self._attach_ring()
                if ring is None:
                    self._stop_event.wait(0.5)
                    continue
                last_sequence = ring.latest_sequence()
                with self._ring_lock:
                    self._ring = ring
            try:
                item = ring.read_latest(last_sequence)
            except Exception as exc:
                with self._stats_lock:
                    self._ipc_errors += 1
                self._set_error(f"Shared-Memory-Lesen fehlgeschlagen: {exc}")
                with self._ring_lock:
                    if self._ring is ring:
                        self._ring = None
                try:
                    ring.close()
                except Exception:
                    pass
                self._stop_event.wait(0.1)
                continue
            if item is None:
                self._stop_event.wait(0.002)
                continue
            sequence, _timestamp, image = item
            if sequence > last_sequence + 1:
                with self._stats_lock:
                    self._dropped_frames += sequence - last_sequence - 1
            last_sequence = sequence
            with self._latest_lock:
                self._raw_frame = image
                self._raw_sequence = int(sequence)
                self._raw_timestamp = float(_timestamp)
            if self.is_ball_detection_enabled():
                try:
                    self._detection_queue.put_nowait(
                        (int(sequence), float(_timestamp), image)
                    )
                except Full:
                    with self._stats_lock:
                        self._detection_queue_dropped += 1
            else:
                with self._latest_lock:
                    self._last_ball = None
                    self._latest_detection_mask = None
                    self._latest_detection_sequence = int(sequence)

    def _detection_loop(self) -> None:
        """Run OpenCV independently from the shared-memory reader."""

        while not self._stop_event.is_set():
            try:
                sequence, timestamp, image = self._detection_queue.get(timeout=0.1)
            except Empty:
                continue
            if not self.is_ball_detection_enabled():
                continue
            self._detect_frame(sequence, timestamp, image)

    def _detect_frame(self, sequence: int, timestamp: float, image: np.ndarray) -> None:
        """Try the OpenCV detector for every frame received from the mailbox."""

        started = time.perf_counter()
        frame = Frame()
        frame.frame = image
        frame.timestamp = float(timestamp)
        try:
            with self._detector_lock:
                self._detector.detect(frame)
                mask = self._detector.last_mask
                mask_copy = None if mask is None else mask.copy()
            ball = None if frame.ball is None else list(frame.ball)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._latest_lock:
                self._last_ball = ball
                self._latest_detection_mask = mask_copy
                self._latest_detection_sequence = int(sequence)
            with self._stats_lock:
                self._detection_attempts += 1
                self._detection_hits += int(ball is not None)
                self._detection_time_ms += elapsed_ms
                self._detection_last_ms = elapsed_ms
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._stats_lock:
                self._detection_attempts += 1
                self._detection_errors += 1
                self._detection_time_ms += elapsed_ms
                self._detection_last_ms = elapsed_ms
            self._set_error(f"Ballerkennung fehlgeschlagen: {exc}")

    def _preview_loop(self) -> None:
        interval = 1.0 / self.PREVIEW_FPS_LIMIT
        next_preview = time.monotonic()
        last_sequence = 0
        report_time = time.monotonic()
        report_frames = 0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if next_preview > now and self._stop_event.wait(next_preview - now):
                return
            with self._latest_lock:
                sequence = self._raw_sequence
                image = None if self._raw_frame is None else self._raw_frame.copy()
            if image is not None and sequence != last_sequence:
                if self._publish_frame(image):
                    last_sequence = sequence
                    report_frames += 1
            now = time.monotonic()
            if now - report_time >= 1.0:
                with self._stats_lock:
                    self._preview_fps = report_frames / (now - report_time)
                report_frames = 0
                report_time = now
            next_preview = max(next_preview + interval, now)

    def _publish_frame(self, raw_image: np.ndarray) -> bool:
        try:
            image = to_bgr(raw_image, self.settings.pixel_format)
            if image is None or image.size == 0:
                return False
            detection_enabled = self.is_ball_detection_enabled()
            if detection_enabled:
                with self._latest_lock:
                    ball = self._last_ball
                ball_preview = image.copy()
                if ball is not None:
                    x, y, radius = (int(value) for value in ball)
                    cv2.circle(
                        ball_preview,
                        (x, y),
                        max(2, radius),
                        (0, 255, 255),
                        3,
                    )
            else:
                ball_preview, mask_preview, ball = image, image, None
            ok, encoded = cv2.imencode(
                ".jpg", ball_preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ok:
                return False
            with self._latest_lock:
                self._latest_frame = image
                self._latest_jpeg = encoded.tobytes()
                # The first detection stage exposes the detected position in
                # the overlay and status. Keep the two auxiliary endpoints
                # on the same single encoded color frame so detection does
                # not add two more JPEG encoders to the 30-FPS web path.
                self._latest_ball_jpeg = self._latest_jpeg
                self._latest_mask_jpeg = self._latest_jpeg
                self._frame_number += 1
                self._received_frames += 1
                self._last_ball = ball
            return True
        except Exception as exc:
            self._set_error(f"Preview-Verarbeitung fehlgeschlagen: {exc}")
            return False

    # ------------------------------------------------------------------
    # Settings, detection and status exposed to the existing web UI
    # ------------------------------------------------------------------
    def update_ball_settings(self, values: Dict[str, Any]) -> None:
        valid_names = {item.name for item in fields(BallDetectionSettings)}
        updated = dict(vars(self.ball_settings))
        for name, value in values.items():
            if name not in valid_names:
                continue
            current = updated[name]
            updated[name] = int(value) if isinstance(current, int) else float(value)
        candidate = BallDetectionSettings(**updated)
        for name, maximum in {
            "hsv_low_h": 179, "hsv_high_h": 179,
            "hsv_low_s": 255, "hsv_high_s": 255,
            "hsv_low_v": 255, "hsv_high_v": 255,
        }.items():
            value = getattr(candidate, name)
            if not 0 <= value <= maximum:
                raise ValueError(f"{name} muss zwischen 0 und {maximum} liegen")
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
        with self._detector_lock:
            self._detector = self._build_detector()

    def set_ball_detection_enabled(self, enabled: bool) -> None:
        self._ball_detection_enabled = bool(enabled)
        if not self._ball_detection_enabled:
            with self._latest_lock:
                self._last_ball = None
                self._latest_detection_mask = None

    def is_ball_detection_enabled(self) -> bool:
        return self._ball_detection_enabled

    def latest_jpeg(self):
        with self._latest_lock:
            return self._latest_jpeg

    def latest_ball_jpeg(self):
        with self._latest_lock:
            return self._latest_ball_jpeg

    def latest_mask_jpeg(self):
        with self._latest_lock:
            return self._latest_mask_jpeg

    def latest_frame(self):
        with self._latest_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status(self) -> Dict[str, Any]:
        remote = self._read_remote_status()
        self._sync_remote_settings_from_status(remote)
        with self._latest_lock:
            frame_number = self._frame_number
            received_frames = self._received_frames
            frame_age = None if self._latest_frame is None else 0.0
        with self._stats_lock:
            preview_fps = self._preview_fps
            dropped = self._dropped_frames
            ipc_errors = self._ipc_errors
            local_error = self._last_error
            detection_attempts = self._detection_attempts
            detection_hits = self._detection_hits
            detection_errors = self._detection_errors
            detection_queue_dropped = self._detection_queue_dropped
            detection_time_ms = self._detection_time_ms
            detection_last_ms = self._detection_last_ms
        detection_elapsed = max(0.001, time.monotonic() - self._detection_started)
        result = dict(remote)
        result.update({
            "frame_number": frame_number,
            "received_frames": received_frames,
            "preview_fps": round(preview_fps, 2),
            "measured_fps": round(preview_fps, 2),
            "dropped_frames": max(dropped, int(remote.get("dropped_frames", 0) or 0)),
            "ipc_errors": ipc_errors,
            "last_error": remote.get("last_error") or local_error,
            # Never replace the CameraProcess settings with stale tuner-local
            # values in the website response.
            "settings": dict(remote.get("settings") or vars(self.settings)),
            "ball_detection": dict(vars(self.ball_settings)),
            "ball_detection_enabled": self.is_ball_detection_enabled(),
            "detected_ball": self._last_ball,
            "detection_attempts": detection_attempts,
            "detection_hits": detection_hits,
            "detection_errors": detection_errors,
            "detection_queue_dropped": detection_queue_dropped,
            "detection_fps": round(detection_attempts / detection_elapsed, 2),
            "detection_avg_ms": round(
                detection_time_ms / max(1, detection_attempts), 3
            ),
            "detection_last_ms": round(detection_last_ms, 3),
            "frame_ipc": remote.get("frame_ipc", {
                "name": self._frame_ipc_name,
                "slots": self._frame_ipc_slots,
                "available": self._ring is not None,
            }),
            "source": "CameraProcessSharedMemory",
            "camera_owner": "tracking_kicker.camera_process",
        })
        return result

    def _sync_remote_settings_from_status(self, remote: Dict[str, Any]) -> None:
        values = remote.get("settings")
        if not isinstance(values, dict):
            return
        try:
            self.settings = CameraSettings(**values)
        except (TypeError, ValueError) as exc:
            self._set_error(f"Kameraeinstellungen des CameraProcess ungueltig: {exc}")

    def _set_error(self, message: str) -> None:
        now = time.monotonic()
        with self._stats_lock:
            if message == self._last_error and now - self._last_error_time < 1.0:
                return
            self._last_error = str(message)
            self._last_error_time = now
        if message:
            print(f"[BALL-TUNER-ERR] {message}")
