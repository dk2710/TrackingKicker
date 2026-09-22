"""v4 high-rate ball detection service.

The v4 runtime owns the Basler directly and performs only ball detection.
There is no shared-memory transport, replay writer, JPEG streamer or goal
video analysis in this process.
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .BaslerSource import BaslerSource
from .config import AppConfig, BallDetectionSettings
from .detection import BallDetector
from .model.VisionConfig import VisionConfig
from .api import create_app


@dataclass(slots=True)
class ConvertedFrame:
    """One BGR frame handed from the conversion worker to detection."""

    frame: np.ndarray
    timestamp: float
    scale: float


class DetectionService:
    """Convert camera frames in one worker and detect them in the main thread."""

    def __init__(self, source: BaslerSource, config: AppConfig):
        self.source = source
        self.config = config
        self.stop_event = threading.Event()
        self._detector_lock = threading.RLock()
        self._detection_scale = self._configured_detection_scale(
            config.ball_detection
        )
        self._detector = self._build_detector(config.ball_detection)
        self._log_lock = threading.Lock()
        self._log_records = deque(maxlen=1000)
        self._next_log_id = 1
        self._state_lock = threading.Lock()
        self._status = "starting"
        self._last_error = ""
        self._frames_received = 0
        self._detection_attempts = 0
        self._detected_frames = 0
        self._failed_reads = 0
        self._last_ball = None
        self._last_frame_timestamp = None
        self._last_detection_timestamp = None
        self._camera_fps = 0.0
        self._detection_fps = 0.0
        self._last_log_report = time.monotonic()
        self._frame_sequence = 0
        self._detection_interval_started = time.monotonic()
        self._detection_interval_frames = 0
        self._processing_enabled = True
        # High-rate capture must not be paused by a periodic stdout write.
        # Errors are always logged; counters remain available through /api/status.
        self._periodic_log_interval = 0.0

        # Keep the conversion stage bounded as well. The capture thread owns
        # the raw-frame queue in BaslerSource; this queue owns the converted
        # BGR handoff to the main detection thread.
        self._converted_condition = threading.Condition()
        self._converted_frames: deque[ConvertedFrame] = deque()
        self._converted_queue_size = 1
        self._converted_dropped = 0
        self._conversion_thread = threading.Thread(
            target=self._conversion_loop,
            name="v4-bgr-conversion",
            daemon=True,
        )
        self._conversion_thread.start()

    @staticmethod
    def _defaults(values: dict[str, Any]) -> dict[str, Any]:
        defaults = BallDetectionSettings()
        result = defaults.as_tracker_dict()
        result.update(values or {})
        return result

    def _build_detector(self, values: dict[str, Any]) -> BallDetector:
        values = self._defaults(values)
        scale = self._configured_detection_scale(values)
        area_scale = scale * scale
        return BallDetector(
            hsv_low=tuple(int(item) for item in values["hsv_low"]),
            hsv_high=tuple(int(item) for item in values["hsv_high"]),
            # Detection receives the reduced BGR image produced by the
            # conversion thread, while the API keeps the v3 values expressed
            # in full-resolution pixels.
            area_min=float(values["area_min"]) * area_scale,
            area_max=float(values["area_max"]) * area_scale,
            # The high-rate v4 path uses the already reduced Bayer colour
            # image. Avoiding blur/morphology keeps the pylon receiver's USB
            # handoff below the frame period; contour geometry still filters
            # the colour mask.
            use_morph=False,
            use_blur=False,
            fast_path=True,
            debug=False,
        )

    @staticmethod
    def _configured_detection_scale(values: dict[str, Any]) -> float:
        scale = float((values or {}).get("detection_scale", 0.5) or 1.0)
        return min(1.0, max(0.05, scale))

    def _log(self, level: str, message: str) -> None:
        now = time.time()
        with self._log_lock:
            record = {
                "id": self._next_log_id,
                "text": str(message),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
                "level": str(level).upper(),
            }
            self._next_log_id += 1
            self._log_records.append(record)
        print(f"[{record['level']}] {record['text']}", flush=True)

    def _set_runtime_error(self, message: str) -> None:
        with self._state_lock:
            self._status = "error"
            self._last_error = str(message)
        self._log("ERROR", str(message))

    def logs(self, *, after: int = 0, limit: int = 200):
        limit = max(1, min(int(limit), 500))
        with self._log_lock:
            entries = [
                item.copy()
                for item in self._log_records
                if int(item["id"]) > int(after)
            ][-limit:]
            latest = self._next_log_id - 1
        return entries, latest

    def detection_parameters(self) -> dict[str, Any]:
        return self._defaults(self.config.ball_detection)

    def validate_detection_update(self, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.detection_parameters()
        unknown = sorted(set(updates) - set(current))
        if unknown:
            raise ValueError(f"Unbekannte Erkennungsparameter: {', '.join(unknown)}")

        for name in ("hsv_low", "hsv_high"):
            if name in updates:
                value = updates[name]
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError(f"{name} muss drei Werte enthalten")
                current[name] = [
                    int(_number(item, f"{name}[{index}]", 0, 255))
                    for index, item in enumerate(value)
                ]

        ranges = {
            "area_min": (0.0, None),
            "area_max": (0.0, None),
            "detection_scale": (0.05, 1.0),
            "radius_min_px": (0.0, None),
            "radius_max_px": (0.0, None),
            "circularity_min": (0.0, 1.0),
            "solidity_min": (0.0, 1.0),
        }
        for name, (minimum, maximum) in ranges.items():
            if name in updates:
                current[name] = _number(updates[name], name, minimum, maximum)
        if current["area_min"] > current["area_max"]:
            raise ValueError("area_min darf nicht groesser als area_max sein")
        if current["radius_min_px"] > current["radius_max_px"]:
            raise ValueError("radius_min_px darf nicht groesser als radius_max_px sein")
        return current

    def update_detection(self, values: dict[str, Any]) -> None:
        with self._detector_lock:
            self.config.ball_detection = dict(values)
            self._detection_scale = self._configured_detection_scale(values)
            self._detector = self._build_detector(values)

    def status(self, *, vision: VisionConfig | None = None) -> dict[str, Any]:
        source_stats = self.source.stats()
        with self._converted_condition:
            converted_queued = len(self._converted_frames)
            converted_dropped = self._converted_dropped
        conversion_alive = bool(
            self._conversion_thread is not None
            and self._conversion_thread.is_alive()
        )
        with self._state_lock:
            result = {
                "status": self._status,
                "source": "BaslerSource",
                "camera": asdict(self.config.camera) if self.config.camera else {},
                "camera_model": (
                    str(self.source.camera.GetDeviceInfo().GetModelName())
                    if self.source.camera is not None else None
                ),
                "pixel_format": getattr(self.config.camera, "pixel_format", "BayerRG8"),
                # camera_fps is the rate measured in the pylon receiver;
                # detection_fps is the rate at which OpenCV completed.
                "camera_fps": float(source_stats["capture_fps"]),
                "source_fps": float(source_stats["capture_fps"]),
                "detection_fps": float(self._detection_fps),
                "processing_fps": float(self._detection_fps),
                "detection_processing_enabled": bool(self._processing_enabled),
                "frames_received": int(self._frames_received),
                "detection_attempts": int(self._detection_attempts),
                "detected_frames": int(self._detected_frames),
                "failed_reads": int(self._failed_reads),
                "last_error": self._last_error or source_stats["last_error"],
                "camera_frames_total": int(source_stats["camera_frames_total"]),
                "failed_grabs": int(source_stats["failed_grabs"]),
                "payload_errors": int(source_stats["payload_errors"]),
                "handoff_dropped": int(source_stats["handoff_dropped"]),
                "handoff_queued": int(source_stats["handoff_queued"]),
                "bgr_conversion_thread_alive": conversion_alive,
                "bgr_conversion_dropped": int(converted_dropped),
                "bgr_conversion_queued": int(converted_queued),
                "last_block_id": source_stats["last_block_id"],
                "resulting_fps": float(source_stats["resulting_fps"]),
                "capture_thread_alive": bool(source_stats["capture_thread_alive"]),
                "last_detection": {
                    "ball": self._last_ball,
                    "timestamp": self._last_detection_timestamp,
                },
                "last_frame_timestamp": self._last_frame_timestamp,
                "ball_detection_enabled": bool(self.config.ball_detection_enabled),
                "video_enabled": False,
                "shared_memory": None,
                "web_live_fps": 0,
                "web_live_delay_seconds": 0,
            }
        if vision is not None:
            result["calibration"] = {
                "goals": asdict(vision.goals) if vision.goals else None,
                "field": asdict(vision.field) if vision.field else None,
            }
        return result

    @staticmethod
    def _to_bgr(image: np.ndarray, pixel_format: str) -> np.ndarray:
        if image.ndim == 2:
            fmt = str(pixel_format or "BayerRG8").replace("_", "").upper()
            if fmt.startswith("BAYERBG"):
                return cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
            if fmt.startswith("BAYERGB"):
                return cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)
            if fmt.startswith("BAYERGR"):
                return cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
            if fmt.startswith("MONO"):
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
        return image

    @staticmethod
    def _to_bgr_detection_image(
        image: np.ndarray,
        pixel_format: str,
        scale: float,
    ) -> np.ndarray:
        """Create the reduced colour image used by the detector.

        BayerRG8 is the camera's one byte per pixel format. At the detector's
        default 0.5 scale, converting the complete 1100x650 image first is
        unnecessary memory traffic. The four pixels of each Bayer 2x2 block
        already contain one red, two green and one blue sample, so they can be
        assembled directly into a reduced BGR image.
        """
        if image.ndim != 2 or abs(float(scale) - 0.5) > 1e-6:
            bgr = DetectionService._to_bgr(image, pixel_format)
            if 0.05 <= scale < 1.0:
                return cv2.resize(
                    bgr,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            return bgr

        fmt = str(pixel_format or "BayerRG8").replace("_", "").upper()
        if not fmt.startswith("BAYER"):
            gray = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            return DetectionService._to_bgr(gray, pixel_format)

        height = (int(image.shape[0]) // 2) * 2
        width = (int(image.shape[1]) // 2) * 2
        raw = image[:height, :width]
        top_left = raw[0::2, 0::2]
        top_right = raw[0::2, 1::2]
        bottom_left = raw[1::2, 0::2]
        bottom_right = raw[1::2, 1::2]
        green = (
            top_right.astype(np.uint16) + bottom_left.astype(np.uint16)
        ) // 2
        green = green.astype(np.uint8)

        if fmt.startswith("BAYERBG"):
            blue, red = top_left, bottom_right
        elif fmt.startswith("BAYERGR"):
            red, blue = top_right, bottom_left
        elif fmt.startswith("BAYERGB"):
            blue, red = top_right, bottom_left
        else:  # BayerRG and unknown Bayer variants use RG as the safe default.
            red, blue = top_left, bottom_right
        return np.stack((blue, green, red), axis=-1)

    def _queue_converted(self, frame: ConvertedFrame) -> None:
        with self._converted_condition:
            if len(self._converted_frames) >= self._converted_queue_size:
                self._converted_frames.popleft()
                self._converted_dropped += 1
            self._converted_frames.append(frame)
            self._converted_condition.notify()

    def _read_converted(self) -> tuple[bool, ConvertedFrame | None]:
        deadline = time.monotonic() + 1.1
        with self._converted_condition:
            while not self._converted_frames and not self.stop_event.is_set():
                if (
                    self._conversion_thread is not None
                    and not self._conversion_thread.is_alive()
                ):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._converted_condition.wait(timeout=remaining)
            if self._converted_frames:
                return True, self._converted_frames.popleft()
        return False, None

    def _conversion_loop(self) -> None:
        """Read raw frames, convert them to BGR, and queue the newest one."""
        consecutive_read_failures = 0
        try:
            while not self.stop_event.is_set():
                ok, raw_frame = self.source.read()
                if not ok or raw_frame is None:
                    if self.stop_event.is_set():
                        break
                    with self._state_lock:
                        self._failed_reads += 1
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 5:
                        error = self.source.last_error or (
                            "BaslerSource.read() lieferte keinen Rohframe"
                        )
                        self._set_runtime_error(error)
                        self.stop_event.set()
                        break
                    continue

                consecutive_read_failures = 0
                with self._detector_lock:
                    scale = self._detection_scale
                image = self._to_bgr_detection_image(
                    raw_frame.frame,
                    getattr(self.config.camera, "pixel_format", "BayerRG8"),
                    scale,
                )
                self._queue_converted(
                    ConvertedFrame(
                        frame=image,
                        timestamp=raw_frame.timestamp,
                        scale=scale,
                    )
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                self._set_runtime_error(f"BGR-Konvertierung abgebrochen: {exc}")
                self.stop_event.set()
        finally:
            with self._converted_condition:
                self._converted_condition.notify_all()

    def _process_detection_frame(self, converted: ConvertedFrame) -> None:
        """Run ball detection on one converted BGR frame."""
        try:
            self._frame_sequence += 1
            stride = max(
                1,
                int(getattr(self.config, "detection_frame_stride", 1) or 1),
            )
            if stride > 1 and self._frame_sequence % stride:
                return

            with self._detector_lock:
                detector = self._detector
            detect_image = converted.frame
            scale = converted.scale
            if not 0.05 <= scale < 1.0:
                scale = 1.0
            ball = detector.detect(detect_image)
            if ball is not None and scale != 1.0:
                ball = tuple(
                    int(round(float(value) / scale))
                    for value in ball
                )

            now = time.monotonic()
            self._detection_interval_frames += 1
            self._frames_received += 1
            self._detection_attempts += 1
            self._detected_frames += int(ball is not None)
            self._last_ball = list(ball) if ball is not None else None
            self._last_frame_timestamp = float(converted.timestamp)
            self._last_detection_timestamp = float(converted.timestamp)

            elapsed = now - self._detection_interval_started
            if elapsed >= 1.0:
                with self._state_lock:
                    self._camera_fps = self._detection_interval_frames / elapsed
                    self._detection_fps = self._detection_interval_frames / elapsed
                self._detection_interval_started = now
                self._detection_interval_frames = 0

            if (
                self._periodic_log_interval > 0
                and now - self._last_log_report >= self._periodic_log_interval
            ):
                with self._state_lock:
                    fps = self._detection_fps
                    frames = self._frames_received
                    detected = self._detected_frames
                source_stats = self.source.stats()
                self._log(
                    "INFO",
                    "detection 30s: "
                    f"frames={frames}, detected={detected}, fps={fps:.2f}, "
                    f"camera_fps={source_stats['capture_fps']:.2f}, "
                    f"camera_frames={source_stats['camera_frames_total']}, "
                    f"handoff_dropped={source_stats['handoff_dropped']}, "
                    f"payload_errors={source_stats['payload_errors']}",
                )
                self._last_log_report = now
        except Exception as exc:
            with self._state_lock:
                self._status = "error"
                self._last_error = str(exc)
            self._log("ERROR", f"Ballerkennung abgebrochen: {exc}")

    def run(
        self,
        *,
        duration_seconds: float = 0.0,
        process_detection: bool = True,
    ) -> None:
        self._processing_enabled = bool(process_detection)
        self._status = "running"
        if process_detection:
            self._log(
                "INFO", "v4 Ball-Erkennung gestartet; Videoausgabe deaktiviert"
            )
        else:
            self._log(
                "INFO",
                "v4 Capture-only gestartet; BGR-Konvertierung aktiv, "
                "Ballerkennung deaktiviert",
            )
        started = time.monotonic()
        consecutive_read_failures = 0
        try:
            while not self.stop_event.is_set():
                if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                    break
                source_stats = self.source.stats()
                if not source_stats["capture_thread_alive"]:
                    self._set_runtime_error(
                        source_stats["last_error"] or "Basler-Capture-Thread beendet"
                    )
                    break
                if (
                    self._conversion_thread is None
                    or not self._conversion_thread.is_alive()
                ):
                    self._set_runtime_error("BGR-Konvertierungs-Thread beendet")
                    break
                if not process_detection:
                    # Keep the service alive so the capture and conversion
                    # workers can be measured without HSV/detection work in
                    # the main thread. The bounded BGR queue intentionally
                    # retains only the newest converted frame in this mode.
                    time.sleep(0.05)
                    continue
                ok, frame = self._read_converted()
                if not ok or frame is None:
                    with self._state_lock:
                        self._failed_reads += 1
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 5:
                        error = self.source.last_error or (
                            "BaslerSource.read() lieferte keinen Frame"
                        )
                        with self._state_lock:
                            self._status = "error"
                            self._last_error = str(error)
                        self._log("ERROR", self._last_error)
                        break
                    continue
                consecutive_read_failures = 0
                self._process_detection_frame(frame)
        except Exception as exc:
            with self._state_lock:
                self._status = "error"
                self._last_error = str(exc)
            self._log("ERROR", f"Ballerkennung abgebrochen: {exc}")
            raise
        finally:
            if self._status != "error":
                self._status = "stopped"

    def stop(self) -> None:
        self.stop_event.set()
        with self._converted_condition:
            self._converted_condition.notify_all()
        thread = self._conversion_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.5)


def _number(value, name, minimum=None, maximum=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} muss eine Zahl sein") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} muss endlich sein")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} muss mindestens {minimum} sein")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} darf hoechstens {maximum} sein")
    return result


def _reserve_capture_cpu(camera_config) -> None:
    """Leave one CPU for the pylon receiver on Linux.

    The capture thread applies its own affinity after it starts. The main
    process is moved to the remaining CPUs first so OpenCV cannot compete
    with USB reception on the reserved core.
    """
    if camera_config is None:
        return
    if not bool(getattr(camera_config, "reserve_capture_cpu", True)):
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return
    value = getattr(camera_config, "capture_cpu", None)
    if value is None:
        return
    try:
        capture_cpu = int(value)
        available = set(os.sched_getaffinity(0))
        if capture_cpu not in available or len(available) < 2:
            return
        worker_cpus = available - {capture_cpu}
        os.sched_setaffinity(0, worker_cpus)
        print(
            f"[APP] CPU {capture_cpu} für Kameraempfang reserviert; "
            f"OpenCV-CPUs={sorted(worker_cpus)}",
            flush=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[APP-WARN] CPU-Trennung nicht setzbar: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 reine Basler-Ballerkennung")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Nur Kamera-Capture und BGR-Konvertierung testen; Erkennung pausieren",
    )
    args = parser.parse_args()

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
    config = AppConfig.load()
    if config.camera is None:
        raise RuntimeError("Keine Kamerakonfiguration in v4/config.json")
    _reserve_capture_cpu(config.camera)
    vision = VisionConfig.load()
    source = BaslerSource(config.camera)
    service = DetectionService(source, config)
    host = args.host or str(config.server.get("host", "0.0.0.0"))
    port = int(args.port or config.server.get("port", 5000))
    api = create_app(service, config, vision)

    def request_stop(_signum, _frame):
        service.stop()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    api_thread = threading.Thread(
        target=lambda: api.run(
            host=host,
            port=port,
            threaded=True,
            use_reloader=False,
        ),
        name="v4-api",
        daemon=True,
    )
    api_thread.start()
    try:
        service.run(
            duration_seconds=args.seconds,
            process_detection=not args.capture_only,
        )
    finally:
        service.stop()
        source.release()
        final = service.status()
        print(
            "[DETECTION] Final: "
            f"frames={final['frames_received']}, "
            f"attempts={final['detection_attempts']}, "
            f"fps={final['detection_fps']:.2f}, "
            f"detected={final['detected_frames']}, "
            f"failed_reads={final['failed_reads']}, "
            f"payload_errors={final['payload_errors']}",
            flush=True,
        )
        print("v4 sauber beendet", flush=True)


if __name__ == "__main__":
    main()
