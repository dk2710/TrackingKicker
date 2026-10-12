"""HTTP endpoints for the tracker dashboard, status and goal media."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from flask import Response, jsonify, request, send_from_directory, url_for

from ..Objects.Frames import Frame
from ..camera.conversion import to_bgr
from ..model.RodConfig import RodConfig
from ..model.Vision import FieldConfig, FieldSize, GoalROI, GoalsConfig
from ..runtime import (
    ANALYSIS_VIDEO_PATH,
    GOAL_VIDEO_FPS,
    GOAL_VIDEO_FRAME_LIMIT,
    config,
    vision,
)
from ..logging_support import latest_log_id, runtime_logs


WEB_PATH = Path(__file__).resolve().parents[1] / "web"
WEB_LIVE_FPS = 30.0
WEB_LIVE_DELAY_SECONDS = 3.0
WEB_LIVE_MAX_WIDTH = 640


def _camera_parameters(camera):
    if camera is None:
        return {}
    names = (
        "source",
        "frame_ipc_name",
        "frame_ipc_slots",
        "width",
        "height",
        "offset_x",
        "offset_y",
        "pixel_format",
        "force_fps",
        "exposure_time",
        "gain",
        "grab_strategy_latest_only",
        "serial_number",
        "max_num_buffer",
    )
    return {name: getattr(camera, name) for name in names}


def _calibration_parameters():
    return {
        "goals": asdict(vision.goals) if vision.goals is not None else None,
        "field": asdict(vision.field) if vision.field is not None else None,
    }


def _distortion_parameters(camera):
    if camera is None:
        return {}
    names = ("k1", "k2", "k3", "p1", "p2", "scale")
    return {name: float(getattr(camera, name)) for name in names}


class ApiMixin:
    """Flask handlers kept separate from camera and analysis workers."""

    def api_dashboard(self):
        response = send_from_directory(str(WEB_PATH), "dashboard.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    def _live_mjpeg(self):
        next_send = None
        playback_timestamp = None
        frame_period = 1.0 / WEB_LIVE_FPS
        while not self.stop_event.is_set():
            delay = (
                next_send - time.monotonic()
                if next_send is not None
                else 0.0
            )
            if delay > 0 and self.stop_event.wait(delay):
                return

            raw_frame = None
            raw_timestamp = None
            with self.replay_lock:
                if self.replay_buffer:
                    latest_timestamp = float(self.replay_buffer[-1][1])
                    oldest_timestamp = float(self.replay_buffer[0][1])
                    delayed_edge = (
                        latest_timestamp - WEB_LIVE_DELAY_SECONDS
                    )

                    if playback_timestamp is None:
                        # The replay buffer is trimmed at the same nominal
                        # duration as the requested delay.  With discrete
                        # frame timestamps, the exact delayed edge can be a
                        # fraction of one frame older than the oldest entry.
                        # Start at that oldest entry once the buffer is full
                        # enough; this keeps the effective delay within one
                        # web frame instead of waiting forever.
                        minimum_span = max(
                            0.0,
                            WEB_LIVE_DELAY_SECONDS - frame_period,
                        )
                        if latest_timestamp - oldest_timestamp >= minimum_span:
                            playback_timestamp = max(
                                oldest_timestamp,
                                delayed_edge,
                            )
                    else:
                        # The rolling buffer may discard the frame at the
                        # current playback cursor between two loop passes.
                        # Clamp the cursor to the new oldest frame so one
                        # discarded frame cannot stall the stream forever.
                        playback_timestamp = max(
                            playback_timestamp,
                            oldest_timestamp,
                        )
                        if playback_timestamp + frame_period <= delayed_edge:
                            playback_timestamp += frame_period

                    if playback_timestamp is not None:
                        for image, timestamp in self.replay_buffer:
                            timestamp = float(timestamp)
                            if timestamp > playback_timestamp:
                                break
                            raw_frame = image
                            raw_timestamp = timestamp

            if raw_frame is None:
                # The delayed stream waits until the replay buffer contains
                # enough history instead of making the camera/analysis loop
                # wait for it.
                if self.stop_event.wait(0.01):
                    return
                continue

            preview = raw_frame
            pixel_format = getattr(config.camera, "pixel_format", "BayerRG8")
            height, width = preview.shape[:2]
            if width > WEB_LIVE_MAX_WIDTH:
                scale = WEB_LIVE_MAX_WIDTH / float(width)
                interpolation = cv2.INTER_AREA
                if (
                    preview.ndim == 2
                    and str(pixel_format).replace("_", "").upper().startswith("BAYER")
                ):
                    # Keep the Bayer pattern intact until demosaicing. Area
                    # averaging here washes out the colour channels.
                    interpolation = cv2.INTER_NEAREST
                preview = cv2.resize(
                    preview,
                    (WEB_LIVE_MAX_WIDTH, max(1, round(height * scale))),
                    interpolation=interpolation,
                )
            image = to_bgr(
                preview,
                pixel_format,
            )
            if image is None:
                continue
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            if not ok:
                continue
            jpg = encoded.tobytes()

            # Keep an absolute schedule. If the OS wakes the thread late, the
            # next frame uses the next schedule slot instead of adding the
            # wake-up delay to every subsequent frame.
            send_started = time.monotonic()
            if next_send is None:
                next_send = send_started + frame_period
            else:
                while next_send <= send_started:
                    next_send += frame_period
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n".encode("ascii")
                + b"Cache-Control: no-cache\r\n\r\n"
                + jpg
                + b"\r\n"
            )

    def api_live_stream(self):
        print(f"[WEB] Livestream verbunden: {request.remote_addr or 'unbekannt'}")
        return Response(
            self._live_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    def _detection_parameters(self):
        names = (
            "hsv_low",
            "hsv_high",
            "area_min",
            "area_max",
            "detection_scale",
            "radius_min_px",
            "radius_max_px",
            "circularity_min",
            "solidity_min",
        )
        values = {name: getattr(self.detector, name) for name in names}
        values["hsv_low"] = list(values["hsv_low"])
        values["hsv_high"] = list(values["hsv_high"])
        return values

    @staticmethod
    def _number(value, name, minimum=None, maximum=None):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} muss eine Zahl sein") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} muss endlich sein")
        if minimum is not None and result < minimum:
            raise ValueError(f"{name} muss mindestens {minimum} sein")
        if maximum is not None and result > maximum:
            raise ValueError(f"{name} darf hoechstens {maximum} sein")
        return result

    def _validated_detection_update(self, payload):
        updates = payload.get("ball_detection", payload)
        if not isinstance(updates, dict):
            raise ValueError("ball_detection muss ein Objekt sein")

        current = self._detection_parameters()
        allowed = set(current)
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(f"Unbekannte Erkennungsparameter: {', '.join(unknown)}")

        for name in ("hsv_low", "hsv_high"):
            if name not in updates:
                continue
            value = updates[name]
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"{name} muss drei Werte enthalten")
            current[name] = [
                int(self._number(item, f"{name}[{index}]", 0, 255))
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
                current[name] = self._number(
                    updates[name], name, minimum, maximum
                )

        if current["area_min"] > current["area_max"]:
            raise ValueError("area_min darf nicht groesser als area_max sein")
        if current["radius_min_px"] > current["radius_max_px"]:
            raise ValueError("radius_min_px darf nicht groesser als radius_max_px sein")
        return current

    def api_parameters(self):
        if request.method == "GET":
            return jsonify({
                "ball_detection": self._detection_parameters(),
                "enabled": bool(self.ball_detection_enabled),
                "camera": _camera_parameters(config.camera),
                "calibration": _calibration_parameters(),
            })

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON-Objekt erwartet"}), 400
        try:
            values = self._validated_detection_update(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        for name, value in values.items():
            value = tuple(value) if name.startswith("hsv_") else value
            setattr(self.detector, name, value)
            setattr(self.snapshot_detector, name, value)
        config.ball_detection = dict(values)

        if bool(payload.get("persist", False)):
            try:
                with config.config_path.open("r", encoding="utf-8") as handle:
                    document = json.load(handle)
                document["ball_detection"] = values
                with config.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(document, handle, indent=2)
                    handle.write("\n")
            except (OSError, ValueError) as exc:
                return jsonify({"error": f"Speichern fehlgeschlagen: {exc}"}), 500

        return jsonify({
            "status": "ok",
            "persisted": bool(payload.get("persist", False)),
            "enabled": bool(self.ball_detection_enabled),
            "ball_detection": self._detection_parameters(),
        })

    def api_status(self):
        with self.replay_lock:
            replay_available = bool(self.replay_buffer)
            replay_timestamp = (
                float(self.replay_buffer[-1][1])
                if replay_available
                else None
            )
            replay_sequence = int(getattr(self, "_frame_sequence", 0))
        live = {
            "available": replay_available,
            "sequence": replay_sequence,
            "frame_timestamp": replay_timestamp,
        }
        with self.detection_state_lock:
            detection = dict(self.last_detection)
        with self.last_analysis_lock:
            latest_goal = dict(self.last_analysis)

        measured_fps = None
        try:
            measured_fps = float(self.video.showFPS())
        except (AttributeError, TypeError, ValueError):
            pass

        source_metrics = {}
        try:
            source_status = getattr(self.video, "status", None)
            if callable(source_status):
                source_metrics = source_status()
        except Exception as exc:
            source_metrics = {"status_error": str(exc)}

        return jsonify({
            "status": "stopping" if self.stop_event.is_set() else "running",
            "source": type(self.video).__name__,
            "camera": _camera_parameters(config.camera),
            "calibration": _calibration_parameters(),
            "distortion": _distortion_parameters(config.camera),
            "shared_memory": {
                "name": getattr(getattr(self.video, "ring", None), "name", None),
                "width": getattr(getattr(self.video, "ring", None), "width", None),
                "height": getattr(getattr(self.video, "ring", None), "height", None),
                "slots": getattr(getattr(self.video, "ring", None), "slots", None),
            },
            "source_fps": measured_fps,
            "source_metrics": source_metrics,
            "processing_fps": float(getattr(self, "processing_fps", 0.0)),
            "live_analysis_enabled": bool(self.live_analysis_enabled),
            "ball_detection_enabled": bool(self.ball_detection_enabled),
            "web_live_fps": WEB_LIVE_FPS,
            "web_live_delay_seconds": WEB_LIVE_DELAY_SECONDS,
            "live": live,
            "detection": detection,
            "last_goal": latest_goal,
            "goal_analysis": (
                latest_goal.get("analysis") if latest_goal else None
            ),
            "urls": {
                "dashboard": url_for("dashboard"),
                "live": url_for("live_stream"),
                "parameters": url_for("parameters"),
            },
        })

    def api_logs(self):
        after = request.args.get("after", default=0, type=int)
        limit = request.args.get("limit", default=200, type=int)
        entries = runtime_logs(after=after or 0, limit=limit or 200)
        return jsonify({
            "logs": entries,
            "latest_id": latest_log_id(),
        })

    def _snapshot_payload(self):
        with self.snapshot_lock:
            generation = self.snapshot_generation
            frames = list(self.snapshot_frames)
        if not frames:
            return None

        timestamps = [float(timestamp) for _, timestamp in frames]
        return {
            "generation": generation,
            "count": len(frames),
            "selected_index": len(frames) - 1,
            "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
            "duration_seconds": max(0.0, timestamps[-1] - timestamps[0]),
            "timestamps": timestamps,
            "frame_url": url_for(
                "snapshot_frame",
                generation=generation,
                index=len(frames) - 1,
                analysis=1,
            ),
        }

    def api_snapshot(self):
        if request.method == "GET":
            snapshot = self._snapshot_payload()
            if snapshot is None:
                return jsonify({"error": "Noch kein Snapshot vorhanden"}), 404
            return jsonify(snapshot)

        with self.replay_lock:
            recorded = list(self.replay_buffer)
        if not recorded:
            return jsonify({"error": "Noch keine Frames im Replay-Puffer"}), 409

        latest_timestamp = float(recorded[-1][1])
        selected = [
            (image, float(timestamp))
            for image, timestamp in recorded
            if latest_timestamp - float(timestamp) <= 1.0
        ]
        if not selected:
            selected = [recorded[-1]]

        with self.snapshot_lock:
            self.snapshot_generation += 1
            self.snapshot_frames = selected
        return jsonify(self._snapshot_payload()), 201

    def _snapshot_frame(self, generation, index):
        with self.snapshot_lock:
            if int(generation) != self.snapshot_generation:
                raise LookupError("Snapshot ist nicht mehr aktuell")
            if index < 0 or index >= len(self.snapshot_frames):
                raise IndexError("Snapshot-Frame ausserhalb des Bereichs")
            image, timestamp = self.snapshot_frames[index]
        return image.copy(), float(timestamp)

    def _draw_snapshot_overlay(self, image, frame):
        height, width = image.shape[:2]
        if vision.field is not None:
            corners = np.asarray(
                [[x * width, y * height] for x, y in vision.field.corners_norm],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            if len(corners) == 4:
                cv2.polylines(image, [corners], True, (0, 255, 0), 3)

        if vision.goals is not None:
            for side, color in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
                goal = getattr(vision.goals, side)
                cv2.rectangle(
                    image,
                    (int(goal.x0 * width), int(goal.y0 * height)),
                    (int(goal.x1 * width), int(goal.y1 * height)),
                    color,
                    3,
                )

        if frame.ball is not None:
            x, y, radius = frame.ball
            cv2.circle(image, (int(x), int(y)), max(2, int(radius)), (0, 255, 255), 3)
            cv2.putText(
                image,
                f"Ball {x},{y} r={radius}",
                (max(10, int(x) - 80), max(25, int(y) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    def api_snapshot_frame(self):
        try:
            generation = request.args.get("generation", type=int)
            index = request.args.get("index", type=int)
            if generation is None or index is None:
                raise ValueError("generation und index sind erforderlich")
            raw_image, timestamp = self._snapshot_frame(generation, index)
        except (ValueError, LookupError, IndexError) as exc:
            return jsonify({"error": str(exc)}), 404

        with self.calibration_lock:
            image = to_bgr(
                raw_image,
                getattr(config.camera, "pixel_format", "BayerRG8"),
            )
            image = self.dist.apply(image)
            frame = Frame()
            frame.frame = image
            frame.timestamp = timestamp
            frame.ball = None
            frame.ball_track_valid = False
            if (
                self.ball_detection_enabled
                and request.args.get("analysis", "1").lower()
                not in {"0", "false", "no"}
            ):
                self.snapshot_detector._reset_track()
                self.snapshot_detector.detect(frame)
            self._draw_snapshot_overlay(image, frame)

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            return jsonify({"error": "Snapshot konnte nicht kodiert werden"}), 500
        response = Response(encoded.tobytes(), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Snapshot-Timestamp"] = str(timestamp)
        return response

    @staticmethod
    def _bounded_float(value, name, minimum=0.0, maximum=1.0):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} muss eine Zahl sein") from exc
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen")
        return result

    def _validate_goal(self, value, name):
        if not isinstance(value, dict):
            raise ValueError(f"{name} muss ein Objekt sein")
        required = ("x0", "y0", "x1", "y1")
        if any(key not in value for key in required):
            raise ValueError(f"{name} benoetigt x0, y0, x1 und y1")
        result = {
            key: self._bounded_float(value[key], f"{name}.{key}")
            for key in required
        }
        if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
            raise ValueError(f"{name} muss eine positive Flaeche haben")
        return result

    def _validate_corners(self, value):
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("field_corners_norm muss vier Punkte enthalten")
        corners = []
        for index, point in enumerate(value):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"field_corners_norm[{index}] muss [x, y] sein")
            corners.append([
                self._bounded_float(point[0], f"field_corners_norm[{index}].x"),
                self._bounded_float(point[1], f"field_corners_norm[{index}].y"),
            ])
        return corners

    def _persist_camera_calibration(self):
        with config.config_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        document.setdefault("camera", {})
        document["camera"].update(_distortion_parameters(config.camera))
        with config.config_path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")

    def api_calibration(self):
        if request.method == "GET":
            return jsonify({
                **_calibration_parameters(),
                "distortion": _distortion_parameters(config.camera),
            })

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON-Objekt erwartet"}), 400

        try:
            with self.calibration_lock:
                current_goals = _calibration_parameters()["goals"]
                current_field = _calibration_parameters()["field"]
                goals = dict(current_goals or {})
                field = dict(current_field or {})

                incoming_goals = payload.get("goals", {})
                if incoming_goals:
                    for side in ("left", "right"):
                        if side in incoming_goals:
                            goals[side] = self._validate_goal(
                                incoming_goals[side], f"goals.{side}"
                            )

                if "field_corners_norm" in payload:
                    field["corners_norm"] = self._validate_corners(
                        payload["field_corners_norm"]
                    )
                if "field_size" in payload:
                    size = payload["field_size"]
                    if not isinstance(size, dict):
                        raise ValueError("field_size muss ein Objekt sein")
                    field["size_cm"] = {
                        "w": self._bounded_float(size.get("w"), "field_size.w", 1.0, 10000.0),
                        "h": self._bounded_float(size.get("h"), "field_size.h", 1.0, 10000.0),
                    }
                if "rod_xs_norm" in payload:
                    rods = payload["rod_xs_norm"]
                    if not isinstance(rods, list) or not rods:
                        raise ValueError("rod_xs_norm muss mindestens eine Position enthalten")
                    field["rods"] = {
                        "xs_norm": [
                            self._bounded_float(value, f"rod_xs_norm[{index}]")
                            for index, value in enumerate(rods)
                        ]
                    }

                if not goals or "left" not in goals or "right" not in goals:
                    raise ValueError("left und right benoetigen Torbereiche")
                if not field or "corners_norm" not in field or "size_cm" not in field:
                    raise ValueError("Das Spielfeld benoetigt Ecken und Groesse")

                vision.goals = GoalsConfig(
                    left=GoalROI(**self._validate_goal(goals["left"], "goals.left")),
                    right=GoalROI(**self._validate_goal(goals["right"], "goals.right")),
                )
                rods = field.get("rods")
                vision.field = FieldConfig(
                    size_cm=FieldSize(**field["size_cm"]),
                    corners_norm=field["corners_norm"],
                    rods=RodConfig(**rods) if rods else None,
                )

                self.analyzer.goals_cfg = vision.goals
                self.analyzer.field_cfg = vision.field
                self.analyzer.H_img2cm = None
                self.detector.table_width_cm = max(
                    float(vision.field.size_cm.w), 1.0
                )
                self.snapshot_detector.table_width_cm = max(
                    float(vision.field.size_cm.w), 1.0
                )

                distortion = payload.get("distortion", {})
                if distortion:
                    allowed = {"k1", "k2", "k3", "p1", "p2", "scale"}
                    unknown = sorted(set(distortion) - allowed)
                    if unknown:
                        raise ValueError(
                            f"Unbekannte Distortionsparameter: {', '.join(unknown)}"
                        )
                    for name, value in distortion.items():
                        minimum = 0.01 if name == "scale" else -10.0
                        maximum = 10.0
                        setattr(
                            config.camera,
                            name,
                            self._bounded_float(value, name, minimum, maximum),
                        )
                    self.dist._last_key = None

                if payload.get("persist", True):
                    vision.save()
                    if distortion:
                        self._persist_camera_calibration()
        except (OSError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({
            "status": "ok",
            "persisted": bool(payload.get("persist", True)),
            **_calibration_parameters(),
            "distortion": _distortion_parameters(config.camera),
        })

    def api_test(self):
        print("TEST")
        return jsonify({"status": "ok"})

    def api_loop(self):
        self.api.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False,
        )

    def api_goal(self):
        """Return the result created by the latest confirmed goal.

        The endpoint never creates a replay by itself.  A goal event freezes
        and analyzes the history first; callers can poll this endpoint while
        the 90 color frames are being encoded.
        """

        requested_generation = request.args.get("generation", type=int)
        with self.goal_video_lock:
            video_path = self.goal_video_path
            preparing = (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            )
            generation = self.goal_video_generation
            error = self.goal_video_error
            result = dict(self.goal_video_result or {})

        if generation == 0 and not preparing and not result:
            return jsonify({
                "status": "no_goal",
                "video_ready": False,
                "message": "Noch kein bestätigtes Tor vorhanden",
            }), 404

        if preparing:
            return jsonify({
                "status": "preparing",
                "video_ready": False,
                "video_generation": generation,
                "retry_after_ms": 100,
                "status_url": url_for("goal", generation=generation),
            }), 202

        if error or video_path is None or not result:
            return jsonify({
                "status": "error",
                "video_ready": False,
                "video_generation": generation,
                "message": error or "Goal-Video ist nicht verfügbar",
            }), 503

        if requested_generation is not None and requested_generation != generation:
            return jsonify({
                "status": "gone",
                "video_ready": False,
                "video_generation": generation,
                "message": "Diese Goal-Generation ist nicht mehr aktuell",
            }), 409

        video_filename = Path(video_path).name
        try:
            video_version = Path(video_path).stat().st_mtime_ns
        except OSError:
            video_version = time.time_ns()

        result.update({
            "status": "ready",
            "video_ready": True,
            "video_generation": generation,
            "ball_speed_kmh": float(result.get("speed_kmh", 0.0)),
            "video_url": url_for(
                "goal_video",
                filename=video_filename,
                v=video_version,
            ),
            "video_filename": video_filename,
            "video_frame_count": int(
                result.get("video_frame_count", GOAL_VIDEO_FRAME_LIMIT)
            ),
            "video_fps": float(result.get("video_fps", GOAL_VIDEO_FPS)),
        })
        return jsonify(result), 200

    def api_goal_video(self, filename):
        response = send_from_directory(
            str(ANALYSIS_VIDEO_PATH),
            Path(filename).name,
            mimetype="video/webm",
            as_attachment=False,
        )
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
