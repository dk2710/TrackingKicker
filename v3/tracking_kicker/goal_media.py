"""Goal events, reverse speed analysis and three-second video generation."""

from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict
from urllib import request as urllib_request

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - fallback is used on minimal installs
    requests = None

from .Objects.Frames import Frame
from .camera.conversion import to_bgr
from .detection import BallDetector
from .goal_replay import (
    calculate_three_point_speed_kmh,
    select_analysis_window,
    select_goal_frames,
)
from .runtime import (
    ANALYSIS_VIDEO_PATH,
    GOAL_ANALYSIS_DURATION_SECONDS,
    GOAL_VIDEO_DURATION_SECONDS,
    GOAL_VIDEO_FRAME_LIMIT,
    GOAL_VIDEO_FPS,
    GOAL_VIDEO_MAX_WIDTH,
    config,
    vision,
)


class GoalMediaMixin:
    """Keep goal side effects out of the camera and trigger workers."""

    def _post_goal_result(self, payload: Dict[str, Any]) -> None:
        """POST a completed goal result without blocking frame processing."""

        if not self.digikicker_post_url:
            print("[DIGIKICKER] Tor-POST deaktiviert")
            return

        headers = {"Content-Type": "application/json", "Origin": "127.0.0.1"}
        try:
            if requests is not None:
                response = requests.post(
                    self.digikicker_post_url,
                    json=payload,
                    headers=headers,
                    timeout=self.digikicker_post_timeout,
                    verify=self.digikicker_post_verify_tls,
                )
                response.raise_for_status()
                print(
                    f"[DIGIKICKER] Tor-POST erfolgreich: "
                    f"{response.status_code} {self.digikicker_post_url}"
                )
                return

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            post_request = urllib_request.Request(
                self.digikicker_post_url,
                data=body,
                headers=headers,
                method="POST",
            )
            ssl_context = None
            if not self.digikicker_post_verify_tls:
                ssl_context = ssl._create_unverified_context()
            with urllib_request.urlopen(
                post_request,
                timeout=self.digikicker_post_timeout,
                context=ssl_context,
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
                print(f"[DIGIKICKER] Tor-POST erfolgreich: {response.status}")
        except Exception as exc:
            print(f"[DIGIKICKER] Tor-POST fehlgeschlagen: {exc}")

    def _capture_goal_candidate(self, candidate: Dict[str, Any]) -> None:
        """Freeze the current three-second history for a goal candidate."""

        goal_timestamp = float(candidate["goal_timestamp"])
        start_timestamp = goal_timestamp - GOAL_VIDEO_DURATION_SECONDS
        with self.replay_lock:
            frames = [
                (image, float(timestamp))
                for image, timestamp in self.replay_buffer
                if start_timestamp <= float(timestamp) <= goal_timestamp
            ]

        with self.goal_capture_lock:
            self.pending_goal_capture = {
                "candidate_id": candidate.get("candidate_id"),
                "side": candidate.get("side"),
                "goal_timestamp": goal_timestamp,
                "frames": frames,
            }
        print(
            f"[GOAL] Kandidat eingefroren: {len(frames)} Rohframes, "
            f"Zeitpunkt={goal_timestamp:.6f}"
        )

    def _discard_goal_candidate(self, candidate: Dict[str, Any]) -> None:
        """Discard a candidate that left the goal ROI before confirmation."""

        candidate_id = candidate.get("candidate_id")
        with self.goal_capture_lock:
            if (
                self.pending_goal_capture is not None
                and self.pending_goal_capture.get("candidate_id") == candidate_id
            ):
                self.pending_goal_capture = None
        print(f"[GOAL] Kandidat verworfen: {candidate_id}")

    def _take_goal_capture(self, event: Dict[str, Any]):
        """Take the frozen candidate capture or use the current history."""

        candidate_id = event.get("candidate_id")
        with self.goal_capture_lock:
            capture = self.pending_goal_capture
            if capture is not None and capture.get("candidate_id") == candidate_id:
                self.pending_goal_capture = None
                return capture
            if candidate_id is not None:
                return None

        goal_timestamp = float(event["goal_timestamp"])
        start_timestamp = goal_timestamp - GOAL_VIDEO_DURATION_SECONDS
        with self.replay_lock:
            frames = [
                (image, float(timestamp))
                for image, timestamp in self.replay_buffer
                if start_timestamp <= float(timestamp) <= goal_timestamp
            ]
        return {
            "candidate_id": candidate_id,
            "side": event.get("side"),
            "goal_timestamp": goal_timestamp,
            "frames": frames,
        }

    def _handle_goal(self, goal) -> None:
        """Start asynchronous retro analysis after a confirmed goal."""

        if not goal or not any(goal):
            return

        event = dict(self.analyzer.last_goal_event or {})
        side = "left" if goal[0] else "right"
        event.update({
            "event": "goal",
            "side": event.get("side", side),
            "goal_side": event.get("goal_side", side),
            "goal": event.get("goal", side),
            "position": event.get("position", side),
            "goal_position": event.get("goal_position", side),
            "speed_kmh": 0.0,
            "ball_speed_kmh": 0.0,
            "speed": 0.0,
            "status": "preparing",
        })

        event_id = event.get("event_id")
        if event_id is not None and event_id == self._last_posted_goal_event_id:
            return

        with self.last_analysis_lock:
            self.last_analysis = event

        capture = self._take_goal_capture(event)
        self._start_goal_video_preparation(event, capture)
        print(
            f"[GOAL] {event['side']} erkannt; "
            "Rückwärtsanalyse und Video werden vorbereitet"
        )

    def _start_goal_video_preparation(self, event, capture):
        """Start one preparation worker for the confirmed goal."""

        with self.goal_video_lock:
            if (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            ):
                return self.goal_video_generation

            self.goal_video_path = None
            self.goal_video_error = None
            self.goal_video_result = None
            self.goal_video_generation += 1
            generation = self.goal_video_generation
            self.goal_video_thread = threading.Thread(
                target=self._prepare_goal_media,
                args=(dict(event), capture, generation),
                name="goal-media",
                daemon=True,
            )
            self.goal_video_thread.start()
            return generation

    def _prepare_goal_media(self, event, capture, generation):
        """Analyze three reverse detections and encode the 30-FPS replay."""

        try:
            if capture is None:
                raise RuntimeError("Kein Drei-Sekunden-Puffer zum Tor vorhanden")

            source_frames = capture.get("frames", [])
            goal_timestamp = float(event["goal_timestamp"])
            if not source_frames:
                raise RuntimeError("Der Drei-Sekunden-Puffer ist leer")
            timestamps = [float(timestamp) for _, timestamp in source_frames]
            if (
                max(timestamps) - min(timestamps)
                < GOAL_VIDEO_DURATION_SECONDS - 1.0 / GOAL_VIDEO_FPS
            ):
                raise RuntimeError(
                    "Der Drei-Sekunden-Puffer ist noch nicht vollständig"
                )

            analysis = self._retro_analyze_last_second(
                source_frames,
                goal_timestamp,
            )
            speed = float(analysis["speed_kmh"])
            event.update({
                "speed_kmh": speed,
                "ball_speed_kmh": speed,
                "speed": speed,
                "analysis": analysis,
            })

            selected = select_goal_frames(
                source_frames,
                goal_timestamp,
                duration_seconds=GOAL_VIDEO_DURATION_SECONDS,
                output_fps=GOAL_VIDEO_FPS,
            )
            if len(selected) != GOAL_VIDEO_FRAME_LIMIT:
                raise RuntimeError(
                    f"Goal-Video braucht {GOAL_VIDEO_FRAME_LIMIT} Frames, "
                    f"erhalten: {len(selected)}"
                )

            color_frames = self._prepare_color_frames(selected)
            video_path = self.create_goal_video(color_frames)
            event.update({
                "status": "ready",
                "video_filename": Path(video_path).name,
                "video_url": f"/goal/video/{Path(video_path).name}",
                "video_frame_count": len(color_frames),
                "video_fps": float(GOAL_VIDEO_FPS),
            })

            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_path = video_path
                    self.goal_video_speed = speed
                    self.goal_video_frame_count = len(color_frames)
                    self.goal_video_fps = float(GOAL_VIDEO_FPS)
                    self.goal_video_result = dict(event)

            with self.last_analysis_lock:
                self.last_analysis = dict(event)

            # The outbound notification is sent only after speed and video
            # metadata are complete, so consumers never receive speed=0 from
            # the provisional goal event.
            if event.get("event_id") is not None:
                self._last_posted_goal_event_id = event["event_id"]
            threading.Thread(
                target=self._post_goal_result,
                args=(dict(event),),
                name="digikicker-goal-post",
                daemon=True,
            ).start()
            print(
                f"[GOAL] Video bereit: {len(color_frames)} Frames @ "
                f"{GOAL_VIDEO_FPS:.0f} FPS, Geschwindigkeit={speed:.2f} km/h"
            )
        except Exception as exc:
            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_error = str(exc)
            with self.last_analysis_lock:
                failed = dict(event)
                failed["status"] = "error"
                failed["error"] = str(exc)
                self.last_analysis = failed
            print(f"[GOAL] Medienaufbereitung fehlgeschlagen: {exc}")

    def _retro_analyze_last_second(self, source_frames, goal_timestamp):
        """Scan the final second backwards until three balls are found."""

        analysis_frames = select_analysis_window(
            source_frames,
            goal_timestamp,
            duration_seconds=GOAL_ANALYSIS_DURATION_SECONDS,
        )
        points = []
        detector = BallDetector.from_config(config, vision.field)
        frames_examined = 0

        if analysis_frames:
            self.analyzer._ensure_homography(
                to_bgr(
                    np.asarray(analysis_frames[0][0]),
                    getattr(config.camera, "pixel_format", "BayerRG8"),
                ).shape
            )

        # Reverse order is intentional: the most recent position is found
        # first, then older positions are collected until three are available.
        for image, timestamp in reversed(analysis_frames):
            frames_examined += 1
            frame = Frame()
            with self.calibration_lock:
                color = to_bgr(
                    np.asarray(image),
                    getattr(config.camera, "pixel_format", "BayerRG8"),
                )
                frame.frame = self.dist.apply(color)
            frame.timestamp = float(timestamp)

            # The normal detector keeps a forward track.  A reverse scan must
            # make each frame an independent seed instead.
            detector._reset_track()
            detector.detect(frame)
            if frame.ball is None:
                continue

            x, y, radius = frame.ball
            world = self.analyzer._img_to_cm(x, y)
            if world is None:
                continue
            points.append({
                "timestamp": float(timestamp),
                "x_px": float(x),
                "y_px": float(y),
                "radius_px": float(radius),
                "x_cm": float(world[0]),
                "y_cm": float(world[1]),
            })
            if len(points) == 3:
                break

        points.sort(key=lambda point: point["timestamp"])
        return {
            "direction": "backward",
            "window_seconds": float(GOAL_ANALYSIS_DURATION_SECONDS),
            "frames_available": len(analysis_frames),
            "frames_examined": frames_examined,
            "point_count": len(points),
            "points": points,
            "speed_kmh": calculate_three_point_speed_kmh(points),
        }

    def _prepare_color_frames(self, selected_frames):
        """Convert selected Bayer frames to corrected color images."""

        prepared = []
        pixel_format = getattr(config.camera, "pixel_format", "BayerRG8")
        for image, timestamp in selected_frames:
            frame = Frame()
            with self.calibration_lock:
                color = to_bgr(np.asarray(image), pixel_format)
                frame.frame = self.dist.apply(color)
            frame.timestamp = float(timestamp)
            prepared.append(frame)
        return prepared

    def create_goal_video(self, frames):
        """Encode exactly three seconds of color frames at 30 FPS."""

        if len(frames) != GOAL_VIDEO_FRAME_LIMIT:
            raise ValueError(
                f"Goal-Video braucht {GOAL_VIDEO_FRAME_LIMIT} Frames, "
                f"erhalten: {len(frames)}"
            )

        ANALYSIS_VIDEO_PATH.mkdir(parents=True, exist_ok=True)
        video_path = ANALYSIS_VIDEO_PATH / "goal_latest.webm"
        first_frame = frames[0].frame
        height, width = first_frame.shape[:2]
        output_width = min(width, GOAL_VIDEO_MAX_WIDTH)
        output_height = int(round(height * output_width / width))
        output_height -= output_height % 2

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg nicht gefunden. Bitte installieren: sudo apt install ffmpeg"
            )

        temp_path = video_path.with_suffix(".tmp.webm")
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{output_width}x{output_height}",
            "-framerate", str(GOAL_VIDEO_FPS), "-i", "-", "-an",
            "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8",
            "-threads", "2", "-crf", "10", "-b:v", "0",
            "-pix_fmt", "yuv420p", "-f", "webm", str(temp_path),
        ]
        taskset = shutil.which("taskset")
        if taskset:
            command = [taskset, "-c", "4-7"] + command

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame in frames:
                image = frame.frame
                if image.shape[1] != output_width or image.shape[0] != output_height:
                    image = cv2.resize(
                        image,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                process.stdin.write(np.ascontiguousarray(image).tobytes())

            process.stdin.close()
            error_output = process.stderr.read().decode(errors="replace")
            process.stderr.close()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise

        if return_code != 0:
            raise RuntimeError(f"FFmpeg-Videoerzeugung fehlgeschlagen: {error_output}")
        temp_path.replace(video_path)
        return str(video_path)
