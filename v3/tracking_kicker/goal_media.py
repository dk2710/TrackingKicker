"""Goal events, outbound notifications and replay video generation."""

import copy
import json
import shutil
import ssl
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Dict
from urllib import request as urllib_request

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - fallback is used on minimal installs
    requests = None

from .camera.conversion import to_bgr
from .Objects.Frames import Frame
from .runtime import (
    ANALYSIS_VIDEO_PATH,
    DIGIKICKER_POST_DEFAULT_URL,
    GOAL_PRE_NPZ_DURATION_SECONDS,
    GOAL_VIDEO_DURATION_SECONDS,
    GOAL_VIDEO_FRAME_LIMIT,
    GOAL_VIDEO_FPS,
    GOAL_VIDEO_MAX_WIDTH,
    HEADLESS,
)


class GoalMediaMixin:
    """Goal side effects and media encoding kept out of the worker loops."""

    def _post_goal_result(self, payload: Dict[str, Any]) -> None:
        """POST a completed goal result without blocking vision."""
        if not self.digikicker_post_url:
            print("[DIGIKICKER] POST deaktiviert (keine URL konfiguriert)")
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
                print(
                    f"[DIGIKICKER] Tor-POST erfolgreich: "
                    f"{response.status} {self.digikicker_post_url}"
                )
        except Exception as exc:
            print(f"[DIGIKICKER] Tor-POST fehlgeschlagen: {exc}")

    def _handle_goal(self, goal) -> None:
        if not goal or not any(goal):
            return

        event = dict(self.analyzer.last_goal_event or {})
        side = "left" if goal[0] else "right"
        speed = float(event.get("speed_kmh", self.analyzer.speed_kmh_s))
        event.update({
            "event": "goal",
            "side": event.get("side", side),
            "goal_side": event.get("goal_side", side),
            "goal": event.get("goal", side),
            "position": event.get("position", side),
            "goal_position": event.get("goal_position", side),
            "speed_kmh": speed,
            "ball_speed_kmh": speed,
            "speed": speed,
        })

        event_id = event.get("event_id")
        if event_id is not None and event_id == self._last_posted_goal_event_id:
            return
        if event_id is not None:
            self._last_posted_goal_event_id = event_id

        with self.last_analysis_lock:
            self.last_analysis = event

        self._start_goal_npz_save(event)
        threading.Thread(
            target=self._post_goal_result,
            args=(event.copy(),),
            name="digikicker-goal-post",
            daemon=True,
        ).start()

        print(
            f"[GOAL] {event['side']} | "
            f"Geschwindigkeit={speed:.2f} km/h | "
            f"Score {self.analyzer.score_left}:{self.analyzer.score_right}"
        )
        if not HEADLESS:
            self.play_last_seconds()

    def _start_goal_npz_save(self, event: Dict[str, Any]) -> None:
        """Start saving every raw frame from the second before a goal."""
        try:
            goal_timestamp = float(event["goal_timestamp"])
        except (KeyError, TypeError, ValueError):
            print("[GOAL-NPZ] Keine gueltige Torzeit; Speicherung uebersprungen")
            return

        start_timestamp = goal_timestamp - GOAL_PRE_NPZ_DURATION_SECONDS
        with self.replay_lock:
            selected = [
                (image, float(timestamp))
                for image, timestamp in self.goal_pre_buffer
                if start_timestamp <= float(timestamp) <= goal_timestamp
            ]

        if not selected:
            print(
                "[GOAL-NPZ] Keine Rohframes im Zeitfenster "
                f"{start_timestamp:.6f}..{goal_timestamp:.6f}"
            )
            return

        frames = [image for image, _ in selected]
        timestamps = [timestamp for _, timestamp in selected]
        threading.Thread(
            target=self._save_goal_npz_async,
            args=(frames, timestamps, goal_timestamp, dict(event)),
            name="goal-npz-save",
            daemon=True,
        ).start()
        print(
            f"[GOAL-NPZ] Speicherung gestartet: {len(frames)} Rohframes "
            f"({GOAL_PRE_NPZ_DURATION_SECONDS:.1f}s vor Tor)"
        )

    def _save_goal_npz_async(
        self,
        frames,
        timestamps,
        goal_timestamp: float,
        event: Dict[str, Any],
    ) -> None:
        """Write one complete pre-goal raw-frame window as an NPZ file."""
        if not frames or not timestamps:
            return

        temp_filename = None
        try:
            first_timestamp = float(timestamps[0])
            relative_times = np.asarray(
                [float(timestamp) - first_timestamp for timestamp in timestamps],
                dtype=np.float64,
            )
            goal_time = float(goal_timestamp) - first_timestamp
            filename = ANALYSIS_VIDEO_PATH / (
                "goal_pre_"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.npz"
            )
            temp_filename = filename.with_name(f".{filename.name}.tmp")
            frame_array = np.stack([np.asarray(frame).copy() for frame in frames])

            with temp_filename.open("wb") as npz_file:
                np.savez(
                    npz_file,
                    frames=frame_array,
                    times=relative_times,
                    timestamps=np.asarray(timestamps, dtype=np.float64),
                    goal_time=np.asarray(goal_time, dtype=np.float64),
                    goal_position=np.asarray(event.get("goal_position", "")),
                    speed_kmh=np.asarray(
                        float(event.get("speed_kmh", 0.0)), dtype=np.float64
                    ),
                )
            temp_filename.replace(filename)
            print(
                f"[GOAL-NPZ] SAVE DONE: {filename} "
                f"({len(frames)} Frames, Torzeit={goal_time:.6f}s)"
            )
        except Exception as exc:
            print(f"[GOAL-NPZ] Speichern fehlgeschlagen: {exc}")
        finally:
            if temp_filename is not None:
                try:
                    temp_filename.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    print(
                        f"[GOAL-NPZ] Temporärdatei nicht entfernbar: "
                        f"{cleanup_error}"
                    )

    def _invalidate_rolling_video(self):
        with self.rolling_video_lock:
            self.rolling_video_path = None
            self.rolling_video_ready_time = 0.0

    def _queue_rolling_video_frame(self, image, now):
        if now - self.rolling_last_submit < 1.0 / GOAL_VIDEO_FPS:
            return

        image = image.copy()
        image = to_bgr(image, self.dist.camera.pixel_format)

        width = min(image.shape[1], GOAL_VIDEO_MAX_WIDTH)
        height = int(round(image.shape[0] * width / image.shape[1]))
        height -= height % 2
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        try:
            if self.rolling_video_queue.full():
                self.rolling_video_queue.get_nowait()
            self.rolling_video_queue.put_nowait(image)
            self.rolling_last_submit = now
        except Empty:
            pass

    def _snapshot_replay_frames(self):
        """Return unchanged raw frames from the latest three seconds."""
        with self.replay_lock:
            source_frames = list(self.replay_buffer)

        frames = []
        for image, timestamp in source_frames:
            frame = Frame()
            frame.frame = image
            frame.timestamp = timestamp
            frames.append(frame)
        return frames

    @staticmethod
    def _select_goal_frames(source_frames):
        """Reduce an arbitrary number of raw frames evenly to 60 images."""
        if len(source_frames) <= GOAL_VIDEO_FRAME_LIMIT:
            return list(source_frames)
        selected_indices = np.linspace(
            0, len(source_frames) - 1, GOAL_VIDEO_FRAME_LIMIT, dtype=int
        )
        return [source_frames[i] for i in selected_indices]

    def _prepare_selected_frames(self, source_frames):
        """Apply color conversion and distortion correction after selection."""
        prepared = []
        for source_frame in source_frames:
            frame = copy.copy(source_frame)
            image = source_frame.frame.copy()
            image = to_bgr(image, self.dist.camera.pixel_format)
            frame.frame = self.dist.apply(image)
            prepared.append(frame)
        return prepared

    def _rolling_video_encoder_loop(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("⚠️ FFmpeg nicht gefunden; Ringpuffer ist deaktiviert")
            return

        process = None
        try:
            first_image = self.rolling_video_queue.get(timeout=10.0)
            height, width = first_image.shape[:2]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pattern = ANALYSIS_VIDEO_PATH / f"goal_ring_{stamp}_%03d.webm"
            playlist = ANALYSIS_VIDEO_PATH / f"goal_ring_{stamp}.csv"
            command = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", str(GOAL_VIDEO_FPS), "-i", "-",
                "-an", "-c:v", "libvpx", "-deadline", "realtime",
                "-cpu-used", "8", "-threads", "2",
                "-g", str(GOAL_VIDEO_FPS), "-crf", "20", "-b:v", "0",
                "-pix_fmt", "yuv420p", "-f", "segment",
                "-segment_time", str(GOAL_VIDEO_DURATION_SECONDS),
                "-segment_wrap", "3", "-reset_timestamps", "1",
                "-segment_list", str(playlist), "-segment_list_type", "csv",
                "-segment_list_size", "3", "-segment_format", "webm",
                str(pattern),
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
            print(f"[RING] FFmpeg gestartet: {width}x{height} @ {GOAL_VIDEO_FPS} FPS")

            pending = first_image
            published_name = None
            while not self.stop_event.is_set():
                if process.poll() is not None:
                    error_output = process.stderr.read().decode(errors="replace")
                    raise RuntimeError(
                        f"FFmpeg beendet (code={process.returncode}): {error_output}"
                    )

                process.stdin.write(np.ascontiguousarray(pending).tobytes())
                try:
                    pending = self.rolling_video_queue.get(timeout=0.5)
                except Empty:
                    continue

                if playlist.exists():
                    rows = [line for line in playlist.read_text().splitlines() if line.strip()]
                    if rows:
                        latest_name = rows[-1].split(",", 1)[0]
                        latest_path = Path(latest_name)
                        if not latest_path.is_absolute():
                            latest_path = ANALYSIS_VIDEO_PATH / latest_path
                        if (
                            latest_path.exists()
                            and latest_path.stat().st_size > 0
                            and latest_path.name != published_name
                        ):
                            with self.rolling_video_lock:
                                self.rolling_video_path = str(latest_path)
                                self.rolling_video_list = playlist
                                self.rolling_video_ready_time = time.monotonic()
                            print(f"[RING] Segment bereit: {latest_path.name}")
                            published_name = latest_path.name
        except Exception as exc:
            print(f"⚠️ Rolling-Video-Encoder beendet: {exc}")
        finally:
            if process is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass

    def _start_goal_video_preparation(self, force=False):
        with self.goal_video_lock:
            if (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            ):
                return self.goal_video_generation
            if self.goal_video_path is not None and not force:
                return

            self.goal_video_path = None
            with self.last_analysis_lock:
                latest_goal = dict(self.last_analysis)
            self.goal_video_speed = float(
                latest_goal.get("speed_kmh", self.analyzer.speed_kmh_s)
            )
            self.goal_video_error = None
            self.goal_video_generation += 1
            self.goal_video_result_delivered = False
            self.goal_video_ready.clear()
            self.goal_video_thread = threading.Thread(
                target=self._prepare_goal_video,
                args=(self.goal_video_speed, self.goal_video_generation),
                daemon=True,
            )
            self.goal_video_thread.start()
            return self.goal_video_generation

    def _prepare_goal_video(self, speed, generation):
        try:
            deadline = time.perf_counter() + 5.0
            frames = []
            while time.perf_counter() < deadline:
                current_frames = self._snapshot_replay_frames()
                span = (
                    current_frames[-1].timestamp - current_frames[0].timestamp
                    if len(current_frames) >= 2
                    else 0.0
                )
                if (
                    len(current_frames) >= GOAL_VIDEO_FRAME_LIMIT
                    and span >= GOAL_VIDEO_DURATION_SECONDS
                ):
                    frames = self._select_goal_frames(current_frames)
                    break
                time.sleep(0.02)

            if len(frames) < GOAL_VIDEO_FRAME_LIMIT:
                raise RuntimeError(
                    f"Zu wenige Frames fuer Goal-Video: {len(frames)}/"
                    f"{GOAL_VIDEO_FRAME_LIMIT}"
                )

            frames = self._prepare_selected_frames(frames)
            video_path = self.create_goal_video(frames)
            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_path = video_path
                    self.goal_video_speed = speed
                    self.goal_video_frame_count = len(frames)
        except Exception as exc:
            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_error = str(exc)
            print(f"Goal-Video Vorbereitung fehlgeschlagen: {exc}")
        finally:
            self.goal_video_ready.set()

    def create_goal_video(self, frames):
        if not frames:
            raise ValueError("Keine Frames vorhanden")
        if len(frames) != GOAL_VIDEO_FRAME_LIMIT:
            raise ValueError(
                f"Goal-Video braucht {GOAL_VIDEO_FRAME_LIMIT} Frames, "
                f"erhalten: {len(frames)}"
            )

        ANALYSIS_VIDEO_PATH.mkdir(parents=True, exist_ok=True)
        video_path = ANALYSIS_VIDEO_PATH / "goal_latest.webm"
        first_frame = frames[0].frame
        height, width = first_frame.shape[:2]
        video_fps = float(GOAL_VIDEO_FPS)
        self.goal_video_fps = video_fps

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
            "-pix_fmt", "bgr24", "-video_size", f"{output_width}x{output_height}",
            "-framerate", str(GOAL_VIDEO_FPS), "-i", "-", "-an", "-c:v", "libvpx",
            "-deadline", "realtime", "-cpu-used", "8", "-threads", "2",
            "-crf", "10", "-b:v", "0", "-pix_fmt", "yuv420p", "-f", "webm",
            str(temp_path),
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
                image = to_bgr(frame.frame, self.dist.camera.pixel_format)
                if image.shape[1] != output_width or image.shape[0] != output_height:
                    image = cv2.resize(
                        image,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                process.stdin.write(np.ascontiguousarray(image).tobytes())

            process.stdin.close()
            error_output = process.stderr.read().decode(errors="replace")
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise

        if return_code != 0:
            raise RuntimeError(f"FFmpeg-Videoerzeugung fehlgeschlagen: {error_output}")
        temp_path.replace(video_path)
        print(f"✅ Goal Video erstellt: {video_path}")
        return str(video_path)
