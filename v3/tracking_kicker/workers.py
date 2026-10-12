"""Camera, goal-trigger and recording worker loops.

The live worker deliberately performs only the ball/goal trigger.  Position
and speed analysis is done after a goal from the frozen three-second history;
this keeps the 180 FPS camera path independent from the expensive analysis.
"""

from __future__ import annotations

from datetime import datetime
import threading
import time
import traceback
from queue import Empty

import cv2
import numpy as np

from .camera.conversion import to_bgr
from .runtime import ANALYSIS_VIDEO_PATH, GOAL_VIDEO_DURATION_SECONDS, HEADLESS


class WorkerMixin:
    """Worker implementations operating on the application's shared queues."""

    def camera_loop(self):
        """Read the camera and retain exactly the latest three seconds.

        The history keeps the original camera arrays and timestamps.  The
        processing worker replaces the ``Frame`` object's image while the
        history still owns the original array, so no extra camera-rate copy is
        needed here.
        """

        while not self.stop_event.is_set():
            try:
                ret, objframe = self.video.read()
                if not ret or objframe is None:
                    time.sleep(0.005)
                    continue

                self._frame_sequence += 1
                objframe.frame_id = self._frame_sequence

                with self.replay_lock:
                    self.replay_buffer.append(
                        (objframe.frame, float(objframe.timestamp))
                    )
                    cutoff = float(objframe.timestamp) - GOAL_VIDEO_DURATION_SECONDS
                    while (
                        self.replay_buffer
                        and float(self.replay_buffer[0][1]) < cutoff
                    ):
                        self.replay_buffer.popleft()

                # The processing queue is intentionally latest-frame oriented.
                # The complete history above remains available for a later
                # reverse analysis, so skipped live frames do not need another
                # copy or a second queue.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(objframe)

                if self.record_event.is_set():
                    try:
                        self.recording_queue.put(objframe.frame.copy())
                    except Exception as exc:
                        print(f"[WORKER] Recording-Queue-Fehler: {exc}")

            except Exception as exc:
                print(f"[WORKER] Kamera-Loop-Fehler: {exc}")
                traceback.print_exc()
                time.sleep(0.05)

    def processing_loop(self):
        """Run ball/goal triggering without continuous speed analysis."""

        last_log = time.perf_counter()
        frame_count = 0
        skipped_count = 0
        stage_time = {"convert": 0.0, "distortion": 0.0, "trigger": 0.0}

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)

                # Keep only the newest frame for the trigger.  Older frames
                # remain represented by the raw three-second history.
                while True:
                    try:
                        frame = self.frame_queue.get_nowait()
                        skipped_count += 1
                    except Empty:
                        break

                started = time.perf_counter()
                frame.frame = to_bgr(
                    frame.frame,
                    getattr(self.dist.camera, "pixel_format", "BayerRG8"),
                )
                stage_time["convert"] += time.perf_counter() - started

                # The trigger can use a corrected half-resolution image.  No
                # full-resolution corrected image is needed while live
                # analysis is disabled.
                detection_image = None
                detection_scale = float(
                    getattr(self.detector, "detection_scale", 1.0) or 1.0
                )
                started = time.perf_counter()
                if HEADLESS and 0.0 < detection_scale < 1.0:
                    detection_image = cv2.resize(
                        frame.frame,
                        None,
                        fx=detection_scale,
                        fy=detection_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                    with self.calibration_lock:
                        detection_image = self.dist.apply(detection_image)
                else:
                    with self.calibration_lock:
                        frame.frame = self.dist.apply(frame.frame)
                stage_time["distortion"] += time.perf_counter() - started

                if self.ball_detection_enabled:
                    self.detector.detect(frame, detection_image=detection_image)
                else:
                    frame.ball = None
                    frame.ball_track_valid = False
                    frame.ball_track_reason = "deaktiviert"
                    frame.ball_track_distance_px = None

                with self.detection_state_lock:
                    self.last_detection = {
                        "frame_id": getattr(frame, "frame_id", None),
                        "timestamp": float(getattr(frame, "timestamp", 0.0)),
                        "ball": list(frame.ball) if frame.ball is not None else None,
                        "ball_track_valid": bool(
                            getattr(frame, "ball_track_valid", False)
                        ),
                        "ball_track_reason": getattr(
                            frame, "ball_track_reason", None
                        ),
                        "ball_track_distance_px": getattr(
                            frame, "ball_track_distance_px", None
                        ),
                    }

                started = time.perf_counter()
                if self.live_analysis_enabled:
                    goal = self.analyzer.process_frame(frame)
                else:
                    goal = self.analyzer.process_goal_frame(frame)
                stage_time["trigger"] += time.perf_counter() - started

                # Freeze the history at the first entry into a goal ROI.  The
                # analyzer still confirms the candidate after its existing
                # missing-ball interval, so a short detector glitch is not a
                # published goal and its capture is discarded.
                candidate = self.analyzer.consume_goal_candidate()
                if candidate is not None:
                    self._capture_goal_candidate(candidate)

                cancelled = self.analyzer.consume_cancelled_goal_candidate()
                if cancelled is not None:
                    self._discard_goal_candidate(cancelled)

                if goal and any(goal):
                    self._handle_goal(goal)

                if not HEADLESS:
                    if self.display_queue.full():
                        try:
                            self.display_queue.get_nowait()
                        except Empty:
                            pass
                    # The display remains a plain color stream.  No live
                    # analysis overlay is drawn while the feature is paused.
                    self.display_queue.put(frame.frame)

                frame_count += 1
                now = time.perf_counter()
                if now - last_log >= 1.0:
                    elapsed = now - last_log
                    self.processing_fps = frame_count / elapsed
                    print(
                        f"[CAM] GoalTriggerFPS={self.processing_fps:.1f} | "
                        f"convert={stage_time['convert'] / max(1, frame_count) * 1000:.1f}ms "
                        f"dist={stage_time['distortion'] / max(1, frame_count) * 1000:.1f}ms "
                        f"trigger={stage_time['trigger'] / max(1, frame_count) * 1000:.1f}ms "
                        f"uebersprungen={skipped_count}"
                    )
                    last_log = now
                    frame_count = 0
                    skipped_count = 0
                    for key in stage_time:
                        stage_time[key] = 0.0

            except Empty:
                continue
            except Exception as exc:
                print(f"[WORKER] Processing-Loop-Fehler: {exc}")
                traceback.print_exc()
                time.sleep(0.01)

    def recording_loop(self):
        """Optionally save raw frames when the local recording key is active."""

        raw_frames = []
        raw_times = []
        recording = False
        started_at = None

        try:
            while not self.stop_event.is_set():
                try:
                    want_record = self.record_event.is_set()
                    if want_record and not recording:
                        print("[WORKER] RAW-Recording gestartet")
                        raw_frames.clear()
                        raw_times.clear()
                        started_at = time.perf_counter()
                        recording = True

                    if not want_record and recording:
                        print(f"[WORKER] RAW-Speicherung gestartet: {len(raw_frames)}")
                        self._start_save_thread(raw_frames, raw_times)
                        raw_frames.clear()
                        raw_times.clear()
                        recording = False
                        started_at = None

                    if not recording:
                        time.sleep(0.01)
                        continue

                    raw_frame = self.recording_queue.get(timeout=0.2)
                    raw_frames.append(raw_frame.copy())
                    raw_times.append(time.perf_counter() - started_at)
                except Empty:
                    continue
                except Exception as exc:
                    print(f"[WORKER] Recording-Loop-Fehler: {exc}")
                    time.sleep(0.05)
        finally:
            if recording and raw_frames:
                self._start_save_thread(raw_frames, raw_times)

    def _start_save_thread(self, raw_frames, raw_times):
        frames_copy = raw_frames.copy()
        times_copy = raw_times.copy()
        threading.Thread(
            target=self._save_raw_async,
            args=(frames_copy, times_copy),
            name="raw-save",
            daemon=True,
        ).start()

    def _save_raw_async(self, frames, times):
        if not frames:
            return

        filename = ANALYSIS_VIDEO_PATH / (
            f"session_raw_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        )
        try:
            np.savez(filename, frames=np.stack(frames), times=np.asarray(times))
            print(f"[WORKER] RAW-Speicherung abgeschlossen: {filename}")
        except Exception as exc:
            print(f"[WORKER] RAW-Speicherung fehlgeschlagen: {exc}")
