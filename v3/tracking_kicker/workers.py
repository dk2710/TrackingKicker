"""Camera, processing, recording and replay worker loops."""

import copy
import time
import threading
import traceback
from collections import deque
from datetime import datetime
from queue import Empty, Full

import numpy as np

from .camera.conversion import to_bgr
from .Objects.Frames import Frame
from .runtime import (
    ANALYSIS_VIDEO_PATH,
    GOAL_VIDEO_DURATION_SECONDS,
    GOAL_VIDEO_FPS,
    RETRO_BUFFER_MAX_FRAMES,
    RETRO_BALL_LOSS_SECONDS,
    RETRO_MAX_REANALYZE_FRAMES,
    RETRO_REARM_MIN_BALL_DETECTIONS,
    RETRO_REARM_WINDOW_SECONDS,
    HEADLESS,
)
from .ui.opencv_gui import (
    destroy_window,
    imshow,
    named_window,
    wait_key,
)


class WorkerMixin:
    """Worker implementations operating on the application's shared queues."""

    def _remember_skipped_frame(self, frame) -> None:
        """Keep a frame removed from the live queue for later re-analysis."""
        if frame is None:
            return
        with self.skipped_frame_lock:
            self.skipped_frame_buffer.append(frame)

    def _take_skipped_frames(self, up_to_frame_id=None):
        """Take skipped frames up to the current live frame in time order."""
        with self.skipped_frame_lock:
            if up_to_frame_id is None:
                selected = list(self.skipped_frame_buffer)
                self.skipped_frame_buffer.clear()
                return selected

            selected = []
            remaining = deque(maxlen=RETRO_BUFFER_MAX_FRAMES)
            for skipped in self.skipped_frame_buffer:
                skipped_id = getattr(skipped, "frame_id", None)
                if skipped_id is None or skipped_id <= up_to_frame_id:
                    selected.append(skipped)
                else:
                    remaining.append(skipped)
            self.skipped_frame_buffer = remaining

        # Keep bounded re-analysis from blocking live analysis for seconds.
        if len(selected) > RETRO_MAX_REANALYZE_FRAMES:
            selected = selected[-RETRO_MAX_REANALYZE_FRAMES:]

        selected.sort(
            key=lambda item: (
                getattr(item, "frame_id", 0),
                getattr(item, "timestamp", 0.0),
            )
        )
        return selected

    def _prepare_and_detect(self, source_frame, *, detector=None, dist=None):
        """Copy, correct and detect one raw frame from the replay buffer."""
        detector = self.detector if detector is None else detector
        dist = self.dist if dist is None else dist
        frame = copy.copy(source_frame)
        raw_image = source_frame.frame
        converted = to_bgr(raw_image, dist.camera.pixel_format)
        frame.frame = converted if converted is not raw_image else raw_image.copy()
        frame.frame = dist.apply(frame.frame)
        detector.detect(frame, image_is_bgr=True)
        return frame

    def _restore_skipped_frames(self, frames) -> None:
        """Return a batch if the bounded retro queue filled meanwhile."""
        if not frames:
            return
        with self.skipped_frame_lock:
            combined = list(self.skipped_frame_buffer)
            combined.extend(frames)
            combined.sort(
                key=lambda item: (
                    getattr(item, "frame_id", 0),
                    getattr(item, "timestamp", 0.0),
                )
            )
            self.skipped_frame_buffer = deque(
                combined[-RETRO_BUFFER_MAX_FRAMES:],
                maxlen=RETRO_BUFFER_MAX_FRAMES,
            )

    def _process_recovered_frames(self, recovered_frames, current_frame):
        """Use recovered positions for speed and register a goal candidate."""
        goal_candidate = None
        for recovered_frame in recovered_frames:
            self.analyzer.observe_recovered_frame(recovered_frame)
            ball = recovered_frame.ball
            if ball is not None:
                recovered_id = getattr(recovered_frame, "frame_id", None)
                if (
                    recovered_id is not None
                    and (
                        self._last_visible_frame_id is None
                        or recovered_id > self._last_visible_frame_id
                    )
                ):
                    self._last_visible_frame_id = recovered_id
            if ball is None or goal_candidate is not None:
                continue

            x, y, radius = ball
            frame_shape = recovered_frame.frame.shape
            if self.analyzer._ball_in_goal_roi(x, y, radius, "left", frame_shape):
                goal_candidate = ("left", float(recovered_frame.timestamp))
            elif self.analyzer._ball_in_goal_roi(
                x, y, radius, "right", frame_shape
            ):
                goal_candidate = ("right", float(recovered_frame.timestamp))

        if goal_candidate is None:
            return False, False

        side, landing_timestamp = goal_candidate
        current_ball = getattr(current_frame, "ball", None)
        current_is_still_in_goal = False
        if current_ball is not None:
            x, y, radius = current_ball
            current_is_still_in_goal = self.analyzer._ball_in_goal_roi(
                x, y, radius, side, current_frame.frame.shape
            )

        if current_ball is None or not current_is_still_in_goal:
            return self.analyzer.confirm_recovered_goal(side, landing_timestamp)
        return False, False

    def _schedule_retro_analysis(self, current_frame_id) -> bool:
        """Move skipped raw frames to the independent retro worker."""
        if self.retro_queue.full():
            return False

        skipped_frames = self._take_skipped_frames(current_frame_id)
        if not skipped_frames:
            return False

        try:
            self.retro_queue.put_nowait(skipped_frames)
            return True
        except Full:
            # Normally impossible because only this analysis thread writes to
            # the queue, but never lose the batch because of a race.
            self._restore_skipped_frames(skipped_frames)
            return False

    def _update_ball_episode(self, frame) -> None:
        """Track whether a new ball episode has armed retro analysis."""
        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))

        if (
            self._retro_triggered_for_episode
            and self._last_ball_seen_timestamp is not None
            and timestamp - self._last_ball_seen_timestamp
            >= RETRO_BALL_LOSS_SECONDS
        ):
            self._waiting_for_new_ball = True

        if frame.ball is None:
            return

        if self._retro_triggered_for_episode and not self._waiting_for_new_ball:
            # The same ball episode is still active. Do not re-arm retro just
            # because the ball remains visible after a goal-triggered batch.
            self._last_ball_seen_timestamp = timestamp
            return

        self._last_ball_seen_timestamp = timestamp
        self._recent_ball_detections.append(timestamp)
        cutoff = timestamp - RETRO_REARM_WINDOW_SECONDS
        while (
            self._recent_ball_detections
            and self._recent_ball_detections[0] < cutoff
        ):
            self._recent_ball_detections.popleft()

        if len(self._recent_ball_detections) >= RETRO_REARM_MIN_BALL_DETECTIONS:
            self._retro_armed = True
            self._retro_triggered_for_episode = False
            self._waiting_for_new_ball = False

    def _retro_loss_elapsed(self, frame) -> bool:
        """Return whether an armed ball episode has been absent for 0.5 s."""
        if not self._retro_armed or self._last_ball_seen_timestamp is None:
            return False

        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))
        return timestamp - self._last_ball_seen_timestamp >= RETRO_BALL_LOSS_SECONDS

    def _start_retro_if_possible(self, current_frame_id, reason: str) -> bool:
        """Start one retro batch and disarm until a new ball episode."""
        if self._retro_triggered_for_episode:
            return False
        if not self._schedule_retro_analysis(current_frame_id):
            return False

        self._retro_armed = False
        self._retro_triggered_for_episode = True
        self._waiting_for_new_ball = False
        self._recent_ball_detections.clear()
        print(f"[RETRO] Einmalige Nachanalyse gestartet ({reason})")
        return True

    def _drain_retro_results(self, current_frame) -> None:
        """Apply retro detections in the stateful live-analysis thread."""
        while True:
            try:
                recovered_frames = self.retro_result_queue.get_nowait()
            except Empty:
                return

            recovered_goal = self._process_recovered_frames(
                recovered_frames, current_frame
            )
            self._handle_goal(recovered_goal)

    def retro_loop(self):
        """Detect skipped frames without blocking live analysis."""
        while not self.stop_event.is_set():
            try:
                skipped_frames = self.retro_queue.get(timeout=0.5)
            except Empty:
                continue

            recovered = []
            for skipped in skipped_frames:
                if self.stop_event.is_set():
                    break
                try:
                    recovered.append(
                        self._prepare_and_detect(
                            skipped,
                            detector=self.retro_detector,
                            dist=self.retro_dist,
                        )
                    )
                except Exception as exc:
                    print(f"[RETRO] Frame konnte nicht analysiert werden: {exc}")

            self._retro_analysis_count += len(recovered)
            if not recovered:
                continue

            # The live worker consumes results quickly. If it is temporarily
            # busy, keep the newest completed batch and avoid blocking this
            # worker or growing memory without limit.
            try:
                if self.retro_result_queue.full():
                    self.retro_result_queue.get_nowait()
                self.retro_result_queue.put_nowait(recovered)
            except (Empty, Full):
                pass

    def capture_loop(self):
        """Capture frames independently from the analysis worker.

        This loop is deliberately kept small: it reads from the source,
        assigns a sequence number and publishes the raw frame. The hand-off
        queue is lossy and non-blocking. When analysis is slower than
        acquisition, the oldest pending analysis frame is discarded so this
        loop can immediately return to pypylon.

        The raw image is never modified here. The analysis worker makes its
        own copy before conversion, so replay and recording can safely share
        the raw frame without another full-resolution copy in this loop.
        """
        while not self.stop_event.is_set():
            try:
                ret, objframe = self.video.read()
                if not ret or objframe is None:
                    self._invalidate_rolling_video()
                    # A disconnected/temporarily unavailable camera must not
                    # turn the worker into a busy loop while recovery runs.
                    time.sleep(0.005)
                    continue

                self._frame_sequence += 1
                objframe.frame_id = self._frame_sequence

                with self.replay_lock:
                    self.replay_buffer.append(
                        (objframe.frame, objframe.timestamp)
                    )
                    while (
                        self.replay_buffer
                        and objframe.timestamp - self.replay_buffer[0][1]
                        > GOAL_VIDEO_DURATION_SECONDS
                    ):
                        self.replay_buffer.popleft()

                while not self.stop_event.is_set():
                    try:
                        self.frame_queue.put_nowait(objframe)
                        break
                    except Full:
                        try:
                            skipped = self.frame_queue.get_nowait()
                        except Empty:
                            continue
                        self._remember_skipped_frame(skipped)

                if self.record_event.is_set():
                    try:
                        # The recording worker copies the raw image outside
                        # this time-critical camera loop.
                        self.recording_queue.put_nowait(objframe)
                    except Full:
                        try:
                            self.recording_queue.get_nowait()
                            self.recording_queue.put_nowait(objframe)
                        except (Empty, Full):
                            pass
                    except Exception as exc:
                        print(f"⚠️ Recording Queue Fehler: {exc}")

            except Exception as exc:
                print(f"❌ Camera-Loop Fehler: {exc}")
                traceback.print_exc()
                time.sleep(0.05)

    # Compatibility for callers that used the old worker name.
    camera_loop = capture_loop

    def analysis_loop(self):
        """Run analysis independently from frame acquisition."""
        self.processing_loop()

    def correction_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.raw_queue.get(timeout=0.5)
                frame.frame = to_bgr(frame.frame, self.dist.camera.pixel_format)
                frame.frame = self.dist.apply(frame.frame)

                if self.processed_queue.full():
                    try:
                        self.processed_queue.get_nowait()
                    except Empty:
                        pass
                self.processed_queue.put(frame)
            except Empty:
                continue
            except Exception as exc:
                print(f"❌ Correction-Loop Fehler: {exc}")
                traceback.print_exc()
                time.sleep(0.05)

    def processing_loop(self):
        last_log = time.perf_counter()
        n = 0
        stage_time = {
            "cvt": 0.0,
            "dist": 0.0,
            "detect": 0.0,
            "overlay": 0.0,
            "analyze": 0.0,
        }
        skipped_processing_frames = 0

        while not self.stop_event.is_set():
            try:
                source_frame = self.frame_queue.get(timeout=0.5)

                while True:
                    try:
                        skipped = self.frame_queue.get_nowait()
                        self._remember_skipped_frame(skipped)
                        skipped_processing_frames += 1
                    except Empty:
                        break

                # Never convert the capture-owned raw frame in place. Replay
                # and recording may still reference the same image. Analysis
                # receives an independent copy instead.
                frame = copy.copy(source_frame)

                stage_start = time.perf_counter()
                raw_image = source_frame.frame
                converted = to_bgr(raw_image, self.dist.camera.pixel_format)
                stage_time["cvt"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                detection_image = None
                detection_scale = getattr(self.detector, "detection_scale", 1.0)
                if HEADLESS and 0.0 < detection_scale < 1.0:
                    # No consumer needs a full corrected image in headless
                    # mode. Keep the full shape on the frame for ROI,
                    # homography, and speed units, while the detector gets a
                    # corrected image at its existing detection scale.
                    frame.frame = converted
                    detection_image = self.analysis_dist.apply_scaled(
                        converted, detection_scale
                    )
                else:
                    # Bayer conversion already allocates a new BGR image. The
                    # remap below also allocates its output, so a second copy
                    # of capture-owned BGR input is unnecessary.
                    frame.frame = self.dist.apply(converted)
                stage_time["dist"] += time.perf_counter() - stage_start

                now = time.perf_counter()
                n += 1

                stage_start = time.perf_counter()
                self.detector.detect(
                    frame,
                    detection_image=detection_image,
                    detection_scale=detection_scale,
                    image_is_bgr=True,
                )
                stage_time["detect"] += time.perf_counter() - stage_start

                current_frame_id = getattr(frame, "frame_id", None)
                self._update_ball_episode(frame)
                loss_elapsed = self._retro_loss_elapsed(frame)

                # Retro detections arrive asynchronously, but all analyzer
                # state is still updated by this live worker only. Keep this
                # after the live gap calculation so late frames cannot alter
                # the gap used for the current frame.
                self._drain_retro_results(frame)

                if frame.ball is not None and current_frame_id is not None:
                    self._last_visible_frame_id = current_frame_id

                stage_start = time.perf_counter()
                goal = self.analyzer.process_frame(frame)
                stage_time["analyze"] += time.perf_counter() - stage_start
                self._handle_goal(goal)

                # A live goal result is an immediate retro trigger. Otherwise
                # an armed ball episode starts exactly once after 0.5 s
                # without a ball. The retro worker itself still processes its
                # complete batch if a goal appears in the middle of it.
                if goal and any(goal):
                    self._start_retro_if_possible(current_frame_id, "Tor erkannt")
                elif loss_elapsed:
                    self._start_retro_if_possible(
                        current_frame_id, "0.5 s ohne Ball"
                    )

                # In headless operation no consumer can display this image.
                # Avoid drawing several overlays and handing a full BGR frame
                # to a queue that is never read.  Analysis still receives the
                # exact same corrected image and ball coordinates above.
                if not HEADLESS:
                    stage_start = time.perf_counter()
                    self.analyzer.draw_overlay(frame)
                    stage_time["overlay"] += time.perf_counter() - stage_start

                    if self.display_queue.full():
                        try:
                            self.display_queue.get_nowait()
                        except Empty:
                            pass
                    self.display_queue.put(frame.frame)

                if now - last_log >= 1.0:
                    dt = now - last_log
                    host_fps = n / dt
                    print(
                        f"[CAM] AnalysierenFPS={host_fps:.1f} | "
                        f"cvt={stage_time['cvt'] / n * 1000:.1f}ms "
                        f"dist={stage_time['dist'] / n * 1000:.1f}ms "
                        f"detect={stage_time['detect'] / n * 1000:.1f}ms "
                        f"overlay={stage_time['overlay'] / n * 1000:.1f}ms "
                        f"analyze={stage_time['analyze'] / n * 1000:.1f}ms "
                        f"uebersprungen={skipped_processing_frames} "
                        f"retro={self._retro_analysis_count}"
                    )
                    last_log = now
                    n = 0
                    skipped_processing_frames = 0
                    for key in stage_time:
                        stage_time[key] = 0.0

            except Empty:
                continue
            except Exception as exc:
                print(f"❌ Processing-Loop Fehler: {exc}")
                traceback.print_exc()
                time.sleep(0.01)

    def recording_loop(self):
        raw_frames = []
        raw_times = []
        recording = False
        t_start = None

        try:
            while not self.stop_event.is_set():
                try:
                    want_record = self.record_event.is_set()
                    if want_record and not recording:
                        print("▶️ RAW Recording gestartet")
                        raw_frames.clear()
                        raw_times.clear()
                        t_start = time.perf_counter()
                        recording = True

                    if not want_record and recording:
                        print(f"💾 SAVE START {len(raw_frames)}")
                        self._start_save_thread(raw_frames, raw_times)
                        raw_frames.clear()
                        raw_times.clear()
                        recording = False
                        t_start = None

                    if not recording:
                        time.sleep(0.01)
                        continue

                    raw_frame = self.recording_queue.get(timeout=0.2)
                    raw_image = getattr(raw_frame, "frame", raw_frame)
                    raw_frames.append(raw_image.copy())
                    raw_times.append(time.perf_counter() - t_start)
                except Empty:
                    continue
                except Exception as exc:
                    print(f"❌ Recording-Loop Fehler: {exc}")
                    time.sleep(0.05)
        finally:
            if recording and raw_frames:
                print(f"💾 SAVE BEIM BEENDEN {len(raw_frames)}")
                self._start_save_thread(raw_frames, raw_times)

    def _start_save_thread(self, raw_frames, raw_times):
        frames_copy = raw_frames.copy()
        times_copy = raw_times.copy()
        threading.Thread(
            target=self._save_raw_async,
            args=(frames_copy, times_copy),
            daemon=True,
        ).start()

    def _save_raw_async(self, frames, times):
        if not frames:
            print("⚠️ Keine Frames zum Speichern vorhanden")
            return

        filename = ANALYSIS_VIDEO_PATH / (
            f"session_raw_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        )
        try:
            print("💾 SAVE THREAD START")
            np.savez(filename, frames=np.stack(frames), times=np.array(times))
            print(f"✅ SAVE DONE: {filename}")
        except Exception as exc:
            print(f"❌ Fehler beim Speichern der Aufnahme: {exc}")

    def play_last_seconds(self):
        with self.replay_lock:
            if len(self.replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für Replay")
                return
            frames = list(self.replay_buffer)

        raw_frames = []
        for image, timestamp in self._select_goal_frames(frames):
            replay_frame = Frame()
            replay_frame.frame = image
            replay_frame.timestamp = timestamp
            raw_frames.append(replay_frame)
        frames = [
            (frame.frame, frame.timestamp)
            for frame in self._prepare_selected_frames(raw_frames)
        ]

        if HEADLESS:
            print("⚠️ HEADLESS aktiv → lokales Replay wird nicht angezeigt")
            return

        named_window("Replay (letzte Sekunden)")
        target_dt = 1.0 / GOAL_VIDEO_FPS
        t_start = frames[0][1]
        t_end = frames[-1][1]
        play_start = time.perf_counter()
        idx = 0

        while not self.stop_event.is_set():
            now = time.perf_counter()
            play_time = t_start + (now - play_start)
            if play_time >= t_end:
                break
            while idx + 1 < len(frames) and frames[idx + 1][1] <= play_time:
                idx += 1
            frame, _ = frames[idx]
            imshow("Replay (letzte Sekunden)", frame)
            if wait_key(int(target_dt * 1000)) == 27:
                break

        destroy_window("Replay (letzte Sekunden)")
