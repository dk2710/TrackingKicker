import cv2
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
from queue import Queue
import threading
from datetime import datetime
import numpy as np

from collections import deque

from v3.tracking_kicker.config import AppConfig
from SocketJpegStreamer import SocketJpegStreamer
from RawReplaySource import RawReplaySource
from BaslerSource import BaslerSource
from storage import JsonStore
from calibration import CalibrationPipeline
from detection import BallDetector
from analysis import KickerAnalyzer
from distoration import DistortionCorrector

config = AppConfig.load()
replay_buffer = deque()          # enthält (frame, timestamp)
replay_lock = threading.Lock()


frame_queue = Queue(maxsize=2)  # WICHTIG: klein halten
display_queue = Queue(maxsize=1)
recording_queue = Queue()

stop_event = threading.Event()
record_event = threading.Event()

VIDEO_PATH = config.videos_dir / "session_raw_2026-01-29_17-03-58.npz"
lens_file = config.config_path

#Analysevideo
Analyse_Video_Path = (Path(__file__).resolve().parents[1] / "videos").resolve()
Analyse_Video_Path.mkdir(parents=True, exist_ok=True)
import os

HEADLESS = os.environ.get("HEADLESS", "0") == "1"

def gui_named_window(name: str):
    if not HEADLESS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

def gui_imshow(name: str, frame):
    if not HEADLESS:
        cv2.imshow(name, frame)

def gui_wait_key(ms: int = 1) -> int:
    if not HEADLESS:
        return cv2.waitKey(ms) & 0xFF
    return 255  # "no key"

def gui_destroy_window(name: str):
    if not HEADLESS:
        cv2.destroyWindow(name)

def gui_destroy_all():
    if not HEADLESS:
        cv2.destroyAllWindows()

class KickerApp:
    def __init__(self, source):
        self.video = source
        self.cfg: Optional[Dict[str, Any]]

        self.dist = DistortionCorrector(config_path=lens_file)

        print(f"📁 Verzerrung config: {lens_file}")
        print(f"🎥 FPS erkannt: {self.video.fps:.3f}")
        cv2.setUseOptimized(True)

    @staticmethod
    def _get_start_mode() -> bool:
        mode = input("1 = Neue Gesamtkalibrierung | 2 = Laden: ").strip()
        return mode == "1"

    def _tune_distortion_on_single_frame(self, frame):
        if frame is None:
            return None

        if HEADLESS:
            print("⚠️ HEADLESS=1 → Distortion Tuning GUI nicht möglich. Abbruch.")
            return None

        gui_named_window("Distortion Preview")

        while True:
            corrected = self.dist.apply(frame)
            gui_imshow("Distortion Preview", corrected)

            key = gui_wait_key(15)
            if key == 27:  # ESC
                gui_destroy_window("Distortion Preview")
                return None

            if key == ord("s"):
                # self.dist.save()
                gui_destroy_window("Distortion Preview")
                return corrected

    def _load_or_calibrate(self) -> None:
        do_calibration = self._get_start_mode()

        if do_calibration:
            print("✅ Kalibrierung gestartet:")
            ret, first_frame = self.video.read()
            bgr = cv2.cvtColor(first_frame, cv2.COLOR_BayerRG2BGR)


            if (not ret) or (bgr is None):
                raise RuntimeError("❌ Kein Frame aus der Quelle lesbar.")

            tuned_frame = self._tune_distortion_on_single_frame(bgr)
            if tuned_frame is None:
                raise RuntimeError("❌ Verzerrung abgebrochen.")

            pipeline = CalibrationPipeline()
            self.cfg = pipeline.run(tuned_frame)
            JsonStore.save(str(config.config_path), self.cfg)
            print("✅ Kalibrierung gespeichert:", str(config.config_path))



        else:

            cfg = JsonStore.load(str(config.config_path))

            if cfg is None:
                raise RuntimeError(f"❌ Kalibrierungsdatei nicht gefunden: {str(config.config_path)}")

            self.cfg = cfg

            print("📁 Kalibrierung geladen:", str(config.config_path))


        self.video.reset()

    def processing_loop(self, in_queue, display_queue, stop_event):
        detector = BallDetector(debug=False)
        analyzer = KickerAnalyzer(
            goals_cfg=config.vision.goals,
            field_cfg=config.vision.field,
        )
        t0 = time.perf_counter()
        last_log = t0
        n=0

        while not stop_event.is_set():
            try:
                frame = in_queue.get(timeout=0.5)
            except:
                continue

            bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)

            meas = detector.detect(bgr)
            overlay = bgr.copy()


            n += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                host_fps = n / (now - t0)

                print(f"[CAM] AnalysierenFPS={host_fps:.1f} | ")

                analyzer.fps =  host_fps
                t0 = now
                last_log = now
                n = 0

            if(analyzer.draw_overlay(overlay, meas) == True):
                self.play_last_seconds()

            # keep-latest policy
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except:
                    pass

            display_queue.put(overlay)

    def play_last_seconds(self):
        with replay_lock:
            if len(replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für Replay")
                return
            frames = list(replay_buffer)

        if HEADLESS:

            return

        gui_named_window("Replay (letzte 3 Sekunden)")

        target_fps = 120
        target_dt = 1.0 / target_fps

        t_start = frames[0][1]
        t_end = frames[-1][1]

        play_start = time.perf_counter()
        idx = 0

        while True:
            now = time.perf_counter()
            play_time = t_start + (now - play_start)

            if play_time >= t_end:
                break

            while idx + 1 < len(frames) and frames[idx + 1][1] <= play_time:
                idx += 1

            frame, _ = frames[idx]
            bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            bgr = bgr[:, :, [2, 1, 0]]

            gui_imshow("Replay (letzte 3 Sekunden)", bgr)
            if gui_wait_key(int(target_dt * 1000)) == 27:
                break

        gui_destroy_window("Replay (letzte 3 Sekunden)")

    def camera_loop(self, video, rt_queue, rec_queue, stop_event):
        t0 = time.perf_counter()
        last_log = t0
        host_fps = 0
        n = 0
        dropped = 0

        while not stop_event.is_set():
            try:
                ret, frame = video.read()
            except Exception as e:
                print(f"⚠️ Camera read exception: {e}")
                time.sleep(0.01)
                continue

            if not ret or frame is None:
                continue

            n += 1

            now = time.perf_counter()


            if rt_queue.full():
                try:
                    rt_queue.get_nowait()
                    dropped += 1
                except:
                    pass
            rt_queue.put(frame)

            rec_queue.put(frame)

            if now - last_log >= 1.0:
                host_fps = n / (now - t0)
                cam_fps = video.showFPS()

                print(
                    f"[CAM] HostFPS={host_fps:.1f} | "
                    f"CamFPS={cam_fps} | "
                    f"Dropped={dropped}"
                )

                t0 = now
                last_log = now
                n = 0
                dropped = 0

            with replay_lock:


                replay_buffer.append((frame.copy(), now))

                max_len = int(host_fps * config.replay_seconds)
                while len(replay_buffer) > max_len:
                    replay_buffer.popleft()


    def save_last_seconds_mp4(self):
        with replay_lock:
            if len(replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für Video")
                return

            frames = list(replay_buffer)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = Analyse_Video_Path / f"replay_{ts}.mp4"

        h, w = frames[0][0].shape[:2]

        target_fps = 60
        input_fps = self.video.fps
        step = max(1, int(round(input_fps / target_fps)))

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (w, h)
        )

        for i in range(0, len(frames), step):
            frame, _ = frames[i]
            bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            bgr = bgr[:, :, [2, 1, 0]]
            writer.write(bgr)

        writer.release()
        print(f"🎞️ Replay gespeichert (≈60 FPS): {out_path}")

    def recording_loop(self, queue, stop_event, record_event):
        raw_frames = []
        raw_times = []
        recording = False
        t_start = None

        while not stop_event.is_set():

            want_record = record_event.is_set()

            # ▶️ START (Flanke)
            if want_record and not recording:
                print("▶️ RAW Recording gestartet")
                raw_frames.clear()
                raw_times.clear()
                t_start = time.perf_counter()
                recording = True

            # ⏹ STOP (Flanke)
            if not want_record and recording:
                print(f"💾 SAVE START {len(raw_frames)}")

                frames_copy = raw_frames.copy()
                times_copy = raw_times.copy()

                # 🔥 ASYNC SAVE
                threading.Thread(
                    target=self._save_raw_async,
                    args=(frames_copy, times_copy),
                    daemon=True
                ).start()

                raw_frames.clear()
                raw_times.clear()
                recording = False
                t_start = None

            if not recording:
                time.sleep(0.01)
                continue

            # 🟢 Recording läuft → Frames sammeln
            try:
                frame = queue.get(timeout=0.2)
            except:
                continue

            raw_frames.append(frame.copy())
            raw_times.append(time.perf_counter() - t_start)

    def _save_raw_async(self, frames, times):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Analyse_Video_Path / f"session_raw_{ts}.npz"

        print("💾 SAVE THREAD START")

        np.savez(
            filename,
            frames=np.stack(frames),
            times=np.array(times)
        )

        print(f"✅ SAVE DONE: {filename}")

    def run(self) -> None:
        self._load_or_calibrate()
        assert self.cfg is not None

        t_cam = threading.Thread(
            target=self.camera_loop,
            args=(self.video, frame_queue, recording_queue, stop_event),
            daemon=True
        )

        t_proc = threading.Thread(
            target=self.processing_loop,
            args=(frame_queue, display_queue, stop_event),
            daemon=True
        )
        t_rec = threading.Thread(
            target=self.recording_loop,
            args=(recording_queue, stop_event, record_event),
            daemon=True
        )
        streamer = SocketJpegStreamer(host="0.0.0.0", port=9999, jpeg_quality=75, send_fps=60)
        t_stream = threading.Thread(
            target=streamer.run,
            args=(display_queue, stop_event),
            daemon=True
        )

        t_stream.start()
        t_cam.start()
        t_proc.start()
        t_rec.start()

        print("⌨️  Taste 'r' = Aufnahme Start/Stop | ESC = Ende")

        if HEADLESS:
            print("🟢 HEADLESS=1 → keine OpenCV-GUI (kein imshow/waitKey). Stream läuft auf :9999")
            try:
                while not stop_event.is_set():
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
        else:
            while not stop_event.is_set():
                try:
                    frame = display_queue.get(timeout=0.5)
                except:
                    continue

                cv2.imshow("Kicker Analyse", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                elif key == ord('k'):
                    print("KEY → Save Replay (letzte 3 Sekunden)")
                    threading.Thread(
                        target=self.save_last_seconds_mp4,
                        daemon=True
                    ).start()

                if key == ord('r'):
                    if record_event.is_set():
                        record_event.clear()
                        print("KEY → Recording OFF")
                    else:
                        record_event.set()
                        print("KEY → Recording ON")

                elif key == ord('k'):
                    print("KEY → Replay letzte 3 Sekunden")
                    threading.Thread(
                        target=self.play_last_seconds,
                        daemon=True
                    ).start()

                elif key == 27:  # ESC
                    break

        stop_event.set()
        record_event.clear()

        t_cam.join(timeout=1)
        t_proc.join(timeout=1)
        t_rec.join(timeout=1)

        self.video.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":


    if config.use_camera:
        try:
            source = BaslerSource(config.camera)
            print("📷 Quelle: Basler Kamera")
        except Exception as e:   # ✅ fängt auch _genicam.RuntimeException
            print(f"❌ Kamera nicht verfügbar: {e}")
            print("➡️ Fallback auf VideoSource…")
            source = RawReplaySource(str(VIDEO_PATH))
    else:
        source = RawReplaySource(str(VIDEO_PATH))

    app = KickerApp(source)
    app.run()


