# -*- coding: utf-8 -*-
import cv2
from typing import Any, Dict, Optional
from pathlib import Path
import time
from queue import Queue
import threading
from datetime import datetime
import numpy as np
import asyncio
from collections import deque
import os

from v3.tracking_kicker.config import AppConfig
from .SocketJpegStreamer import SocketJpegStreamer
from .RawReplaySource import RawReplaySource
from .BaslerSource import BaslerSource
from .calibration import CalibrationPipeline
from .detection import BallDetector
from .analysis import KickerAnalyzer
from .distoration import DistortionCorrector
from .model.VisionConfig import VisionConfig
from .storage import JsonStore
from .ui.distortion_tuner import DistortionTunerUI
from .comms.ws_gateway import WSGateway


# -------------------------------------------------
# Globale Konfiguration / Queues / Buffer
# -------------------------------------------------
vision = VisionConfig.load()
config = AppConfig.load()

replay_buffer = deque(maxlen=int(60 * config.replay_seconds))  # (frame, timestamp)
replay_lock = threading.Lock()

frame_queue = Queue(maxsize=2)   # klein halten
display_queue = Queue(maxsize=1) # keep-latest
recording_queue = Queue()

stop_event = threading.Event()
record_event = threading.Event()

VIDEO_PATH = config.videos_dir / "test.npz"

Analyse_Video_Path = (Path(__file__).resolve().parents[1] / "videos").resolve()
Analyse_Video_Path.mkdir(parents=True, exist_ok=True)

HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# -------------------------------------------------
# GUI helpers (no-op bei HEADLESS)
# -------------------------------------------------
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

# -------------------------------------------------
# KickerApp
# -------------------------------------------------
class KickerApp:
    def __init__(self, source):
        self.video = source
        self.cfg: Optional[Dict[str, Any]] = None
        self.dist = DistortionCorrector(config.camera)

        # WebSocket-Gateway (für Angular)
        self.ws = WSGateway(host="0.0.0.0", port=8777, token="SUPERSECRET")
        self.ws_loop: Optional[asyncio.AbstractEventLoop] = None

        print(f"📁 Verzerrung config: {config.camera}")
        cv2.setUseOptimized(True)

        # Replay-Steuerung
        self._replay_last_store = 0.0
        self._replay_interval = 1.0 / 60.0
        self._max_replay_frames = int(60 * config.replay_seconds)

    @staticmethod
    def _get_start_mode() -> bool:
        mode = input("1 = Neue Gesamtkalibrierung | 2 = Laden: ").strip()
        return mode == "1"

    def tune_distortion_live(self):
        if HEADLESS:
            print("❌ HEADLESS → kein Distortion UI")
            return

        ret, frame = self.video.read()
        if not ret or frame is None:
            print("❌ Kein Frame verfügbar")
            return

        ui = DistortionTunerUI(self.dist.camera)
        while True:
            ui.update_camera_if_needed()
            corrected = self.dist.apply(frame.frame)
            cv2.imshow("Distortion Tuning", corrected)
            key = cv2.waitKey(15) & 0xFF
            if key == 27:
                break
            if key == ord("s"):
                ui.save_to_config(config.config_path)
        cv2.destroyWindow("Distortion Tuning")

    def _load_or_calibrate(self) -> None:
        do_calibration = self._get_start_mode()

        if do_calibration:
            # Distortion live tunen
            self.tune_distortion_live()

            ret, frame = self.video.read()
            if not ret or frame is None:
                raise RuntimeError("❌ Kein Frame nach Distortion-Tuning")

            frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
            frame.frame = self.dist.apply(frame.frame)

            pipeline = CalibrationPipeline()
            self.cfg = pipeline.run(frame.frame)
            vision.save()

            print("✅ Kalibrierung gespeichert:", str(vision.DEFAULT_PATH))
        else:
            cfg = JsonStore.load(str(config.config_path))
            if cfg is None:
                raise RuntimeError(f"❌ Kalibrierungsdatei nicht gefunden: {str(config.config_path)}")
            self.cfg = cfg
            print("📁 Kalibrierung geladen:", str(config.config_path))

        self.video.reset()

    # ---------------------------------------------
    # WS: Helper
    # ---------------------------------------------
    def _start_ws_gateway(self):
        """Startet den WebSocket-Gateway in eigenem Event-Loop/Thread."""
        self.ws_loop = asyncio.new_event_loop()

        def _run_ws():
            asyncio.set_event_loop(self.ws_loop)
            self.ws_loop.run_until_complete(self.ws.start())

        t_ws = threading.Thread(target=_run_ws, daemon=True)
        t_ws.start()

    def _publish_json(self, payload: Dict[str, Any]):
        """Thread-sicher JSON an alle WS-Clients senden."""
        if self.ws_loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self.ws.broadcast_json(payload), self.ws_loop)
        # optional: fut.result(timeout=?)

    def _broadcast_frame(self, jpg_bytes: bytes):
        """Thread-sicher Binärframe an alle WS-Clients senden."""
        if self.ws_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.ws.broadcast_frame(jpg_bytes), self.ws_loop)

    def send_replay_over_ws(self, target_fps: int = 60, jpeg_quality: int = 80):
        """Schickt die letzten Sekunden als JPEG-Binärframes über WebSocket an Angular."""
        with replay_lock:
            if len(replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für WS-Replay")
                return
            frames = list(replay_buffer)

        input_fps = getattr(self.video, "fps", 60)
        step = max(1, int(round(float(input_fps) / float(target_fps))))
        sub = frames[::step] if step > 1 else frames

        # 1) Startsignal
        self._publish_json({"type": "replay_start", "fps": target_fps, "frames": len(sub)})

        # 2) Frames (JPEG / Binär) senden
        for (frame, _) in sub:
            # Dein Buffer speichert "frame" roh (meist RGB). Für imencode brauchen wir BGR:
            bgr = frame[:, :, [2, 1, 0]] if frame.ndim == 3 and frame.shape[2] == 3 else frame
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue
            self._broadcast_frame(buf.tobytes())

        # 3) Ende
        self._publish_json({"type": "replay_end"})

    # ---------------------------------------------
    # Loops
    # ---------------------------------------------
    def processing_loop(self, frame_queue, dp_queue, stop_event):
        detector = BallDetector(debug=False)
        analyzer = KickerAnalyzer(
            goals_cfg=vision.goals,
            field_cfg=vision.field,
        )
        t0 = time.perf_counter()
        last_log = t0
        n = 0

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except:
                continue

            bgr = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
            bgr = self.dist.apply(bgr)
            meas = detector.detect(bgr)
            overlay = bgr.copy()

            n += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                dt = now - last_log
                host_fps = n / dt
                print(f"[CAM] AnalysierenFPS={host_fps:.1f} | ")
                analyzer.fps = host_fps
                last_log = now
                n = 0

            # Replay-Buffer füllen (rohe Frames + Zeit)
            if now - self._replay_last_store >= self._replay_interval:
                with replay_lock:
                    replay_buffer.append((frame.frame, frame.timestamp))
                    if len(replay_buffer) > self._max_replay_frames:
                        replay_buffer.popleft()
                self._replay_last_store = now

            # Tor-Event erkannt?
            if analyzer.draw_overlay(overlay, meas) is True:
                # Wenn HEADLESS -> an Angular streamen (WS).
                # Wenn nicht HEADLESS -> lokales Fenster abspielen.
                self.play_last_seconds()

            # display_queue: keep-latest
            if dp_queue.full():
                try:
                    dp_queue.get_nowait()
                except:
                    pass
            dp_queue.put(overlay)

    def play_last_seconds(self):
        """Spielt lokal ab (GUI) oder sendet im HEADLESS-Fall das Replay an Angular (WS)."""
        with replay_lock:
            if len(replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für Replay")
                return
            frames = list(replay_buffer)

        if HEADLESS:
            # Kein lokales Fenster -> WS-Output an Angular
            self.send_replay_over_ws(target_fps=60, jpeg_quality=80)
            return

        # Lokales OpenCV-Playback
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            bgr = frame[:, :, [2, 1, 0]]

            gui_imshow("Replay (letzte 3 Sekunden)", bgr)
            if gui_wait_key(int(target_dt * 1000)) == 27:
                break

        gui_destroy_window("Replay (letzte 3 Sekunden)")

    def camera_loop(self, video, frame_queue, rec_queue, stop_event):
        dropped = 0

        while not stop_event.is_set():
            try:
                ret, objframe = video.read()
            except Exception as e:
                print(f"⚠️ Camera read exception: {e}")
                time.sleep(0.01)
                continue

            if not ret or objframe is None:
                continue

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    dropped += 1
                except:
                    pass

            frame_queue.put(objframe)

            if record_event.is_set():
                rec_queue.put(objframe)

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
        input_fps = getattr(self.video, "fps", 60)
        step = max(1, int(round(input_fps / target_fps)))

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (w, h)
        )

        for i in range(0, len(frames), step):
            frame, _ = frames[i]
            bgr = frame[:, :, [2, 1, 0]]
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

    # ---------------------------------------------
    # Run
    # ---------------------------------------------
    def run(self) -> None:
        # WS-Gateway starten (für Angular)
        self._start_ws_gateway()

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

        # Live-Stream bleibt wie gehabt
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
            print("🟢 HEADLESS=1 → keine OpenCV-GUI (kein imshow/waitKey).")
            print("🟢 Live-Stream: http://<host>:9999/  |  WS für Angular: ws://<host>:8777/?token=SUPERSECRET")
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

                gui_imshow("Kicker Analyse", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('k'):
                    print("KEY → Save Replay (letzte 3 Sekunden)")
                    threading.Thread(
                        target=self.save_last_seconds_mp4,
                        daemon=True
                    ).start()
                elif key == ord('r'):
                    if record_event.is_set():
                        record_event.clear()
                        print("KEY → Recording OFF")
                    else:
                        record_event.set()
                        print("KEY → Recording ON")
                elif key == 27:  # ESC
                    break
                elif key == ord('P'):
                    return

        stop_event.set()
        record_event.clear()

        t_cam.join(timeout=1)
        t_proc.join(timeout=1)
        t_rec.join(timeout=1)

        self.video.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    while True:
        if config.use_camera:
            try:
                source = BaslerSource(config.camera)
                print("📷 Quelle: Basler Kamera")
            except Exception as e:  # fängt auch _genicam.RuntimeException
                print(f"❌ Kamera nicht verfügbar: {e}")
                print("➡️ Fallback auf VideoSource…")
                source = RawReplaySource(str(VIDEO_PATH))
        else:
            source = RawReplaySource(str(VIDEO_PATH))

        app = KickerApp(source)
        app.run()