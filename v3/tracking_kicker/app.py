import cv2
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
from queue import Queue
import threading
from datetime import datetime
import numpy as np

frame_queue = Queue(maxsize=2)  # WICHTIG: klein halten
display_queue = Queue(maxsize=1)
recording_queue = Queue()

stop_event = threading.Event()
record_event = threading.Event()

from tracking_kicker.RawReplaySource import RawReplaySource
from tracking_kicker.BaslerSource import BaslerSource
from tracking_kicker.storage import JsonStore
from tracking_kicker.calibration import CalibrationPipeline
from tracking_kicker.detection import BallDetector
from tracking_kicker.analysis import KickerAnalyzer
from tracking_kicker.distoration import DistortionCorrector

VIDEO_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\v3\videos\session_raw_original.npz"
lens_file = (Path(__file__).resolve().parents[1] / "lens_config.json").resolve()
CALIBRATION_FILE = "table_calibration.json"

#Analysevideo
Analyse_Video_Path = (Path(__file__).resolve().parents[1] / "videos").resolve()
Analyse_Video_Path.mkdir(parents=True, exist_ok=True)

class KickerApp:
    def __init__(self, source, calibration_path: str):
        self.video = source
        self.calibration_path = calibration_path
        self.cfg: Optional[Dict[str, Any]] = None

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

        cv2.namedWindow("Distortion Preview", cv2.WINDOW_NORMAL)

        while True:
            corrected = self.dist.apply(frame)
            cv2.imshow("Distortion Preview", corrected)

            key = cv2.waitKey(15) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow("Distortion Preview")
                return None

            if key == ord("s"):
                self.dist.save()  # speichert lens_config.json
                cv2.destroyWindow("Distortion Preview")
                return corrected

    def _load_or_calibrate(self) -> None:
        do_calibration = self._get_start_mode()

        if do_calibration:
            ret, first_frame = self.video.read()
            if (not ret) or (first_frame is None):
                raise RuntimeError("❌ Kein Frame aus der Quelle lesbar.")

            tuned_frame = self._tune_distortion_on_single_frame(first_frame)
            if tuned_frame is None:
                raise RuntimeError("❌ Verzerrung abgebrochen.")

            pipeline = CalibrationPipeline(self.rod_layout)
            self.cfg = pipeline.run(tuned_frame)
            JsonStore.save(self.calibration_path, self.cfg)
            print("✅ Kalibrierung gespeichert:", self.calibration_path)



        else:

            cfg = JsonStore.load(self.calibration_path)

            if cfg is None:
                raise RuntimeError(f"❌ Kalibrierungsdatei nicht gefunden: {self.calibration_path}")

            self.cfg = cfg

            print("📁 Kalibrierung geladen:", self.calibration_path)


        self.video.reset()

    def processing_loop(self, in_queue, display_queue, stop_event):
        detector = BallDetector(debug=False)
        analyzer = KickerAnalyzer(
            fps=self.video.fps,
            goals_cfg=self.cfg["goals"],
            field_cfg=self.cfg.get("field"),
        )

        while not stop_event.is_set():
            try:
                frame = in_queue.get(timeout=0.5)
            except:
                continue

            bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            bgr = bgr[:, :, [2, 1, 0]]  # Farbfix

            meas = detector.detect(bgr)
            overlay = bgr.copy()
            analyzer.draw_overlay(overlay, meas)

            # keep-latest policy
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except:
                    pass

            display_queue.put(overlay)

    def camera_loop(self, video, rt_queue, rec_queue, stop_event):
        t0 = time.time()
        n = 0
        dropped = 0

        last_log = time.time()

        while not stop_event.is_set():
            ret, frame = video.read()
            if not ret:
                continue

            n += 1

            # Queue-Policy: keep-latest
            if rt_queue.full():
                try:
                    rt_queue.get_nowait()
                    dropped += 1
                except:
                    pass
            rt_queue.put(frame)

            rec_queue.put(frame)
            # --- FPS LOGGING ---
            now = time.time()
            if now - last_log >= 1.0:
                host_fps = n / (now - t0)
                cam_fps = video.showFPS()  # ResultingFrameRate

                print(
                    f"[CAM] HostFPS={host_fps:.1f} | "
                    f"CamFPS={cam_fps} | "
                    f"Dropped={dropped}"
                )

                # reset
                t0 = now
                n = 0
                dropped = 0
                last_log = now

    def recording_loop(self, queue, stop_event, record_event):
        raw_frames = []
        raw_times = []
        recording = False

        while not stop_event.is_set():

            want_record = record_event.is_set()

            # ▶️ START (Flanke)
            if want_record and not recording:
                print("▶️ RAW Recording gestartet")
                raw_frames.clear()
                raw_times.clear()
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

            if not recording:
                time.sleep(0.01)
                continue

            # 🟢 Recording läuft → Frames sammeln
            try:
                frame = queue.get(timeout=0.2)
            except:
                continue

            raw_frames.append(frame.copy())
            raw_times.append(time.perf_counter())

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

        t_cam.start()
        t_proc.start()
        t_rec.start()

        print("⌨️  Taste 'r' = Aufnahme Start/Stop | ESC = Ende")
        while not stop_event.is_set():
            try:
                frame = display_queue.get(timeout=0.5)
            except:
                continue

            cv2.imshow("Kicker Analyse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('r'):
                if record_event.is_set():
                    record_event.clear()
                    print("KEY → Recording OFF")
                else:
                    record_event.set()
                    print("KEY → Recording ON")


            elif key == 27:  # ESC
                break

        stop_event.set()
        record_event.clear()

        t_cam.join(timeout=1)
        t_proc.join(timeout=1)
        t_rec.join(timeout=1)

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    USE_CAMERA = True


    if USE_CAMERA:
        try:
            source = BaslerSource( latest_only=False)
            print("📷 Quelle: Basler Kamera")
        except Exception as e:   # ✅ fängt auch _genicam.RuntimeException
            print(f"❌ Kamera nicht verfügbar: {e}")
            print("➡️ Fallback auf VideoSource…")
            source = RawReplaySource(VIDEO_PATH)
    else:
        source = RawReplaySource(VIDEO_PATH)

    app = KickerApp(source, CALIBRATION_FILE)
    app.run()


