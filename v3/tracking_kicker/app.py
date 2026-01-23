import cv2
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

cv2.setUseOptimized(True)

from tracking_kicker.video import VideoSource
from tracking_kicker.BaslerSource import BaslerSource
from tracking_kicker.storage import JsonStore
from tracking_kicker.calibration import CalibrationPipeline
from tracking_kicker.detection import BallDetector
from tracking_kicker.analysis import KickerAnalyzer
from tracking_kicker.distoration import load_params, DistortionCorrector

CALIBRATION_FILE = "table_calibration.json"   # ✅ richtig!
ROD_LAYOUT: List[Dict[str, Any]] = [
    {"team": "black", "expected": 1},
    {"team": "black", "expected": 2},
    {"team": "white", "expected": 2},
    {"team": "black", "expected": 5},
    {"team": "white", "expected": 5},
    {"team": "black", "expected": 3},
    {"team": "white", "expected": 2},
    {"team": "white", "expected": 1},
]


class KickerApp:
    def __init__(self, source, calibration_path: str, rod_layout: List[Dict[str, Any]]):
        self.video = source
        self.calibration_path = calibration_path
        self.rod_layout = rod_layout
        self.cfg: Optional[Dict[str, Any]] = None

        self.lens_file = (Path(__file__).resolve().parents[1] / "lens_config.json").resolve()
        print("LOADING lens_config FROM:", self.lens_file)

        params = load_params(self.lens_file)
        self.dist = DistortionCorrector(params, config_path=self.lens_file, enable_ui=False)

        self._reload_lens_distortion()  # ✅ erzwingt Load + Print + Cache reset

        print(f"📁 Verzerrung config: {self.lens_file}")
        print(f"🎥 FPS erkannt: {self.video.fps:.3f}")

    @staticmethod
    def _get_start_mode() -> bool:
        mode = input("1 = Neue Gesamtkalibrierung | 2 = Laden: ").strip()
        return mode == "1"

    def _tune_distortion_on_single_frame(self, frame):
        if frame is None:
            return None

        self.dist.enable_ui(True)
        cv2.namedWindow("Distortion Preview", cv2.WINDOW_NORMAL)

        while True:
            corrected = self.dist.apply(frame)
            cv2.imshow("Distortion Preview", corrected)

            key = cv2.waitKey(15) & 0xFF
            if key == 27:  # ESC
                self.dist.enable_ui(False)
                cv2.destroyWindow("Distortion Preview")
                return None

            if key == ord("s"):
                self.dist.save()  # speichert lens_config.json
                self.dist.enable_ui(False)
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

            self._reload_lens_distortion()  # ✅ DAS ist der entscheidende Fix

        self.video.reset()

    import time

    def run(self) -> None:
        self._load_or_calibrate()
        assert self.cfg is not None

        detector = BallDetector(debug=False)
        analyzer = KickerAnalyzer(
            fps=self.video.fps,
            goals_cfg=self.cfg["goals"],
            field_cfg=self.cfg.get("field"),
        )

        # --- Echtzeit-Planung ---
        frame_dt = 1.0 / self.video.fps
        t0 = time.perf_counter()
        frame_idx = 0

        display_fps = 60.0
        skip = max(1, int(round(self.video.fps / display_fps)))  # bei 186 -> 3
        i = 0

        while True:
            ret, frame = self.video.read()
            if (not ret) or (frame is None):
                break

            i += 1

            # Optional: Wenn du wirklich Performance willst:
            # Skip früh, bevor du teures processing machst (nur Anzeige schneller)
            if i % skip != 0:
                continue

            frame = self.dist.apply(frame)

            meas = detector.detect(frame)
            ball_mask = detector.mask(frame)
            analyzer.draw_overlay(frame, meas, ball_mask=ball_mask, draw_goal_boxes=True)

            cv2.imshow("Kicker Analyse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("s"):
                self.dist.save()

        self.video.release()
        cv2.destroyAllWindows()

    def _reload_lens_distortion(self) -> None:
        params = load_params(self.lens_file)
        self.dist.params = params

        # Cache killen (sonst bleibt alte Map aktiv!)
        self.dist._map_x = None
        self.dist._map_y = None
        self.dist._last_key = None

        print("✅ Active lens params:", self.dist.params)


if __name__ == "__main__":

    USE_CAMERA = True

    if USE_CAMERA:
        try:
            source = BaslerSource(fps_hint=220.0, latest_only=True)
            print("📷 Quelle: Basler Kamera")
        except Exception as e:   # ✅ fängt auch _genicam.RuntimeException
            print(f"❌ Kamera nicht verfügbar: {e}")
            print("➡️ Fallback auf VideoSource…")
            VIDEO_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos\test2.mp4"
            source = VideoSource(VIDEO_PATH)
    else:
        VIDEO_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos\test2.mp4"
        source = VideoSource(VIDEO_PATH)

    app = KickerApp(source, CALIBRATION_FILE, ROD_LAYOUT)
    app.run()


