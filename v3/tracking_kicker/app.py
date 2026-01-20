import cv2

from typing import Any, Dict, List, Optional

cv2.setUseOptimized(True)

from tracking_kicker.video import VideoSource
from tracking_kicker.storage import JsonStore
from tracking_kicker.calibration import CalibrationPipeline
from tracking_kicker.detection import BallDetector
from tracking_kicker.analysis import KickerAnalyzer


# ================================================================
# CONFIG
# ================================================================
VIDEO_PATH = r"C:\Users\kleis\PycharmProjects\TrackingKicker\videos\Video.mp4"
CALIBRATION_FILE = "table_calibration.json"

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
    def __init__(self, video_path: str, calibration_path: str, rod_layout: List[Dict[str, Any]]):
        self.video = VideoSource(video_path)
        self.calibration_path = calibration_path
        self.rod_layout = rod_layout
        self.cfg: Optional[Dict[str, Any]] = None

        print(f"🎥 Video FPS erkannt: {self.video.fps}")

    @staticmethod
    def _get_start_mode() -> bool:
        print("\n=====================================")
        print("STARTMODUS")
        print("1 = Neue Gesamtkalibrierung")
        print("2 = Vorhandene Kalibrierung laden")
        print("=====================================")
        mode = input("Auswahl (1/2): ").strip()
        return mode == "1"

    def _load_or_calibrate(self) -> None:
        do_calibration = self._get_start_mode()

        ret, first_frame = self.video.read()
        if not ret or first_frame is None:
            raise RuntimeError("❌ Kein Frame aus dem Video lesbar.")

        if do_calibration:
            pipeline = CalibrationPipeline(self.rod_layout)
            self.cfg = pipeline.run(first_frame)
            JsonStore.save(self.calibration_path, self.cfg)
            print("✅ Kalibrierung gespeichert:", self.calibration_path)
        else:
            cfg = JsonStore.load(self.calibration_path)
            if cfg is None:
                raise RuntimeError(f"❌ Kalibrierungsdatei nicht gefunden: {self.calibration_path}")
            self.cfg = cfg
            print("📁 Kalibrierung geladen:", self.calibration_path)

        self.video.reset()

    def run(self) -> None:
        self._load_or_calibrate()
        assert self.cfg is not None

        detector = BallDetector(debug=False)  # gleiche Settings wie vorher
        analyzer = KickerAnalyzer(
            fps=self.video.fps,
            goals_cfg=self.cfg["goals"],
            field_cfg=self.cfg.get("field")
        )

        delay_ms = int(round(1000 / self.video.fps))

        while True:
            ret, frame = self.video.read()
            if not ret or frame is None:
                break

            meas = detector.detect(frame)  # (x,y,r) oder None
            ball_mask = detector.mask(frame)  # für Tor-Check

            analyzer.draw_overlay(frame, meas, ball_mask=ball_mask, draw_goal_boxes=True)

            cv2.imshow("Kicker Analyse", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == 27:
                break
            elif key == ord('+'):
                delay_ms = max(1, int(delay_ms * 0.8))
            elif key == ord('-'):
                delay_ms = int(delay_ms * 1.25)

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = KickerApp(VIDEO_PATH, CALIBRATION_FILE, ROD_LAYOUT)
    app.run()
