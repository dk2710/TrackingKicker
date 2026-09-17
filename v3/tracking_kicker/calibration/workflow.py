"""Calibration lifecycle for the tracker.

The tracker starts with the saved calibration. Interactive calibration is
provided by the web dashboard while live detection continues to run.
"""

import cv2
import time

from .pipeline import CalibrationPipeline
from ..storage import JsonStore
from ..ui.distortion_tuner import DistortionTunerUI
from ..camera.conversion import to_bgr
from ..runtime import HEADLESS, config, vision


class CalibrationWorkflowMixin:
    def tune_distortion_live(self):
        if HEADLESS:
            print("HEADLESS: kein lokales Distortion UI")
            return

        ret, frame = self.video.read()
        if not ret or frame is None:
            print("Kein Frame verfuegbar")
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
        """Load saved values and enter detection without an interactive prompt."""
        cfg = JsonStore.load(str(config.config_path))
        if cfg is None:
            raise RuntimeError(
                f"Kalibrierungsdatei nicht gefunden: {config.config_path}"
            )
        self.cfg = cfg
        self.test_camera_once()
        print(
            "Kalibrierung geladen; Erkennung startet direkt. "
            "Anpassungen sind parallel ueber die Tracking-Weboberflaeche moeglich."
        )
        self.video.reset()

    def test_camera_once(self, timeout_seconds: float = 3.0) -> bool:
        """Read and convert one frame so camera/IPC setup is verified at startup."""
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        last_error = None
        while time.monotonic() < deadline:
            try:
                ret, frame = self.video.read()
                if ret and frame is not None and getattr(frame, "frame", None) is not None:
                    image = to_bgr(
                        frame.frame,
                        getattr(config.camera, "pixel_format", "BayerRG8"),
                    )
                    if image is None or image.size == 0:
                        raise RuntimeError("Frame konnte nicht in BGR umgewandelt werden")
                    corrected = self.dist.apply(image)
                    if corrected is None or corrected.size == 0:
                        raise RuntimeError("Distortion-Korrektur lieferte kein Bild")
                    print(
                        f"[CAMERA-TEST] OK: ein Frame gelesen und umgewandelt "
                        f"({corrected.shape[1]}x{corrected.shape[0]})"
                    )
                    return True
            except Exception as exc:
                last_error = exc
            time.sleep(0.05)

        message = "[CAMERA-TEST] FEHLER: kein gueltiger Frame"
        if last_error is not None:
            message += f": {last_error}"
        print(message)
        raise RuntimeError(message)

    def calibrate_from_frame(self, frame_bgr):
        """Legacy helper for an explicit local calibration request."""
        result = CalibrationPipeline().run(frame_bgr)
        vision.save()
        return result
