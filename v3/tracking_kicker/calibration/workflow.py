"""Startup calibration workflow."""

import cv2

from ..camera.conversion import to_bgr
from .pipeline import CalibrationPipeline
from ..storage import JsonStore
from ..ui.distortion_tuner import DistortionTunerUI
from ..runtime import HEADLESS, config, vision


class CalibrationWorkflowMixin:
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
            self.tune_distortion_live()
            ret, frame = self.video.read()
            if not ret or frame is None:
                raise RuntimeError("❌ Kein Frame nach Distortion-Tuning")

            frame.frame = to_bgr(frame.frame, self.dist.camera.pixel_format)
            frame.frame = self.dist.apply(frame.frame)

            self.cfg = CalibrationPipeline().run(frame.frame)
            vision.save()
            print("✅ Kalibrierung gespeichert:", str(vision.DEFAULT_PATH))
        else:
            cfg = JsonStore.load(str(config.config_path))
            if cfg is None:
                raise RuntimeError(
                    f"❌ Kalibrierungsdatei nicht gefunden: {str(config.config_path)}"
                )
            self.cfg = cfg
            print("📁 Kalibrierung geladen:", str(config.config_path))

        self.video.reset()
