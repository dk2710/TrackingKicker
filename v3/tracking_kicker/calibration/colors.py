import cv2
import numpy as np
from typing import Any, Dict, List, Tuple


class ColorCalibrator:
    @staticmethod
    def roi_to_model(lab_roi: np.ndarray, min_sigma: float = 10.0) -> Tuple[List[float], List[float], float]:
        A = lab_roi.reshape(-1, 3).astype(np.float32)
        mu = A.mean(axis=0)
        sigma = np.maximum(A.std(axis=0), min_sigma)
        d2 = np.sum(((A - mu) / sigma) ** 2, axis=1)
        thr = float(np.percentile(d2, 99.5) * 1.6 + 1.0)
        return mu.tolist(), sigma.tolist(), thr

    def calibrate(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        win = "FARB-KALIBRIERUNG (ENTER bestätigt)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        print("🟦 WHITE ROI wählen")
        r1 = cv2.selectROI(win, frame_bgr, False)
        print("⬛ BLACK ROI wählen")
        r2 = cv2.selectROI(win, frame_bgr, False)
        print("🟩 BG ROI wählen")
        r3 = cv2.selectROI(win, frame_bgr, False)
        cv2.destroyWindow(win)

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        def crop(r):
            x, y, w, h = r
            if w <= 0 or h <= 0:
                raise RuntimeError("❌ ROI ungültig (Breite/Höhe = 0).")
            return lab[y:y + h, x:x + w]

        mu_w, sg_w, thr_w = self.roi_to_model(crop(r1))
        mu_b, sg_b, thr_b = self.roi_to_model(crop(r2), min_sigma=14.0)
        mu_bg, sg_bg, thr_bg = self.roi_to_model(crop(r3))

        return {
            "white": {"mu": mu_w, "sigma": sg_w, "thr": thr_w},
            "black": {"mu": mu_b, "sigma": sg_b, "thr": thr_b},
            "bg": {"mu": mu_bg, "sigma": sg_bg, "thr": thr_bg},
            "bg_margin": 0.92
        }
