import cv2
import numpy as np
from typing import List, Optional


class RodCalibrator:
    def calibrate(self, frame_bgr: np.ndarray, n_rods: int = 8) -> Optional[List[float]]:
        h, w = frame_bgr.shape[:2]
        clicks: List[int] = []

        def on_mouse(e, x, y, flags, param):
            if e == cv2.EVENT_LBUTTONDOWN:
                clicks.append(int(x))

        win = f"STANGEN ({n_rods} Klicks links->rechts, 's' zum Speichern)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            vis = frame_bgr.copy()
            for i, x in enumerate(sorted(clicks)):
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 2)
                cv2.putText(vis, str(i + 1), (x + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(win, vis)

            k = cv2.waitKey(15) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(win)
                return None
            if k == ord('s') and len(clicks) >= n_rods:
                cv2.destroyWindow(win)
                xs = sorted(clicks)[:n_rods]
                return [x / w for x in xs]
