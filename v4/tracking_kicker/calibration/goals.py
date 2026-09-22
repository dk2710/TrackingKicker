import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class GoalCalibrator:
    def calibrate(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        clicks: List[Tuple[int, int]] = []

        def on_mouse(e, x, y, flags, param):
            if e == cv2.EVENT_LBUTTONDOWN:
                clicks.append((int(x), int(y)))

        win = "TOR-KALIBRIERUNG (4 Klicks, 's' zum Speichern)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            vis = frame_bgr.copy()
            if len(clicks) >= 2:
                cv2.rectangle(vis, clicks[0], clicks[1], (0, 0, 255), 2)
            if len(clicks) >= 4:
                cv2.rectangle(vis, clicks[2], clicks[3], (255, 0, 0), 2)
            cv2.imshow(win, vis)

            k = cv2.waitKey(15) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(win)
                return None
            if k == ord('s') and len(clicks) == 4:
                cv2.destroyWindow(win)
                (x0, y0), (x1, y1), (x2, y2), (x3, y3) = clicks

                return {
                    "left": {
                        "x0": min(x0, x1) / w, "y0": min(y0, y1) / h,
                        "x1": max(x0, x1) / w, "y1": max(y0, y1) / h
                    },
                    "right": {
                        "x0": min(x2, x3) / w, "y0": min(y2, y3) / h,
                        "x1": max(x2, x3) / w, "y1": max(y2, y3) / h
                    }
                }
