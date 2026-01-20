import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class FieldCalibrator:
    """
    Click order:
    1) top-left
    2) top-right
    3) bottom-right
    4) bottom-left
    Stores normalized coordinates (x/w, y/h).
    """
    def calibrate(self, frame_bgr: np.ndarray) -> Optional[List[List[float]]]:
        h, w = frame_bgr.shape[:2]
        pts: List[Tuple[int, int]] = []

        def on_mouse(e, x, y, flags, param):
            if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((int(x), int(y)))

        win = "SPIELFELD-ECKEN (4 Klicks: OL, OR, UR, UL) | 'r' reset | 's' save"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)

        labels = ["1 OL", "2 OR", "3 UR", "4 UL"]

        while True:
            vis = frame_bgr.copy()

            # draw points + labels
            for i, (x, y) in enumerate(pts):
                cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(vis, labels[i], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # draw polygon if 4 points
            if len(pts) == 4:
                poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=3)

            cv2.putText(vis, "Click order: top-left, top-right, bottom-right, bottom-left",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(win, vis)
            k = cv2.waitKey(15) & 0xFF

            if k == 27:  # ESC
                cv2.destroyWindow(win)
                return None
            if k == ord('r'):
                pts.clear()
            if k == ord('s') and len(pts) == 4:
                cv2.destroyWindow(win)
                # normalize
                corners_norm = [[x / w, y / h] for (x, y) in pts]
                return corners_norm
