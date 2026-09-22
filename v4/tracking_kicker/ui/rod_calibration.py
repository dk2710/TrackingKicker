import cv2
from typing import List
from dataclasses import dataclass


@dataclass
class RodCalibrationResult:
    xs_norm: List[float]


class RodCalibrationUI:
    def __init__(self, window_name="Rod Calibration"):
        self.win = window_name
        self.xs_px: List[int] = []
        self.frame = None
        self._w = 1

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.xs_px) < 8:
                self.xs_px.append(x)
                print(f"➕ Stange bei x={x}")
            else:
                print("⚠️ Maximal 8 Stangen erlaubt")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.xs_px:
                removed = self.xs_px.pop()
                print(f"➖ Entfernt x={removed}")

    def run(self, frame) -> RodCalibrationResult | None:
        self.frame = frame
        self._w = frame.shape[1]

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

        print("🟦 Linksklick = Stange setzen")
        print("🟥 Rechtsklick = letzte entfernen")
        print("💾 S = Speichern | ESC = Abbrechen")

        while True:
            vis = frame.copy()

            for x in self.xs_px:
                cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 255, 255), 2)

            cv2.putText(
                vis,
                f"Stangen: {len(self.xs_px)}/8",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(self.win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                cv2.destroyWindow(self.win)
                return None

            if key == ord("s"):
                if len(self.xs_px) != 8:
                    print("❌ Genau 8 Stangen benötigt!")
                    continue

                xs_norm = sorted([x / self._w for x in self.xs_px])
                cv2.destroyWindow(self.win)
                return RodCalibrationResult(xs_norm=xs_norm)
