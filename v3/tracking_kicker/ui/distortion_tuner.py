import cv2
import json
from pathlib import Path
from v3.tracking_kicker.model.Camera import Camera


class DistortionTunerUI:
    def __init__(self, camera: Camera, win="Distortion Tuning"):
        self.camera = camera
        self.win = win
        self._dirty = True

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        self._create("k1", camera.k1, -0.5, 0.5)
        self._create("k2", camera.k2, -0.5, 0.5)
        self._create("k3", camera.k3, -0.5, 0.5)
        self._create("p1", camera.p1, -0.2, 0.2)
        self._create("p2", camera.p2, -0.2, 0.2)
        self._create("scale", camera.scale, 0.8, 1.2)

    # -------------------------------------------------

    def _create(self, name, value, vmin, vmax):
        self.__dict__[f"{name}_min"] = vmin
        self.__dict__[f"{name}_max"] = vmax

        pos = int((value - vmin) / (vmax - vmin) * 1000)

        cv2.createTrackbar(
            name,
            self.win,
            pos,
            1000,
            self._on_change
        )

    def _on_change(self, _):
        # ❗ WICHTIG: NICHT lesen
        self._dirty = True

    def update_camera_if_needed(self):
        if not self._dirty:
            return

        self.camera.k1 = self._read("k1")
        self.camera.k2 = self._read("k2")
        self.camera.k3 = self._read("k3")
        self.camera.p1 = self._read("p1")
        self.camera.p2 = self._read("p2")
        self.camera.scale = self._read("scale")

        self._dirty = False

    def _read(self, name):
        pos = cv2.getTrackbarPos(name, self.win)
        vmin = self.__dict__[f"{name}_min"]
        vmax = self.__dict__[f"{name}_max"]
        return vmin + (vmax - vmin) * (pos / 1000.0)

    def save_to_config(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg.setdefault("camera", {}).update({
            "k1": self.camera.k1,
            "k2": self.camera.k2,
            "k3": self.camera.k3,
            "p1": self.camera.p1,
            "p2": self.camera.p2,
            "scale": self.camera.scale,
        })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        print("💾 Distortion gespeichert")
