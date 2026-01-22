import json
import numpy as np
import cv2
from typing import Optional, Tuple


class LensUndistorter:
    def __init__(self, config_path: str = "lens_config.json"):
        self.config_path = config_path

        # Defaults (falls keine Datei existiert)
        self.k1 = -0.20
        self.k2 = 0.05
        self.k3 = 0.00
        self.p1 = 0.00
        self.p2 = 0.00
        self.scale = 1.00

        # Cache
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._last_key: Optional[Tuple[int, int, float, float, float, float, float, float]] = None

        self.reload()  # einmal laden beim Start

    def reload(self) -> None:
        """Lädt lens_config.json neu (wenn vorhanden)."""
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)

            self.k1 = float(cfg["k1"])
            self.k2 = float(cfg["k2"])
            self.k3 = float(cfg["k3"])
            self.p1 = float(cfg["p1"])
            self.p2 = float(cfg["p2"])
            self.scale = float(cfg["scale"])

            print("📁 Verzerrung geladen:", cfg)
        except Exception as e:
            print("⚠️ Keine gespeicherten Verzerrungsdaten gefunden – nutze Default-Werte.", e)

        # Cache invalidieren (damit neue Map berechnet wird)
        self._map_x = None
        self._map_y = None
        self._last_key = None

    def _compute_map(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.indices((h, w), np.float32)

        x = (xs - cx) / cx
        y = (ys - cy) / cy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        x_t = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_t = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

        x2 = (x * radial + x_t) * self.scale
        y2 = (y * radial + y_t) * self.scale

        mx = (x2 * cx + cx).astype(np.float32)
        my = (y2 * cy + cy).astype(np.float32)
        return mx, my

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Wendet die Entzerrung auf ein BGR-Frame an."""
        h, w = frame_bgr.shape[:2]

        key = (w, h, self.k1, self.k2, self.k3, self.p1, self.p2, self.scale)
        if self._map_x is None or self._map_y is None or self._last_key != key:
            self._map_x, self._map_y = self._compute_map(w, h)
            self._last_key = key

        return cv2.remap(frame_bgr, self._map_x, self._map_y, cv2.INTER_LINEAR)
