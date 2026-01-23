# tracking_kicker/distortion.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import cv2


@dataclass
class DistortionParams:
    k1: float = -0.20
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00  # ✅ wichtig: 1.0, nicht 0.0


def load_params(config_path: str | Path) -> DistortionParams:
    p = Path(config_path)
    if not p.exists():
        return DistortionParams()

    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return DistortionParams(
        k1=float(cfg.get("k1", -0.20)),
        k2=float(cfg.get("k2", 0.05)),
        k3=float(cfg.get("k3", 0.00)),
        p1=float(cfg.get("p1", 0.00)),
        p2=float(cfg.get("p2", 0.00)),
        scale=float(cfg.get("scale", 1.00)),
    )


def save_params(config_path: str | Path, params: DistortionParams) -> None:
    p = Path(config_path)
    data = {
        "k1": params.k1,
        "k2": params.k2,
        "k3": params.k3,
        "p1": params.p1,
        "p2": params.p2,
        "scale": params.scale,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Verzerrung gespeichert: {p} -> {data}")


class DistortionCorrector:
    """
    Nutzt deine DIY-Map. Optional mit UI-Trackbars zum Tuning.
    """

    def __init__(self, params: DistortionParams, *, config_path: str | Path | None = None, enable_ui: bool = False):
        self.params = params
        self.config_path = Path(config_path) if config_path is not None else None

        self._map_x = None
        self._map_y = None
        self._last_key = None  # (w,h,k1,k2,k3,p1,p2,scale)

        # UI
        self._ui_enabled = False
        self._ui_window = "Verzerrung"
        if enable_ui:
            self.enable_ui(True)

    # -----------------------------
    # Map / Remap
    # -----------------------------
    @staticmethod
    def _compute_map(w, h, k1, k2, k3, p1, p2, scale):
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.indices((h, w), np.float32)
        x = (xs - cx) / cx
        y = (ys - cy) / cy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_t = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_t = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

        x2 = (x * radial + x_t) * scale
        y2 = (y * radial + y_t) * scale

        mx = (x2 * cx + cx).astype(np.float32)
        my = (y2 * cy + cy).astype(np.float32)
        return mx, my


    def _invalidate_cache(self):
        self._map_x = None
        self._map_y = None
        self._last_key = None

    def apply(self, frame):
        print("apply params:", self.params)

        if frame is None:
            return None

        # falls UI aktiv: live Parameter einlesen
        if self._ui_enabled:
            self.update_from_ui()

        h, w = frame.shape[:2]
        p = self.params
        key = (w, h, p.k1, p.k2, p.k3, p.p1, p.p2, p.scale)

        if key != self._last_key or self._map_x is None:
            self._map_x, self._map_y = self._compute_map(w, h, p.k1, p.k2, p.k3, p.p1, p.p2, p.scale)
            self._last_key = key

        return cv2.remap(frame, self._map_x, self._map_y, cv2.INTER_LINEAR)

    # -----------------------------
    # UI Trackbars
    # -----------------------------
    def enable_ui(self, enabled: bool = True):
        if enabled and not self._ui_enabled:
            cv2.namedWindow(self._ui_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._ui_window, 650, 260)

            # Trackbar-Ranges:
            # k/p: -2.000 .. +2.000 (x1000)
            # scale: 0.50 .. 1.50 (x1000)
            cv2.createTrackbar("k1 x1000", self._ui_window, 0, 4000, lambda v: None)
            cv2.createTrackbar("k2 x1000", self._ui_window, 0, 4000, lambda v: None)
            cv2.createTrackbar("k3 x1000", self._ui_window, 0, 4000, lambda v: None)
            cv2.createTrackbar("p1 x1000", self._ui_window, 0, 4000, lambda v: None)
            cv2.createTrackbar("p2 x1000", self._ui_window, 0, 4000, lambda v: None)

            cv2.createTrackbar("scale x1000", self._ui_window, 0, 2000, lambda v: None)

            # initial Werte setzen
            self._set_trackbar_values_from_params()

            print("🔧 Verzerrungs-Regler aktiv. (Tipp: im App-Window 's' drücken -> speichern)")
            self._ui_enabled = True

        elif (not enabled) and self._ui_enabled:
            try:
                cv2.destroyWindow(self._ui_window)
            except Exception:
                pass
            self._ui_enabled = False

    def _set_trackbar_values_from_params(self):
        p = self.params
        # offset 2000 => -2..+2
        cv2.setTrackbarPos("k1 x1000", self._ui_window, int(round(p.k1 * 1000 + 2000)))
        cv2.setTrackbarPos("k2 x1000", self._ui_window, int(round(p.k2 * 1000 + 2000)))
        cv2.setTrackbarPos("k3 x1000", self._ui_window, int(round(p.k3 * 1000 + 2000)))
        cv2.setTrackbarPos("p1 x1000", self._ui_window, int(round(p.p1 * 1000 + 2000)))
        cv2.setTrackbarPos("p2 x1000", self._ui_window, int(round(p.p2 * 1000 + 2000)))

        # scale: 0..2.0 => x1000
        cv2.setTrackbarPos("scale x1000", self._ui_window, int(round(p.scale * 1000)))

    def update_from_ui(self):
        """
        Liest die Trackbars und schreibt in self.params.
        Invalidate Cache nur wenn sich Werte geändert haben.
        """
        if not self._ui_enabled:
            return

        def read_k(name: str) -> float:
            return (cv2.getTrackbarPos(name, self._ui_window) - 2000) / 1000.0

        new_k1 = read_k("k1 x1000")
        new_k2 = read_k("k2 x1000")
        new_k3 = read_k("k3 x1000")
        new_p1 = read_k("p1 x1000")
        new_p2 = read_k("p2 x1000")
        new_scale = cv2.getTrackbarPos("scale x1000", self._ui_window) / 1000.0

        # clamp scale sinnvoll
        new_scale = max(0.1, min(3.0, new_scale))

        p = self.params
        changed = (
            new_k1 != p.k1 or new_k2 != p.k2 or new_k3 != p.k3 or
            new_p1 != p.p1 or new_p2 != p.p2 or new_scale != p.scale
        )

        if changed:
            p.k1, p.k2, p.k3, p.p1, p.p2, p.scale = new_k1, new_k2, new_k3, new_p1, new_p2, new_scale
            self._invalidate_cache()

    def save(self):
        """Speichert aktuelle Params, falls config_path gesetzt ist."""
        if self.config_path is None:
            print("⚠️ Kein config_path gesetzt – kann nicht speichern.")
            return
        save_params(self.config_path, self.params)
