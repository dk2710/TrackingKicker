# tracking_kicker/distortion.py
from __future__ import annotations

from dataclasses import dataclass, asdict
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
    scale: float = 1.00  # wichtig: 1.0, nicht 0.0


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_params(config_path: str | Path, params: DistortionParams) -> Path:
    p = Path(config_path).expanduser().resolve()
    _ensure_parent_dir(p)

    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=4)
    return p


def load_or_create_params(config_path: str | Path, default: DistortionParams | None = None) -> DistortionParams:
    """
    Lädt Params aus JSON. Wenn Datei nicht existiert oder kaputt ist:
    -> erstellt sie (mit Defaults) und liefert Defaults zurück.
    """
    default = default or DistortionParams()
    p = Path(config_path).expanduser().resolve()
    _ensure_parent_dir(p)

    if not p.exists():
        save_params(p, default)
        return default

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        # JSON kaputt -> neu schreiben
        save_params(p, default)
        return default

    return DistortionParams(
        k1=float(cfg.get("k1", default.k1)),
        k2=float(cfg.get("k2", default.k2)),
        k3=float(cfg.get("k3", default.k3)),
        p1=float(cfg.get("p1", default.p1)),
        p2=float(cfg.get("p2", default.p2)),
        scale=float(cfg.get("scale", default.scale)),
    )


class DistortionCorrector:
    """
    Korrigiert Verzerrung ausschließlich über JSON-Config.
    Keine UI, keine Trackbars. JSON wird bei Bedarf automatisch erstellt.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.params = load_or_create_params(self.config_path)

        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._last_key: tuple | None = None  # (w,h,k1,k2,k3,p1,p2,scale)

    @staticmethod
    def _compute_map(w: int, h: int, k1: float, k2: float, k3: float, p1: float, p2: float, scale: float):
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.indices((h, w), np.float32)

        # normiert auf [-1..1] ungefähr (über cx/cy)
        x = (xs - cx) / cx
        y = (ys - cy) / cy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        x2 = (x * radial + x_t) * scale
        y2 = (y * radial + y_t) * scale

        mx = (x2 * cx + cx).astype(np.float32)
        my = (y2 * cy + cy).astype(np.float32)
        return mx, my

    def reload_from_json(self) -> None:
        """
        Optional: falls du die JSON während Laufzeit manuell editierst,
        kannst du damit neu laden.
        """
        self.params = load_or_create_params(self.config_path, self.params)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._map_x = None
        self._map_y = None
        self._last_key = None

    def apply(self, frame: np.ndarray, *, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        p = self.params
        key = (w, h, p.k1, p.k2, p.k3, p.p1, p.p2, p.scale)

        if key != self._last_key or self._map_x is None:
            self._map_x, self._map_y = self._compute_map(w, h, p.k1, p.k2, p.k3, p.p1, p.p2, p.scale)
            self._last_key = key

        return cv2.remap(frame, self._map_x, self._map_y, interpolation)

    def ensure_json_exists(self) -> Path:
        """
        Stellt sicher, dass die JSON existiert (falls du das explizit triggern willst).
        """
        if not self.config_path.exists():
            save_params(self.config_path, self.params)
        return self.config_path
