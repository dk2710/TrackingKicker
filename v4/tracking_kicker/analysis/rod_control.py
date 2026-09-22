from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RodControlConfig:
    frame_window: int = 30   # wie viele Frames Zeit
    min_rods: int = 2        # Safety


class RodControlTracker:
    """
    Erkennt Ballkontrolle zwischen Stangen anhand virtueller Midlines.
    """

    def __init__(self, rods_x_norm: List[float], cfg: RodControlConfig | None = None):
        self._prev_x: float | None = None
        self._last_zone: int | None = None
        self._zone_start: int = 0
        self._frame_idx: int = 0
        self._max_zone: int = 0

        if len(rods_x_norm) < 2:
            raise ValueError("Mindestens 2 Stangen nötig")

        self.rods = rods_x_norm
        self.cfg = cfg or RodControlConfig()

        # Midlines (1.5, 2.5, ...)
        self.mids = [
            (rods_x_norm[i] + rods_x_norm[i + 1]) / 2
            for i in range(len(rods_x_norm) - 1)
        ]

        # Zähler pro Stange
        self.control_count = [0 for _ in rods_x_norm]

        # Pending: (mid_index, frame_idx)
        self._pending: Optional[Tuple[int, int]] = None
        self._frame_idx = 0

    # -------------------------------------------------

    def update(self, x_norm: float) -> None:
        self._frame_idx += 1
        zone = self._zone_of(x_norm)

        # erstes Frame
        if self._last_zone is None:
            self._last_zone = zone
            self._zone_start = self._frame_idx
            self._max_zone = zone
            return

        # Zone gewechselt
        if zone != self._last_zone:
            duration = self._frame_idx - self._zone_start

            # ✅ ausreichend lange in Zone (oder tiefer) gewesen
            if duration >= self.cfg.frame_window:
                rod_idx = self._max_zone
                self.control_count[rod_idx] += 1
                print(f"[ROD] ✅ Kontrolle an Rod {rod_idx + 1}")

            # neue Phase starten
            self._zone_start = self._frame_idx
            self._max_zone = zone

        else:
            # gleiche Zone → ggf. tiefere Zone merken
            self._max_zone = max(self._max_zone, zone)

        self._last_zone = zone

    # -------------------------------------------------

    def reset(self) -> None:
        self._pending = None

    def get_counts(self) -> List[int]:
        return self.control_count

    def _zone_of(self, x_norm: float) -> int:
        for i, mid in enumerate(self.mids):
            if x_norm < mid:
                return i
        return len(self.mids)
