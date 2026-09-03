"""Replay source implementing the same interface as a live camera."""

import time

import numpy as np

from ..Objects.Frames import Frame


MIN_DT = 2.0 / 210.0


class RawReplaySource:
    def __init__(self, path: str):
        # Close the NPZ handle immediately after copying the arrays.  Keeping
        # it alive for the whole application can otherwise leak file handles
        # when the source is recreated during a restart.
        with np.load(path) as data:
            self.frames = np.array(data["frames"], copy=True)
            self.times = np.array(data["times"], dtype=float, copy=True)
        self._i = 0
        self._t0 = time.perf_counter()

        if len(self.times) >= 2:
            self.fps = float(1.0 / np.mean(np.diff(self.times)))
        else:
            self.fps = 60.0

        self._frame_dt = 1.0 / self.fps
        self._loop_duration = (
            float(self.times[-1]) + self._frame_dt
            if len(self.times) > 0
            else self._frame_dt
        )
        self._time_offset = 0.0

    def read(self):
        if len(self.frames) == 0:
            return False, None

        if self._i >= len(self.frames):
            self._time_offset += self._loop_duration
            self._i = 0
            self._t0 = time.perf_counter()

        target_time = self._t0 + self.times[self._i]
        target_time = max(target_time, self._t0 + self._i * MIN_DT)
        now = time.perf_counter()
        if now < target_time:
            time.sleep(target_time - now)

        obj_frame = Frame()
        obj_frame.frame = self.frames[self._i]
        obj_frame.timestamp = self._time_offset + self.times[self._i]
        self._i += 1
        return True, obj_frame

    def reset(self):
        self._i = 0
        self._t0 = time.perf_counter()
        self._time_offset = 0.0

    def release(self):
        pass

    def showFPS(self) -> str:
        return f"{self.fps:.2f}"
