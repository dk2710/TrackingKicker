import time
import numpy as np
MIN_DT = 2.0 / 210.0  # harte Kamera-Grenze

class RawReplaySource:
    def __init__(self, path: str):
        data = np.load(path)
        self.frames = data["frames"]
        self.times = data["times"]  # bereits RELATIV

        self._i = 0
        self._t0 = time.perf_counter()

        if len(self.times) >= 2:
            self.fps = float(1.0 / np.mean(np.diff(self.times)))
        else:
            self.fps = 60.0



    def read(self):
        if self._i >= len(self.frames):
            return False, None

        target_time = self._t0 + self.times[self._i]
        target_time = max(target_time, self._t0 + self._i * MIN_DT)

        now = time.perf_counter()
        if now < target_time:
            time.sleep(target_time - now)

        frame = self.frames[self._i]
        self._i += 1
        return True, frame

    def reset(self):
        self._i = 0
        self._t0 = time.perf_counter()

    def release(self):
        pass

    def showFPS(self) -> str:
        return f"{self.fps:.2f}"


