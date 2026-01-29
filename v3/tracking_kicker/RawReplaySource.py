import time
import numpy as np

class RawReplaySource:
    def __init__(self, path: str):
        data = np.load(path)

        self.frames = data["frames"]
        self.times = data["times"]

        # relative Zeiten (Start = 0)
        self.times = self.times - self.times[0]

        self._i = 0
        self._t0 = time.perf_counter()

        # für Anzeige / API
        if len(self.times) >= 2:
            self.fps = float(1.0 / np.mean(np.diff(self.times)))
        else:
            self.fps = 60.0

    def read(self):
        if self._i >= len(self.frames):
            return False, None

        # Soll-Zeitpunkt dieses Frames
        target_t = self._t0 + self.times[self._i]

        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)

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
