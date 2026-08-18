
import time
import numpy as np
from v3.tracking_kicker.Objects.Frames import Frame

MIN_DT = 2.0 / 210.0  # harte Kamera-Grenze


class RawReplaySource:
    def __init__(self, path: str):
        data = np.load(path)
        self.objFrame = Frame()
        self.frames = data["frames"]
        self.times = data["times"]  # bereits RELATIV

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

        # Replay automatisch von vorne starten, sobald das Video endet.
        # Der Zeitstempel bleibt dabei monoton, damit die Historie und die
        # Geschwindigkeitsanalyse keinen Sprung zurueck auf 0 sehen.
        if self._i >= len(self.frames):
            self._time_offset += self._loop_duration
            self._i = 0
            self._t0 = time.perf_counter()

        # 🎯 Zielzeit für Playback (Realtime-Synchronisation)
        target_time = self._t0 + self.times[self._i]
        target_time = max(target_time, self._t0 + self._i * MIN_DT)

        now = time.perf_counter()
        if now < target_time:
            time.sleep(target_time - now)

        # ✅ Neues Frame-Objekt erzeugen
        objFrame = Frame()
        objFrame.frame = self.frames[self._i]

        # ✅ WICHTIG: Zeit aus dem VIDEO!
        objFrame.timestamp = self._time_offset + self.times[self._i]

        self._i += 1
        return True, objFrame

    def reset(self):
        self._i = 0
        self._t0 = time.perf_counter()
        self._time_offset = 0.0

    def release(self):
        pass

    def showFPS(self) -> str:
        return f"{self.fps:.2f}"


