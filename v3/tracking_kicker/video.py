import cv2
import numpy as np
from typing import Optional, Tuple


class VideoSource:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        print("Frames:", self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Dauer:", self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # -> 186.636

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Konnte Video nicht öffnen: {video_path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 else 186.6  # Fallback

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()
