# VideoSource.py
from __future__ import annotations

from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VideoSource:
    """
    Universelle Video-Quelle für OpenCV:
    - Videodatei (Pfad)
    - Webcam (Index, z.B. 0)
    - Stream-URL (RTSP/HTTP)

    API:
      - fps: float
      - read() -> (ret: bool, frame: np.ndarray|None)
      - reset()
      - release()
    """

    def __init__(self, source: Union[str, int]):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Konnte Quelle nicht öffnen: {source}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Bei Webcams/Streams liefert OpenCV oft 0 -> Fallback
        self.fps = fps if fps > 0 else 30.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def reset(self) -> None:
        # Nur bei Video-Dateien sinnvoll; bei Webcam/RTSP egal
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()


# -------------------------------------------------------------------
# OPTIONAL: BaslerSource (nur wenn pypylon installiert ist)
# -------------------------------------------------------------------

class BaslerSource:
    """
    Basler-Kamera Quelle (pypylon).
    Benötigt:
      pip install pypylon
      + Basler pylon Software Suite installiert.

    API wie VideoSource:
      fps, read(), reset(), release()
    """

    def __init__(
        self,
        fps_hint: float = 220.0,
        max_grab_ms: int = 5000,
        latest_only: bool = True,
        force_fps: Optional[float] = None,
    ):
        from pypylon import pylon  # lazy import, damit VideoSource alleine geht

        self._pylon = pylon
        self.max_grab_ms = max_grab_ms
        self._closed = False

        factory = pylon.TlFactory.GetInstance()
        device = factory.CreateFirstDevice()
        self.camera = pylon.InstantCamera(device)
        self.camera.Open()

        self.camera.Height.Value = 736
        self.camera.OffsetY.Value = 200

        self.camera.Width.Value = 1100
        self.camera.OffsetX.Value = 220


        print("✅ Basler geöffnet:", self.camera.GetDeviceInfo().GetModelName())

        # Optional FPS setzen
        if force_fps is not None:
            try:
                if hasattr(self.camera, "AcquisitionFrameRateEnable"):
                    self.camera.AcquisitionFrameRateEnable.SetValue(True)
                if hasattr(self.camera, "AcquisitionFrameRate"):
                    self.camera.AcquisitionFrameRate.SetValue(float(force_fps))
                print(f"🎯 Kamera-FPS gesetzt auf: {force_fps}")
            except Exception as e:
                print(f"⚠️ Konnte Kamera-FPS nicht setzen ({e}). Nutze Auto/Default.")

        # FPS lesen
        self.fps = fps_hint
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                cam_fps = float(self.camera.ResultingFrameRate.GetValue())
                if cam_fps > 0:
                    self.fps = cam_fps
            elif hasattr(self.camera, "AcquisitionFrameRate"):
                cam_fps = float(self.camera.AcquisitionFrameRate.GetValue())
                if cam_fps > 0:
                    self.fps = cam_fps
        except Exception:
            pass

        # Converter -> BGR
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        strategy = pylon.GrabStrategy_LatestImageOnly if latest_only else pylon.GrabStrategy_OneByOne
        self.camera.StartGrabbing(strategy)

        cv2.setUseOptimized(True)
        print(f"📷 Live FPS (für Analyzer/Delay): {self.fps:.2f}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._closed or not self.camera.IsGrabbing():
            return False, None

        grab = self.camera.RetrieveResult(self.max_grab_ms, self._pylon.TimeoutHandling_Return)
        if grab is None:
            return False, None

        try:
            if not grab.GrabSucceeded():
                return False, None
            img = self.converter.Convert(grab)
            frame = img.GetArray()  # BGR8packed
            return True, frame
        finally:
            grab.Release()

    def reset(self) -> None:
        # Live Kamera: kein Reset
        return

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception:
            pass
        try:
            if self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass


# -------------------------------------------------------------------
# Mini-Demo: Direkt ausführen
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Video-Datei:
    #src = VideoSource(r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos")

    # 2) Webcam:
    # src = VideoSource(0)

    # 3) Basler:
    src = BaslerSource()

    delay_ms = int(round(1000 / max(1.0, src.fps)))

    while True:
        ret, frame = src.read()
        if not ret or frame is None:
            break

        cv2.imshow("Source Test", frame)
        k = cv2.waitKey(delay_ms) & 0xFF
        if k == 27:
            break
        elif k == ord('r'):
            src.reset()

    src.release()
    cv2.destroyAllWindows()
