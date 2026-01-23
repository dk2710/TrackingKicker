import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from pypylon import pylon


DISTORTION_CFG = {
    "k1": -0.167,
    "k2": 0.05,
    "k3": -0.005,
    "p1": 0.014,
    "p2": 0.005,
    "scale": 1.12
}


class BaslerSource:
    def __init__(
        self,
        fps_hint: float = 220.0,
        max_grab_ms: int = 5000,
        latest_only: bool = True,
        force_fps: Optional[float] = None,
        undistort: bool = True,
        distortion_cfg: Optional[Dict] = None,
    ):
        self.max_grab_ms = max_grab_ms
        self._closed = False

        self.undistort = undistort
        self.dist_cfg = distortion_cfg or DISTORTION_CFG

        # --- Basler öffnen ---
        factory = pylon.TlFactory.GetInstance()
        device = factory.CreateFirstDevice()
        self.camera = pylon.InstantCamera(device)
        self.camera.Open()

        self.camera.Height.Value = 750
        self.camera.OffsetY.Value = 230

        self.camera.Width.Value = 1250
        self.camera.OffsetX.Value = 150

        print(f"✅ Basler geöffnet: {self.camera.GetDeviceInfo().GetModelName()}")

        # Optional FPS setzen
        if force_fps is not None:
            try:
                if hasattr(self.camera, "AcquisitionFrameRateEnable"):
                    self.camera.AcquisitionFrameRateEnable.SetValue(True)
                if hasattr(self.camera, "AcquisitionFrameRate"):
                    self.camera.AcquisitionFrameRate.SetValue(float(force_fps))
                print(f"🎯 Kamera-FPS gesetzt auf: {force_fps}")
            except Exception as e:
                print(f"⚠️ Konnte Kamera-FPS nicht setzen ({e}).")

        # FPS lesen
        self.fps = fps_hint
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                v = float(self.camera.ResultingFrameRate.GetValue())
                if v > 0:
                    self.fps = v
        except Exception:
            pass

        # Converter -> BGR
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        strategy = pylon.GrabStrategy_LatestImageOnly if latest_only else pylon.GrabStrategy_OneByOne
        self.camera.StartGrabbing(strategy)
        time.sleep(0.05)

        cv2.setUseOptimized(True)
        print(f"📷 Live FPS (für Analyzer/Delay): {self.fps:.2f}")

        # --- Undistort maps vorbereiten (schnell!) ---
        self._map1 = None
        self._map2 = None
        self._undist_ready = False

        if self.undistort:
            # einmal ein Frame holen um (w,h) zu kennen
            ok, frame = self._grab_raw_once()
            if ok and frame is not None:
                self._init_undistort_maps(frame.shape[1], frame.shape[0])
            else:
                print("⚠️ Konnte kein Frame für Undistort-Init holen. Undistort deaktiviert.")
                self.undistort = False

    def _grab_raw_once(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Einmalig raw Frame grabben (ohne Undistort)."""
        if self._closed or not self.camera.IsGrabbing():
            return False, None

        grab = self.camera.RetrieveResult(self.max_grab_ms, pylon.TimeoutHandling_Return)
        if grab is None:
            return False, None
        try:
            if not grab.GrabSucceeded():
                return False, None
            img = self.converter.Convert(grab)
            return True, img.GetArray()
        finally:
            grab.Release()

    def _init_undistort_maps(self, w: int, h: int) -> None:
        # Distortion coeffs
        k1 = float(self.dist_cfg["k1"])
        k2 = float(self.dist_cfg["k2"])
        p1 = float(self.dist_cfg["p1"])
        p2 = float(self.dist_cfg["p2"])
        k3 = float(self.dist_cfg["k3"])
        scale = float(self.dist_cfg.get("scale", 1.0))

        D = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        # --- Kameramatrix K (Näherung) ---
        # Wenn du echte fx/fy/cx/cy hast, hier ersetzen!
        cx = w * 0.5
        cy = h * 0.5
        f = 0.9 * w  # grobe Näherung, funktioniert oft "okay" als Startwert

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Optimal new camera matrix (alpha=0 -> keine schwarzen Ränder)
        newK, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)

        # dein "scale": >1 = reinzoomen (weniger Rand), <1 = rauszoomen
        newK[0, 0] *= scale
        newK[1, 1] *= scale

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, newK, (w, h), m1type=cv2.CV_16SC2
        )
        self._undist_ready = True
        print(f"✅ Undistort bereit (w={w}, h={h}, scale={scale})")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # raw grab
        ok, frame = self._grab_raw_once()
        if not ok or frame is None:
            return False, None

        # undistort
        if self.undistort and self._undist_ready:
            frame = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

        return True, frame

    def reset(self) -> None:
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
