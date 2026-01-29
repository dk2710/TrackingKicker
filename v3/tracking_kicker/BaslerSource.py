import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from pypylon import pylon, genicam


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
        fps_hint: float = 0,
        max_grab_ms: int = 5000,
        latest_only: bool = False,
        force_fps: float = None,
    ):
        self.max_grab_ms = max_grab_ms
        self._closed = False

        #Für FPS Logs
        self._t0 = time.time()
        self._n = 0
        self._fps_host = 0.0

        # --- Basler öffnen ---
        factory = pylon.TlFactory.GetInstance()
        device = factory.CreateFirstDevice()
        self.camera = pylon.InstantCamera(device)
        self.camera.Open()

        self.logging(self.camera)

        self.camera.Height.Value = 650
        self.camera.OffsetY.Value = 266

        self.camera.Width.Value = 1100
        self.camera.OffsetX.Value = 240


        print(f"✅ Basler geöffnet: {self.camera.GetDeviceInfo().GetModelName()}")


        if force_fps is not None:
            # Setzen (wenn Nodes existieren)
            self.try_set(self.camera, "AcquisitionFrameRateEnable", True)
            self.try_set(self.camera, "AcquisitionFrameRate", float(force_fps))

        # Immer: nur lesen (Diagnose)
        fps_res = self.try_get(self.camera, "BslResultingAcquisitionFrameRate")
        if fps_res is None:
            fps_res = self.try_get(self.camera, "ResultingFrameRate")

        print("➡️ Resulting FPS:", fps_res)

        # FPS lesen
        self.fps = fps_hint
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                v = float(self.camera.ResultingFrameRate.GetValue())
                if v > 0:
                    self.fps = v
        except Exception:
            pass

        print(f"{self.getTemperatur(self.camera)}")


        strategy = pylon.GrabStrategy_LatestImageOnly if True else pylon.GrabStrategy_OneByOne
        self.camera.StartGrabbing(strategy)
        time.sleep(0.05)

#------------------------------------------------ --------------------------------------------------------------------------------
    def showfps(self) -> str:
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                v = float(self.camera.ResultingFrameRate.GetValue())
                if v > 0:
                    self.fps = v
                    return str(self.fps)
        except Exception:
            return str(Exception)


    def _grab_raw_once(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Einmalig raw Frame grabben (ohne Undistort)."""


        if self._closed or not self.camera.IsGrabbing():
            return False, None

        grab = None
        try:
            grab = self.camera.RetrieveResult(
                self.max_grab_ms,
                pylon.TimeoutHandling_Return
            )

            if grab is None or not grab.GrabSucceeded():
                return False, None

            frame = grab.GetArray()
            return True, frame

        except Exception as e:
            print(f"⚠️ Basler grab failed: {e}")
            return False, None

        finally:
            if grab is not None:
                grab.Release()


    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # raw grab
        ok, frame = self._grab_raw_once()

        if not ok or frame is None:
            return False, None

        return True, frame

    def showFPS(self) -> str:
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                v = float(self.camera.ResultingFrameRate.GetValue())
                if v > 0:
                    self.fps = v
        except Exception:
            pass
        return str(self.fps)

    def reset(self) -> None:
        return

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception as e:
            print(f"⚠️ StopGrabbing failed: {e}")

        try:
            if self.camera.IsOpen():
                self.camera.Close()
        except Exception as e:
            print(f"⚠️ Camera close failed: {e}")

    def getTemperatur(self, camera) -> str:
        try:
            if hasattr(camera, "BslTemperatureStatus"):
                return str(camera.DeviceTemperature.Value)  # ohne genicam-cast
        except Exception:
            pass
        return str(Exception)

    def try_set(self, cam, name, value) -> bool:
        try:
            node = getattr(cam, name)  # kann "Node not existing" werfen
            node.Value = value  # kann "not writable" werfen
            print(f"✅ {name} = {value}")
            return True
        except Exception as e:
            print(f"⚠️ {name} nicht verfügbar/setzbar: {e}")
            return False

    def try_get(self, cam, name):
        try:
            v = getattr(cam, name).Value
            print(f"ℹ️ {name} = {v}")
            return v
        except (AttributeError, genicam.GenericException) as e:
            print(f"⚠️ {name} nicht lesbar/verfügbar: {e}")
            return None


    def host_fps_tick(self) -> float:
        # im __init__ setzen:
        # self._t0 = time.time(); self._n = 0; self._fps_host = 0.0
        self._n += 1
        now = time.time()
        dt = now - self._t0
        if dt >= 1.0:
            self._fps_host = self._n / dt
            self._n = 0
            self._t0 = now
        return self._fps_host




    def logging(self, cam):
        # nach Open(), vor StartGrabbing()
        cam.ExposureAuto.Value = "Off"
        cam.PixelFormat.Value = "BayerRG8"
        cam.GainAuto.Value = "Off"
        # nach Open(), vor StartGrabbing()
        self.try_set(cam, "ExposureAuto", "Off")
        self.try_set( cam,"GainAuto", "Off")

        self.try_set(cam, "ExposureTime", 4000.0)  # 8 ms (sichtbar)
        # optional etwas Gain, wenn nötig:
        self.try_set(cam, "Gain", 5.0)

        print("ExposureTime:", self.try_get(self.camera, "ExposureTime"))
        print("Gain:", self.try_get(self.camera, "Gain"))
        print("Resulting FPS:", cam.BslResultingAcquisitionFrameRate.Value)
        print("AcqEnable:", self.try_get(cam, "AcquisitionFrameRateEnable"))
        print("AcqFPS:", self.try_get(cam, "AcquisitionFrameRate"))
        print("ResultFPS:",
              self.try_get(cam, "BslResultingAcquisitionFrameRate") or self.try_get(cam, "ResultingFrameRate"))
        print("PixelFormat:", self.try_get(cam, "PixelFormat"))
        print("ROI:", self.try_get(cam, "Width"), self.try_get(cam, "Height"))
        print("ExposureAuto:", self.try_get(cam, "ExposureAuto"))
        print("ExposureTime:", self.try_get(cam, "ExposureTime"))
        print("GainAuto:", self.try_get(cam, "GainAuto"))
        print("Gain:", self.try_get(cam, "Gain"))

        # nach cam.Open()
        print("PixelFormat:", self.try_get(cam, "PixelFormat"))
        print("PixelColorFilter:", self.try_get(cam, "PixelColorFilter"))  # z.B. "BayerRG", "BayerBG", ...




