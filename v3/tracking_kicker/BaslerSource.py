from __future__ import annotations
from typing import Optional, Tuple, Dict
import time
import numpy as np
from pypylon import pylon, genicam

from v3.tracking_kicker.model.Camera import Camera


class BaslerSource:
    def __init__(
        self,
        cfg: Camera
    ):
        self.cfg = cfg
        self._closed = False # Dafür da um die Kamera zu stoppen

        # --- Basler öffnen ---
        factory = pylon.TlFactory.GetInstance()
        device = factory.CreateFirstDevice()
        self.camera = pylon.InstantCamera(device)
        self.camera.Open()
        # nach Open(), vor StartGrabbing()
        self.camera.ExposureAuto.Value = "Off"
        self.camera.PixelFormat.Value = "BayerRG8"
        self.camera.GainAuto.Value = "Off"
        # nach Open(), vor StartGrabbing()
        #self.try_set(self.camera, "ExposureAuto", "Off")
        self.try_set(self.camera, "GainAuto", "Off")

        self.try_set(self.camera, "ExposureTime", cfg.camera.exposure_time)  # 8 ms (sichtbar)
        # optional etwas Gain, wenn nötig:
        self.try_set(self.camera, "Gain", cfg.camera.exposure_gain.lue)


        self.camera.Height.Value = cfg.camera.Height
        self.camera.OffsetY.Value = cfg.camera.offset_y

        self.camera.Width.Value = cfg.camera.width
        self.camera.OffsetX.Value = cfg.camera.offset_x

        strategy = pylon.GrabStrategy_LatestImageOnly
        self.camera.StartGrabbing(strategy)
        time.sleep(0.05)
        self.loggingonce(self.camera)

#------------------------------------------------ --------------------------------------------------------------------------------

    def _grab_raw_once(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._closed or not self.camera.IsGrabbing():
            return False, None

        grab = None
        try:
            grab = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab is None:
                return False, None

            if not grab.IsValid():
                grab.Release()
                return False, None

            if not grab.GrabSucceeded():
                grab.Release()
                return False, None

            frame = grab.GetArray().copy()
            grab.Release()
            return True, frame

        except Exception as e:
            print(f"⚠️ Basler grab failed: {e}")
            try:
                if grab is not None:
                    grab.Release()
            except:
                pass
            return False, None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
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



    def logging(self, cam):
        try:
            if hasattr(self.camera, "ResultingFrameRate"):
                v = float(self.camera.ResultingFrameRate.GetValue())
                if v > 0:
                    self.fps = v
                    print("FPS:", v)
        except Exception:
            pass
        print("Temperatur:", {self.getTemperatur(self.camera)})
        return str(self.fps)

    def loggingonce(self, cam):
        print(f"✅ Basler geöffnet: {self.camera.GetDeviceInfo().GetModelName()}")
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



