from __future__ import annotations
from typing import Optional, Tuple, Dict


import time

from pypylon import pylon, genicam

from v3.tracking_kicker.model.Camera import Camera
from v3.tracking_kicker.Objects.Frames import Frame


class BaslerSource:
    def __init__(
        self,
        cfg: Camera
    ):
        self.cfg = cfg
        self._read_count = 0
        self._read_failures = 0
        self._read_log_time = time.perf_counter()
        self._last_grab_error = None
        self._last_grab_error_time = 0.0
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

        self.try_set(self.camera, "ExposureTime", cfg.exposure_time)  # 8 ms (sichtbar)
        # optional etwas Gain, wenn nötig:
        self.try_set(self.camera, "Gain", cfg.gain)


        self.camera.Height.Value = cfg.height
        self.camera.OffsetY.Value = cfg.offset_y

        self.camera.Width.Value = cfg.width
        self.camera.OffsetX.Value = cfg.offset_x

        # Die konfigurierte Bildrate bisher nur zu protokollieren reicht
        # nicht: Die Kamera lief dadurch mit ca. 209 FPS. Das kann am
        # USB-Hub zu Payload-Verlusten und Block-ID-Spruengen fuehren.
        # Begrenze die Kamera deshalb explizit auf die konfigurierte Rate.
        if getattr(cfg, "force_fps", 0) and cfg.force_fps > 0:
            self.try_set(self.camera, "AcquisitionFrameRateEnable", True)
            self.try_set(self.camera, "AcquisitionFrameRate", float(cfg.force_fps))

        strategy = pylon.GrabStrategy_LatestImageOnly
        self.camera.StartGrabbing(strategy)
        time.sleep(0.05)
        self.loggingonce(self.camera)

#------------------------------------------------ --------------------------------------------------------------------------------

    def _grab_raw_once(self) -> Tuple[bool, Optional[Frame]]:
        if self._closed:
            return False, None
        if not self.camera.IsGrabbing():
            self._log_grab_error("Kamera grabbt nicht (IsGrabbing=False)")
            return False, None

        grab = None
        try:
            grab = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab is None:
                self._log_grab_error("RetrieveResult liefert None")
                return False, None

            if not grab.IsValid():
                self._log_grab_error(
                    "Grab-Ergebnis ist ungueltig: "
                    f"code={self._grab_error_code(grab)} "
                    f"text={self._grab_error_text(grab)}"
                )
                grab.Release()
                return False, None

            if not grab.GrabSucceeded():
                self._log_grab_error(
                    "Grab fehlgeschlagen: "
                    f"code={self._grab_error_code(grab)} "
                    f"text={self._grab_error_text(grab)}"
                )
                grab.Release()
                return False, None

            objFrame = Frame()
            objFrame.frame = grab.GetArray().copy()
            objFrame.timestamp = time.perf_counter()
            grab.Release()

            return True, objFrame

        except Exception as e:
            print(f"⚠️ Basler grab failed: {e}")
            try:
                if grab is not None:
                    grab.Release()
            except:
                pass
            return False, None

    @staticmethod
    def _grab_error_code(grab) -> str:
        try:
            return str(grab.GetErrorCode())
        except Exception:
            return "unbekannt"

    @staticmethod
    def _grab_error_text(grab) -> str:
        try:
            return str(grab.GetErrorDescription())
        except Exception:
            return "unbekannt"

    def _log_grab_error(self, message: str) -> None:
        now = time.perf_counter()
        if message != self._last_grab_error or now - self._last_grab_error_time >= 1.0:
            print(f"[CAM-ERR] {message}")
            self._last_grab_error = message
            self._last_grab_error_time = now

    def read(self) -> Tuple[bool, Optional[Frame]]:
        ok, objframe = self._grab_raw_once()

        now = time.perf_counter()
        if not ok or objframe is None:
            self._read_failures += 1
        else:
            self._read_count += 1

        if now - self._read_log_time >= 1.0:
            elapsed = now - self._read_log_time
            delivered_fps = self._read_count / elapsed
            print(
                f"[CAM-RAW] geliefert={self._read_count} "
                f"fps={delivered_fps:.1f} fehlgeschlagen={self._read_failures}"
            )
            self._read_count = 0
            self._read_failures = 0
            self._read_log_time = now

        #frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        if not ok or objframe is None:
            return False, None

        return True, objframe

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

        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except BaseException as e:
            print(f"⚠️ StopGrabbing failed: {e}")

        try:
            if self.camera.IsOpen():
                self.camera.Close()
        except BaseException as e:
            print(f"⚠️ Camera close failed: {e}")

        try:
            del self.camera
        except BaseException:
            pass

        self._closed = True

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



