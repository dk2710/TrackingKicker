from pypylon import pylon
import cv2
import time
import threading

# ... dein Flask Zeug wie gehabt ...

def camera_loop():
    global latest_jpeg

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Converter für "immer BGR" (also Farbe)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if grab and grab.GrabSucceeded():
                # ✅ Basler konvertiert Bayer/Mono intern korrekt zu BGR
                img = converter.Convert(grab)
                frame = img.GetArray()   # jetzt 3-kanalig (H,W,3)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with lock:
                        latest_jpeg = buf.tobytes()

            if grab:
                grab.Release()

            time.sleep(0.001)
    finally:
        try:
            camera.StopGrabbing()
        except Exception:
            pass
        camera.Close()
