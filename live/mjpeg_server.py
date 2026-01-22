import time
import threading

import cv2
from flask import Flask, Response
from pypylon import pylon

from LensUndistorter import LensUndistorter

cv2.setUseOptimized(True)

app = Flask(__name__)
undistorter = LensUndistorter("lens_config.json")

lock = threading.Lock()
latest_jpeg = None
latest_id = 0


def camera_loop():
    global latest_jpeg, latest_id

    TARGET_FPS = 25
    JPEG_QUALITY = 65
    frame_interval = 1.0 / TARGET_FPS
    last_frame_ts = 0.0

    print("📷 camera_loop started (with reconnect)")

    while True:
        camera = None
        try:
            devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            if not devices:
                print("⚠️ No camera found. Retrying in 2s...")
                time.sleep(2)
                continue

            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()
            print("✅ Camera opened:", camera.GetDeviceInfo().GetModelName(), "|", camera.PixelFormat.Value)

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            while camera.IsGrabbing():
                grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)

                if grab and grab.GrabSucceeded():
                    img = converter.Convert(grab)
                    frame = img.GetArray()

                    # ✅ Entzerren aktivieren (wenn du willst)
                    frame = undistorter.apply(frame)

                    # ✅ FPS-Limit (stabil)
                    now = time.time()
                    dt = now - last_frame_ts
                    if dt < frame_interval:
                        time.sleep(frame_interval - dt)
                    last_frame_ts = time.time()

                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok:
                        jpg_bytes = buf.tobytes()
                        with lock:
                            latest_jpeg = jpg_bytes
                            latest_id += 1

                if grab is not None:
                    grab.Release()

        except Exception as e:
            print("❌ Camera loop error:", repr(e))

        finally:
            try:
                if camera is not None:
                    try:
                        if camera.IsGrabbing():
                            camera.StopGrabbing()
                    except Exception:
                        pass
                    try:
                        if camera.IsOpen():
                            camera.Close()
                    except Exception:
                        pass
            except Exception:
                pass

        print("🔁 Reconnecting in 2s...")
        time.sleep(2)


@app.route("/")
def index():
    return """
    <html>
      <head><title>Basler MJPEG</title></head>
      <body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh;">
        <img src="/video" />
      </body>
    </html>
    """


@app.route("/video")
def video():
    def gen():
        last_sent = -1
        while True:
            with lock:
                jpg = latest_jpeg
                fid = latest_id

            if jpg is None or fid == last_sent:
                time.sleep(0.01)
                continue

            last_sent = fid
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            # keine fixe 50fps Bremse nötig – wir senden nur neue Frames
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("🚀 Starting MJPEG server on 0.0.0.0:8080")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
