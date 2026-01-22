import time
import threading

import cv2
from flask import Flask, Response
from pypylon import pylon
from LensUndistorter import LensUndistorter

global latest_jpeg, latest_id


cv2.setUseOptimized(True)

app = Flask(__name__)
undistorter = LensUndistorter("lens_config.json")

lock = threading.Lock()
latest_jpeg = None
latest_id = 0


def camera_loop():
    TARGET_FPS = 25
    frame_interval = 1.0 / TARGET_FPS
    last = 0.0

    global latest_id
    with lock:
        latest_jpeg = buf.tobytes()
        latest_id += 1

    print("📷 camera_loop started (with reconnect)")

    while True:
        now = time.time()
        sleep_t = frame_interval - (now - last)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last = time.time()

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

                    # falls du Undistorter nutzt:
                    # frame = undistorter.apply(frame)

                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    if ok:
                        jpg_bytes = buf.tobytes()
                        with lock:
                            latest_jpeg = jpg_bytes
                            latest_id += 1

                _toggle = grab is not None
                if _toggle:
                    grab.Release()

                time.sleep(0.005)

        except Exception as e:
            print("❌ Camera loop error:", repr(e))
            # hier landen auch USB disconnects etc.
        finally:
            try:
                if 'camera' in locals():
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
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    with lock:
        return {
            "has_frame": latest_jpeg is not None,
            "latest_id": latest_id
        }


if __name__ == "__main__":
    print("🚀 Starting MJPEG server on 0.0.0.0:8080")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8080, threaded=True)