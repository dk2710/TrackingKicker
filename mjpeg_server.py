cd ~/dev/tischkicker

cat > TrackingKicker/mjpeg_server.py <<'EOF'
import time
import threading

import cv2
from flask import Flask, Response
from pypylon import pylon

cv2.setUseOptimized(True)

app = Flask(__name__)

lock = threading.Lock()
latest_jpeg = None


def camera_loop():
    global latest_jpeg

    print("📷 camera_loop started")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print("✅ Camera opened:", camera.GetDeviceInfo().GetModelName())
    print("ℹ️ PixelFormat:", camera.PixelFormat.Value)

    # BayerRG8 -> BGR (FARBE)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if grab and grab.GrabSucceeded():
                img = converter.Convert(grab)
                frame = img.GetArray()  # HxWx3 BGR

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with lock:
                        latest_jpeg = buf.tobytes()

            if grab:
                grab.Release()

            time.sleep(0.005)

    finally:
        try:
            camera.StopGrabbing()
        except Exception:
            pass
        camera.Close()
        print("🛑 Camera closed")


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
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.02)  # ~50 FPS max im Browser
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("🚀 Starting MJPEG server on 0.0.0.0:8080")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
EOF
