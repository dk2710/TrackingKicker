import threading
import time

import cv2
from flask import Flask, Response
from pypylon import pylon

app = Flask(__name__)

latest_jpeg = None
lock = threading.Lock()


def camera_loop():
    global latest_jpeg

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Optional: ein paar sinnvolle Defaults (kannst du später anpassen)
    # camera.PixelFormat = "Mono8"  # falls unterstützt
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if grab and grab.GrabSucceeded():
                frame = grab.Array  # numpy array

                # JPEG encoden (Qualität 80 ist meist gut & schnell)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    jpg = buf.tobytes()
                    with lock:
                        latest_jpeg = jpg

            if grab:
                grab.Release()

            # kleine Pause verhindert 100% CPU, optional
            time.sleep(0.001)

    finally:
        try:
            camera.StopGrabbing()
        except Exception:
            pass
        camera.Close()


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

            # Browser braucht keine 200 FPS 😄
            time.sleep(0.02)  # ~50 FPS max

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return """
    <html>
      <head><title>Basler MJPEG</title></head>
      <body style="margin:0; background:#111; display:flex; justify-content:center; align-items:center; height:100vh;">
        <img src="/video" />
      </body>
    </html>
    """


if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    # 0.0.0.0 = von außen erreichbar, Port kannst du ändern
    app.run(host="0.0.0.0", port=8080, threaded=True)
