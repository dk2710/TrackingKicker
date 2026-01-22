import os
import json
import time
import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, send_file, request
from pypylon import pylon

cv2.setUseOptimized(True)

# ================================================================
# CONFIG
# ================================================================
CONFIG_FILE = "live/lens_config.json"

# Kamera/Stream Settings
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080
JPEG_QUALITY = 80

# Stream FPS limit (Browser braucht keine 200 FPS)
STREAM_MAX_FPS = 40.0  # 0 = unlimited

# Ball Detection HSV (wie bei dir)
HSV_LOW = (18, 80, 110)
HSV_HIGH = (38, 255, 255)
AREA_MIN = 80.0
AREA_MAX = 8000.0

# ================================================================
# Flask
# ================================================================
app = Flask(__name__)

lock = threading.Lock()
latest_jpeg_processed: Optional[bytes] = None
latest_jpeg_raw: Optional[bytes] = None
latest_snapshot_path = "/tmp/tischkicker_snapshot.jpg"

# ================================================================
# Verzerrungsparameter (Default + Laden aus JSON)
# ================================================================
k1, k2, k3, p1, p2, scale = -0.20, 0.05, 0.00, 0.00, 0.00, 1.00

def load_lens_config():
    global k1, k2, k3, p1, p2, scale
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        k1 = float(cfg.get("k1", k1))
        k2 = float(cfg.get("k2", k2))
        k3 = float(cfg.get("k3", k3))
        p1 = float(cfg.get("p1", p1))
        p2 = float(cfg.get("p2", p2))
        scale = float(cfg.get("scale", scale))
        print("📁 Verzerrung geladen:", cfg)
    except Exception as e:
        print("⚠️ Keine gespeicherten Verzerrungsdaten gefunden – nutze Default-Werte.", e)

# ================================================================
# Distortion map cache
# ================================================================
map_x: Optional[np.ndarray] = None
map_y: Optional[np.ndarray] = None
last_params: Optional[Tuple[float, float, float, float, float, float, int, int]] = None

def compute_map(w, h, k1_, k2_, k3_, p1_, p2_, scale_):
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.indices((h, w), np.float32)
    x = (xs - cx) / cx
    y = (ys - cy) / cy

    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2

    radial = 1 + k1_*r2 + k2_*r4 + k3_*r6
    x_t = 2*p1_*x*y + p2_*(r2 + 2*x*x)
    y_t = p1_*(r2 + 2*y*y) + 2*p2_*x*y

    x2 = (x * radial + x_t) * scale_
    y2 = (y * radial + y_t) * scale_

    mx = (x2 * cx + cx).astype(np.float32)
    my = (y2 * cy + cy).astype(np.float32)
    return mx, my

# ================================================================
# Ball detection
# ================================================================
def detect_ball(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_LOW, np.uint8)
    upper = np.array(HSV_HIGH, np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circ = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if not (AREA_MIN < area < AREA_MAX):
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 4:
            continue

        per = cv2.arcLength(c, True)
        if per <= 0:
            continue

        circ = 4.0 * np.pi * area / (per * per)
        if circ > 0.5 and circ > best_circ:
            best_circ = circ
            best = (int(x), int(y), int(r))

    return best

# ================================================================
# Kamera-FPS (wie bei dir)
# ================================================================
def get_camera_fps(camera: pylon.InstantCamera) -> float:
    try:
        return float(camera.ResultingFrameRate.GetValue())
    except Exception:
        try:
            return float(camera.AcquisitionFrameRate.GetValue())
        except Exception:
            return 200.0

# ================================================================
# Hauptthread: Kamera grabben -> raw + processed JPEG aktualisieren
# ================================================================
def camera_loop():
    global latest_jpeg_processed, latest_jpeg_raw
    global map_x, map_y, last_params
    global latest_snapshot_path

    load_lens_config()

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # ---- Kamera-Settings (aus deinem Script) ----
    camera.ExposureAuto.SetValue("Off")
    camera.GainAuto.SetValue("Off")
    camera.ExposureTime.SetValue(5000)
    camera.Gain.SetValue(10)
    try:
        camera.Gamma.SetValue(1.5)
    except Exception:
        print("⚠️ Gamma nicht verfügbar.")

    # ROI (wie bei dir)
    try:
        camera.Height.Value = 736
        camera.OffsetY.Value = 200
        camera.Width.Value = 1100
        camera.OffsetX.Value = 180
    except Exception as e:
        print("⚠️ ROI konnte nicht gesetzt werden:", e)

    # ---- Converter: BayerRG8 -> BGR8packed (FARBE!) ----
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    prev_ball = None
    smoothed_speed = 0.0

    last_stream_ts = 0.0
    min_dt = (1.0 / STREAM_MAX_FPS) if STREAM_MAX_FPS and STREAM_MAX_FPS > 0 else 0.0

    print("✅ Kamera-Thread läuft (Stream ready).")

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if not grab or not grab.GrabSucceeded():
                if grab:
                    grab.Release()
                continue

            # Raw Bayer -> BGR
            img = converter.Convert(grab)
            frame = img.GetArray()  # HxWx3 BGR
            grab.Release()

            cam_fps = get_camera_fps(camera)

            # RAW JPEG (nur farbiges Rohbild ohne Korrektur/Overlay)
            ok_raw, buf_raw = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok_raw:
                with lock:
                    latest_jpeg_raw = buf_raw.tobytes()

            # Optional: Stream-Rate limit (damit CPU nicht sinnlos ballert)
            now = time.time()
            if min_dt > 0 and (now - last_stream_ts) < min_dt:
                continue
            last_stream_ts = now

            # Verzerrungskorrektur (cache)
            h, w = frame.shape[:2]
            params = (k1, k2, k3, p1, p2, scale, w, h)
            if params != last_params or map_x is None or map_y is None:
                map_x, map_y = compute_map(w, h, k1, k2, k3, p1, p2, scale)
                last_params = params

            corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

            # Ball detection + speed
            ball = detect_ball(corrected)
            if ball:
                x, y, r = ball
                if prev_ball:
                    dx = x - prev_ball[0]
                    dy = y - prev_ball[1]
                    dist = math.hypot(dx, dy)
                    inst_speed = dist * cam_fps
                    smoothed_speed = smoothed_speed * 0.8 + inst_speed * 0.2
                prev_ball = (x, y)

                cv2.circle(corrected, (x, y), int(r * 1.6), (0, 255, 255), 3)
                cv2.circle(corrected, (x, y), int(r * 0.5), (0, 255, 255), -1)
            else:
                prev_ball = None

            # Overlay text
            cv2.putText(corrected, f"Speed: {smoothed_speed:.1f} px/s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(corrected, f"FPS (cam): {cam_fps:.1f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Processed JPEG
            ok, buf = cv2.imencode(".jpg", corrected, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                jpg = buf.tobytes()
                with lock:
                    latest_jpeg_processed = jpg
                # optional snapshot file (für /snapshot.jpg)
                try:
                    with open(latest_snapshot_path, "wb") as f:
                        f.write(jpg)
                except Exception:
                    pass

    finally:
        try:
            camera.StopGrabbing()
        except Exception:
            pass
        camera.Close()
        print("🛑 Kamera-Thread gestoppt.")

# ================================================================
# Flask Endpoints
# ================================================================
@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Tischkicker Basler Stream</title>
        <style>
          body { margin:0; background:#111; color:#eee; font-family:Arial; }
          .wrap { display:flex; flex-direction:column; align-items:center; padding:16px; gap:12px; }
          img { max-width: 98vw; max-height: 90vh; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,.45); }
          a { color:#8bd; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <h2>Basler MJPEG (processed)</h2>
          <img src="/video" />
          <div>
            <a href="/raw" target="_blank">raw stream</a> |
            <a href="/snapshot.jpg" target="_blank">snapshot</a>
          </div>
        </div>
      </body>
    </html>
    """

def mjpeg_generator(which: str):
    boundary = b"--frame\r\n"
    header = b"Content-Type: image/jpeg\r\n\r\n"
    while True:
        with lock:
            jpg = latest_jpeg_processed if which == "processed" else latest_jpeg_raw
        if jpg is None:
            time.sleep(0.05)
            continue
        yield boundary + header + jpg + b"\r\n"
        # kleine Pause, damit Browser/CPU nicht unnötig glüht
        time.sleep(0.005)

@app.route("/video")
def video():
    return Response(mjpeg_generator("processed"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/raw")
def raw():
    return Response(mjpeg_generator("raw"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot.jpg")
def snapshot():
    # Falls Datei nicht existiert, versuche aus RAM
    if os.path.exists(latest_snapshot_path):
        return send_file(latest_snapshot_path, mimetype="image/jpeg")
    with lock:
        jpg = latest_jpeg_processed
    if jpg is None:
        return ("No snapshot yet", 404)
    return Response(jpg, mimetype="image/jpeg")

@app.route("/reload_config")
def reload_config():
    # bequem: lens_config.json neu laden ohne Neustart
    load_lens_config()
    return {"ok": True, "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2, "scale": scale}

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print(f"🌐 HTTP Stream: http://0.0.0.0:{HTTP_PORT} (von außen: http://<PI-IP>:{HTTP_PORT})")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True)
