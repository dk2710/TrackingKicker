import cv2
import numpy as np
import math
import json
import datetime
import os
from pypylon import pylon

cv2.setUseOptimized(True)

# ================================================================
# CONFIG-FILE
# ================================================================
CONFIG_FILE = "lens_config.json"

k1 = -0.20
k2 = 0.05
k3 = 0.00
p1 = 0.00
p2 = 0.00
scale = 1.00

# ------------------------------------------------------------
# CONFIG LADEN
# ------------------------------------------------------------
try:
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)
        k1 = cfg["k1"]
        k2 = cfg["k2"]
        k3 = cfg["k3"]
        p1 = cfg["p1"]
        p2 = cfg["p2"]
        scale = cfg["scale"]
        print("📁 Verzerrung geladen:", cfg)
except:
    print("⚠️ Keine gespeicherten Verzerrungsdaten gefunden – nutze Default-Werte.")

# ================================================================
# BASLER-KAMERA
# ================================================================
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.ExposureAuto.SetValue("Off")
camera.GainAuto.SetValue("Off")
camera.ExposureTime.SetValue(5000)
camera.Gain.SetValue(10)

try:
    camera.Gamma.SetValue(1.5)
except:
    print("⚠️ Gamma nicht verfügbar.")
camera.Height.Value = 736
camera.OffsetY.Value = 200

camera.Width.Value = 1100
camera.OffsetX.Value = 180
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


# ================================================================
# WICHTIG: ECHTE KAMERA-FPS FUNKTION
# ================================================================
def get_camera_fps():
    """Gibt echte FPS zurück. Funktioniert sicher bei deiner Kamera."""
    try:
        return camera.ResultingFrameRate.GetValue()
    except:
        try:
            return camera.AcquisitionFrameRate.GetValue()
        except:
            return 200.0  # Fallback


# ================================================================
# BALLTRACKING
# ================================================================
prev_ball = None
prev_radius = None
ball_miss_frames = 0
smoothed_speed = 0.0
balltracking_enabled = True


def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 80, 110], np.uint8)
    upper = np.array([38, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circ = 0

    for c in contours:
        area = cv2.contourArea(c)
        if not (80 < area < 8000):
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 4:
            continue

        per = cv2.arcLength(c, True)
        if per == 0:
            continue

        circ = 4 * np.pi * area / (per * per)
        if circ > 0.5 and circ > best_circ:
            best_circ = circ
            best = (int(x), int(y), int(r))

    return best


# ================================================================
# DISTORTION MAP CACHE
# ================================================================
map_x = None
map_y = None
last_params = None


def compute_map(w, h, k1, k2, k3, p1, p2, scale):
    cx, cy = w / 2, h / 2
    ys, xs = np.indices((h, w), np.float32)
    x = (xs - cx) / cx
    y = (ys - cy) / cy

    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2

    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x_t = 2*p1*x*y + p2*(r2 + 2*x*x)
    y_t = p1*(r2 + 2*y*y) + 2*p2*x*y

    x2 = (x * radial + x_t) * scale
    y2 = (y * radial + y_t) * scale

    mx = (x2 * cx + cx).astype(np.float32)
    my = (y2 * cy + cy).astype(np.float32)
    return mx, my


# ================================================================
# HOMOGRAPHIE
# ================================================================
clicked = []
H = None
TOP_W = 2000
TOP_H = 1000

POINT_NAMES = [
    "oben links", "oben mitte", "oben rechts",
    "rechts mitte",
    "unten rechts", "unten mitte", "unten links",
    "links mitte"
]


def target_pts():
    return np.array([
        [0, 0],
        [TOP_W / 2, 0],
        [TOP_W, 0],
        [TOP_W, TOP_H / 2],
        [TOP_W, TOP_H],
        [TOP_W / 2, TOP_H],
        [0, TOP_H],
        [0, TOP_H / 2],
    ], np.float32)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 8:
        print(f"Punkt {len(clicked)+1}/8 ({POINT_NAMES[len(clicked)]}) = {x},{y}")
        clicked.append([x, y])


def compute_H(src):
    src = np.array(src, np.float32)
    return cv2.findHomography(src, target_pts(), cv2.RANSAC)[0]


# ================================================================
# SLIDER
# ================================================================
sliders_on = False
apply_correction = True


def init_sliders():
    cv2.namedWindow("Verzerrung", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verzerrung", 600, 250)


    print("🔧 Slider geöffnet – 'S' speichert")


def read_sliders():
    kk1 = (cv2.getTrackbarPos("k1 x1000", "Verzerrung") - 2000) / 1000
    kk2 = (cv2.getTrackbarPos("k2 x1000", "Verzerrung") - 2000) / 1000
    kk3 = (cv2.getTrackbarPos("k3 x1000", "Verzerrung") - 2000) / 1000
    pp1 = (cv2.getTrackbarPos("p1 x1000", "Verzerrung") - 2000) / 1000
    pp2 = (cv2.getTrackbarPos("p2 x1000", "Verzerrung") - 2000) / 1000
    sc = cv2.getTrackbarPos("scale x100", "Verzerrung") / 100
    return kk1, kk2, kk3, pp1, pp2, sc


def save_config():
    data = {
        "k1": k1, "k2": k2, "k3": k3,
        "p1": p1, "p2": p2,
        "scale": scale
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print("💾 Verzerrung gespeichert:", data)


# ================================================================
# VIDEO
# ================================================================
recording = False
video_writer = None
os.makedirs("videos", exist_ok=True)

cv2.namedWindow("Basler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Basler", 1920, 1080)
cv2.setMouseCallback("Basler", click_event)

# ================================================================
# MAIN LOOP
# ================================================================
while True:

    # Live FPS der KAMERA (nicht Software!)
    cam_fps = get_camera_fps()

    grab = camera.RetrieveResult(5000)
    if not grab.GrabSucceeded():
        continue

    frame = converter.Convert(grab).GetArray()
    grab.Release()

    h, w = frame.shape[:2]

    params = (k1, k2, k3, p1, p2, scale)
    if sliders_on:
        params = read_sliders()

    if params != last_params or map_x is None:
        map_x, map_y = compute_map(w, h, *params)
        last_params = params
        k1, k2, k3, p1, p2, scale = params

    corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    if H is not None:
        processed = cv2.warpPerspective(corrected, H, (TOP_W, TOP_H))
    else:
        processed = corrected

    # ============================================================
    # BALLTRACKING + Geschwindigkeit
    # ============================================================
    if balltracking_enabled:
        ball = detect_ball(processed)
    else:
        ball = None

    if ball:
        x, y, r = ball

        if prev_ball:
            dx = x - prev_ball[0]
            dy = y - prev_ball[1]
            dist = math.hypot(dx, dy)

            inst_speed = dist * cam_fps
            smoothed_speed = smoothed_speed * 0.8 + inst_speed * 0.2

        prev_ball = (x, y)

        cv2.circle(processed, (x, y), int(r * 1.6), (0, 255, 255), 3)
        cv2.circle(processed, (x, y), int(r * 0.5), (0, 255, 255), -1)
    else:
        prev_ball = None

    # TEXT: Speed + Kamera-FPS
    cv2.putText(processed, f"Speed: {smoothed_speed:.1f} px/s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.putText(processed, f"FPS: {cam_fps:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    # LETTERBOX
    disp = np.zeros((1080, 1920, 3), np.uint8)
    factor = min(1920 / processed.shape[1], 1080 / processed.shape[0])
    nw = int(processed.shape[1] * factor)
    nh = int(processed.shape[0] * factor)
    resized = cv2.resize(processed, (nw, nh))

    xo = (1920 - nw) // 2
    yo = (1080 - nh) // 2
    disp[yo:yo+nh, xo:xo+nw] = resized

    cv2.circle(disp, (960, 540), 6, (0, 0, 255), -1)

    cv2.imshow("Basler", disp)

    # RECORDING
    if recording and video_writer:
        video_writer.write(disp)

    # TASTEN
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord("v") and not sliders_on:
        init_sliders()
        sliders_on = True

    if key == ord("s") and sliders_on:
        save_config()
        cv2.destroyWindow("Verzerrung")

        sliders_on = False

    if key == ord("\r") and len(clicked) == 8:
        print("🔧 Berechne Homographie…")
        H = compute_H(clicked)
        clicked = []

    if key == ord("k"):
        if not recording:
            fps_rec = cam_fps
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"videos/record_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(filename, fourcc, fps_rec, (1920, 1080))

            recording = True
            balltracking_enabled = False
            print(f"🔴 Aufnahme gestartet ({fps_rec:.1f} FPS): {filename}")

        else:
            recording = False
            balltracking_enabled = True
            if video_writer:
                video_writer.release()
            print("⚪ Aufnahme gestoppt.")

camera.StopGrabbing()
cv2.destroyAllWindows()