import cv2
import numpy as np
import json
import math
import os
import datetime

cv2.setUseOptimized(True)

# ================================================================
# VIDEO
# ================================================================
VIDEO_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos\Video2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Konnte Video nicht öffnen!")

video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎥 Video FPS erkannt: {video_fps}")

# ================================================================
# STARTABFRAGE: Verzerrung an / aus ?
# ================================================================
print("\n-------------------------------------")
print("VIDEO-ANALYSE MODUS")
print("1 = Mit Verzerrungskorrektur (empfohlen)")
print("2 = Ohne Verzerrung (Rohvideo)")
print("-------------------------------------")
choice = input("Auswahl (1/2): ").strip()
apply_distortion = (choice == "1")
print(f"➡️ Verzerrung aktiv: {apply_distortion}")

# ================================================================
# LENS CONFIG (optional)
# ================================================================
LENS_CONFIG_FILE = "lens_config.json"
k1, k2, k3 = -0.20, 0.05, 0.00
p1, p2 = 0.00, 0.00
scale = 1.00

if apply_distortion:
    try:
        with open(LENS_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        k1 = cfg.get("k1", k1)
        k2 = cfg.get("k2", k2)
        k3 = cfg.get("k3", k3)
        p1 = cfg.get("p1", p1)
        p2 = cfg.get("p2", p2)
        scale = cfg.get("scale", scale)
        print("📁 Verzerrungsdaten geladen:", cfg)
    except Exception as e:
        print("⚠️ Keine Verzerrungsdaten gefunden – nutze Defaults.", e)

# ================================================================
# FILES
# ================================================================
RODS_CONFIG_FILE = "rods.config"
PLAYERS_COLOR_FILE = "players_color.config"

# ================================================================
# DEIN STANGEN-LAYOUT (links -> rechts)
# ================================================================
ROD_LAYOUT = [
    {"team": "black", "expected": 1},  # 1. Stange
    {"team": "black", "expected": 2},  # 2.
    {"team": "white", "expected": 2},  # 3.
    {"team": "black", "expected": 5},  # 4.
    {"team": "white", "expected": 5},  # 5.
    {"team": "black", "expected": 3},  # 6.
    {"team": "white", "expected": 2},  # 7.
    {"team": "white", "expected": 1},  # 8.
]

# ================================================================
# TUNING (Performance + Robustheit)
# ================================================================
SHOW_MASKS = False

# Spielbereich crop (Ränder/Griffe raus)
PLAY_Y0 = 120
PLAY_Y1_MARGIN = 120

# nur in Streifen um Stangenlinie
STRIPE_HALF_W = 30
ROD_CORE_HALF_W = 18  # breiter ausblenden -> weniger Metall als "Farbe"

# Segment-Filter
MIN_SEG_H = 18
MAX_SEG_H = 240

# Minimum Farbe im Segment (Pixelanzahl)
MIN_COLOR_PIXELS = 450

# NEU: Segment muss "breit genug" sein (Figur ragt seitlich raus)
MIN_PLAYER_W = 22

# Profil-Schwelle: Anteil der Streifenbreite muss "Farbe" sein
PROFILE_FRAC = 0.10

# Morphologie
k3 = np.ones((3, 3), np.uint8)
k5 = np.ones((5, 5), np.uint8)

# ================================================================
# JSON helpers
# ================================================================
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ================================================================
# VERZERRUNG MAP
# ================================================================
map_x = None
map_y = None

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
# RODS CONFIG: 8 Klicks (x) -> speichern als Liste von rods mit team/expected
# ================================================================
def calibrate_rods_clicks(frame_bgr):
    h, w = frame_bgr.shape[:2]
    clicks = []
    win = "ROD KALIBRIERUNG (8 Klicks Stangenmitte)"
    help_text = "8x klicken (links->rechts). [U]=Undo  [R]=Reset  [S]=Speichern  [ESC]=Abbruch"

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(int(x))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = frame_bgr.copy()
        xs = sorted(clicks)

        for i, rx in enumerate(xs):
            cv2.line(vis, (rx, 0), (rx, h), (0, 255, 0), 2)
            cv2.putText(vis, f"{i+1}", (rx + 5, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(vis, help_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(15) & 0xFF

        if key == 27:
            cv2.destroyWindow(win)
            return None

        if key in (ord('u'), ord('U')) and clicks:
            clicks.pop()

        if key in (ord('r'), ord('R')):
            clicks = []

        if key in (ord('s'), ord('S')):
            xs = sorted(clicks)
            if len(xs) < 8:
                print(f"⚠️ Noch nicht genug Klicks: {len(xs)}/8")
                continue
            if len(xs) > 8:
                step = len(xs) / 8
                xs = [xs[int(i * step)] for i in range(8)]
            cv2.destroyWindow(win)
            return [x / w for x in xs]

def save_rods_config(path, x_norm_list, w, h):
    rods = []
    for i, xn in enumerate(x_norm_list):
        layout = ROD_LAYOUT[i] if i < len(ROD_LAYOUT) else {"team": "unknown", "expected": 0}
        rods.append({
            "x_norm": float(xn),
            "team": layout["team"],
            "expected": int(layout["expected"])
        })

    data = {
        "version": 2,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "frame_size": [int(w), int(h)],
        "layout": ROD_LAYOUT,
        "rods": rods
    }
    save_json(path, data)
    print(f"✅ rods.config gespeichert: {path}")

def load_rods_config(path):
    cfg = load_json(path)
    if not cfg:
        return None
    if "rods" in cfg and isinstance(cfg["rods"], list) and len(cfg["rods"]) >= 8:
        return cfg
    if "x_norm" in cfg and isinstance(cfg["x_norm"], list) and len(cfg["x_norm"]) >= 8:
        rods = []
        for i, xn in enumerate(cfg["x_norm"][:8]):
            layout = ROD_LAYOUT[i]
            rods.append({"x_norm": float(xn), "team": layout["team"], "expected": int(layout["expected"])})
        return {"version": 2, "rods": rods}
    return None

def rods_pixels(rods_cfg, w, h):
    rods = []
    for r in rods_cfg["rods"][:8]:
        rx = int(float(r["x_norm"]) * w)
        rods.append({"x": rx, "team": r["team"], "expected": int(r["expected"])})
    rods.sort(key=lambda d: d["x"])
    return rods

# ================================================================
# COLOR CONFIG via ROI (White / Black / BG) in LAB
# Schritt B: sigma clamp + toleranter thr
# ================================================================
def roi_to_model(lab_roi, min_sigma=10.0):
    A = lab_roi.reshape(-1, 3).astype(np.float32)
    if A.shape[0] < 400:
        raise ValueError("ROI zu klein")

    mu = A.mean(axis=0)
    sigma = A.std(axis=0)
    sigma = np.maximum(sigma, float(min_sigma))  # sigma clamp

    d2 = np.sum(((A - mu) / sigma) ** 2, axis=1)
    thr = float(np.percentile(d2, 99.5) * 1.6 + 1.0)  # toleranter
    return mu.tolist(), sigma.tolist(), thr

def calibrate_colors_roi(frame_bgr):
    win = "ROI COLOR KALIBRIERUNG (ENTER bestaetigt, ESC bricht ab)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("\n🟦 ROI wählen: WHITE (weiße Figurflächen) -> ENTER bestätigen")
    r1 = cv2.selectROI(win, frame_bgr, showCrosshair=True, fromCenter=False)
    if r1 == (0, 0, 0, 0):
        cv2.destroyWindow(win)
        return None

    print("⬛ ROI wählen: BLACK (schwarze Figurflächen) -> ENTER bestätigen")
    r2 = cv2.selectROI(win, frame_bgr, showCrosshair=True, fromCenter=False)
    if r2 == (0, 0, 0, 0):
        cv2.destroyWindow(win)
        return None

    print("🟩 ROI wählen: BG (Spielfeld/Hintergrund, keine weißen Linien) -> ENTER bestätigen")
    r3 = cv2.selectROI(win, frame_bgr, showCrosshair=True, fromCenter=False)
    if r3 == (0, 0, 0, 0):
        cv2.destroyWindow(win)
        return None

    cv2.destroyWindow(win)

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    def crop(rect):
        x, y, w, h = rect
        return lab[int(y):int(y+h), int(x):int(x+w)].copy()

    mu_w, sig_w, thr_w = roi_to_model(crop(r1), min_sigma=10.0)
    mu_b, sig_b, thr_b = roi_to_model(crop(r2), min_sigma=14.0)  # black noch toleranter
    mu_bg, sig_bg, thr_bg = roi_to_model(crop(r3), min_sigma=10.0)

    cfg = {
        "version": 3,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "space": "LAB",
        "white": {"mu": mu_w, "sigma": sig_w, "thr": thr_w},
        "black": {"mu": mu_b, "sigma": sig_b, "thr": thr_b},
        "bg":    {"mu": mu_bg, "sigma": sig_bg, "thr": thr_bg},
        "bg_margin": 0.92,
        "note": "ROI-basierte Farbmodelle (nur Farbe) + sigma clamp."
    }
    return cfg

def load_or_calibrate_colors(frame_bgr):
    cfg = load_json(PLAYERS_COLOR_FILE)
    ok = cfg and "white" in cfg and "black" in cfg and "bg" in cfg
    if ok:
        print("📁 players_color.config geladen (ROI/LAB).")
        return cfg

    print("🛠️ players_color.config fehlt/alt -> ROI Farb-Kalibrierung startet.")
    cfg = calibrate_colors_roi(frame_bgr)
    if cfg is None:
        print("⚠️ Farb-Kalibrierung abgebrochen.")
        return None
    save_json(PLAYERS_COLOR_FILE, cfg)
    print(f"✅ players_color.config gespeichert: {PLAYERS_COLOR_FILE}")
    return cfg

# ================================================================
# PLAYER DETECTION (pro Stange nur Teamfarbe + expected count)
# Schritt A: MIN_PLAYER_W Breitenfilter
# ================================================================
def color_mask_for_team(lab_roi_f32, team_model, bg_model, bg_margin):
    mu_t = np.array(team_model["mu"], dtype=np.float32)
    sg_t = np.array(team_model["sigma"], dtype=np.float32)
    thr_t = float(team_model["thr"])

    mu_bg = np.array(bg_model["mu"], dtype=np.float32)
    sg_bg = np.array(bg_model["sigma"], dtype=np.float32)
    thr_bg = float(bg_model["thr"])

    d2_t = np.sum(((lab_roi_f32 - mu_t) / sg_t) ** 2, axis=2)
    d2_bg = np.sum(((lab_roi_f32 - mu_bg) / sg_bg) ** 2, axis=2)

    mask = (d2_t < thr_t) & (d2_t < d2_bg * bg_margin) & (d2_bg < thr_bg * 4.0)
    return (mask.astype(np.uint8) * 255)

def segments_from_profile(mask_u8, frac_thresh, min_h, max_h, y0_global):
    prof = cv2.reduce(mask_u8, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
    width = max(1, mask_u8.shape[1])
    thr = int(width * 255 * frac_thresh)
    on = prof > thr

    segs = []
    i = 0
    n = len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        hseg = j - i
        if min_h <= hseg <= max_h:
            segs.append((y0_global + i, y0_global + j))
        i = j
    return segs

def detect_players(processed_bgr, rods, color_cfg):
    h, w = processed_bgr.shape[:2]
    y0 = PLAY_Y0
    y1 = h - PLAY_Y1_MARGIN
    if y1 <= y0 + 10:
        return [], None

    lab_full = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_margin = float(color_cfg.get("bg_margin", 0.92))
    model_bg = color_cfg["bg"]

    players = []
    debug_mask = np.zeros((h, w), np.uint8) if SHOW_MASKS else None

    for rod in rods:
        rx = rod["x"]
        team = rod["team"]
        expected = int(rod["expected"])
        model_team = color_cfg[team]  # "white" oder "black"

        x0s = max(0, rx - STRIPE_HALF_W)
        x1s = min(w, rx + STRIPE_HALF_W)
        roi_lab = lab_full[y0:y1, x0s:x1s, :]

        mask = color_mask_for_team(roi_lab, model_team, model_bg, bg_margin)

        # Metallkern raus (breit)
        cx = (x1s - x0s) // 2
        c0 = max(0, cx - ROD_CORE_HALF_W)
        c1 = min(x1s - x0s, cx + ROD_CORE_HALF_W)
        mask[:, c0:c1] = 0

        # Morphologie
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)

        segs = segments_from_profile(mask, PROFILE_FRAC, MIN_SEG_H, MAX_SEG_H, y0)

        scored = []
        for (ya, yb) in segs:
            ya2 = ya - y0
            yb2 = yb - y0
            seg = mask[ya2:yb2, :]

            seg_pixels = int(np.count_nonzero(seg))
            if seg_pixels < MIN_COLOR_PIXELS:
                continue

            # NEU: Breitenfilter -> Stange-only Segmente fliegen raus
            cols = np.where(np.any(seg > 0, axis=0))[0]
            if cols.size == 0:
                continue
            seg_w = int(cols[-1] - cols[0] + 1)
            if seg_w < MIN_PLAYER_W:
                continue

            scored.append((seg_pixels, ya, yb))

        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[:expected]
        scored.sort(key=lambda t: (t[1] + t[2]) / 2)

        for seg_pixels, ya, yb in scored:
            players.append({
                "rod_x": rx,
                "team": team,
                "y0": int(ya),
                "y1": int(yb),
                "score": int(seg_pixels),
                "expected": expected
            })

        if SHOW_MASKS and debug_mask is not None:
            debug_mask[y0:y1, x0s:x1s] = cv2.max(debug_mask[y0:y1, x0s:x1s], mask)

    return players, debug_mask

# ================================================================
# BALL DETECTION
# ================================================================
prev_ball = None
smoothed_speed = 0.0

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 80, 110], np.uint8)
    upper = np.array([38, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_circ = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if not (80 < area < 8000):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 4:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        if circ > 0.5 and circ > best_circ:
            best_circ = circ
            best = (int(x), int(y), int(r))
    return best

# ================================================================
# FIRST FRAME -> rods + colors
# ================================================================
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Konnte erstes Frame nicht lesen.")

h0, w0 = first_frame.shape[:2]
if apply_distortion:
    map_x, map_y = compute_map(w0, h0, k1, k2, k3, p1, p2, scale)
    first_processed = cv2.remap(first_frame, map_x, map_y, cv2.INTER_LINEAR)
else:
    first_processed = first_frame.copy()

# rods
rods_cfg = load_rods_config(RODS_CONFIG_FILE)
if rods_cfg is None:
    print("🛠️ rods.config fehlt -> Stangen-Kalibrierung startet.")
    x_norm = calibrate_rods_clicks(first_processed)
    if x_norm is None:
        print("⚠️ Rod-Kalibrierung abgebrochen. Ende.")
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit
    save_rods_config(RODS_CONFIG_FILE, x_norm, first_processed.shape[1], first_processed.shape[0])
    rods_cfg = load_rods_config(RODS_CONFIG_FILE)
else:
    print("📁 rods.config geladen.")

# colors
colors_cfg = load_or_calibrate_colors(first_processed)
if colors_cfg is None:
    print("⚠️ Keine Farbkonfiguration. Ende.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# restart
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================================================================
# MAIN LOOP
# ================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video fertig.")
        break

    h, w = frame.shape[:2]

    if apply_distortion:
        if map_x is None or map_x.shape[:2] != (h, w):
            map_x, map_y = compute_map(w, h, k1, k2, k3, p1, p2, scale)
        processed = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    else:
        processed = frame

    # Ball
    ball = detect_ball(processed)
    if ball:
        x, y, r = ball
        if prev_ball is not None:
            dx = x - prev_ball[0]
            dy = y - prev_ball[1]
            dist = math.hypot(dx, dy)
            inst_speed = dist * video_fps
            smoothed_speed = smoothed_speed * 0.8 + inst_speed * 0.2
        prev_ball = (x, y)
        cv2.circle(processed, (x, y), int(r * 1.6), (0, 255, 255), 3)
        cv2.circle(processed, (x, y), int(r * 0.5), (0, 255, 255), -1)
    else:
        prev_ball = None

    # Rods + Players
    rods = rods_pixels(rods_cfg, w, h)

    for i, rod in enumerate(rods, start=1):
        rx = rod["x"]
        cv2.line(processed, (rx, 0), (rx, h), (0, 255, 0), 2)
        cv2.putText(processed, f"{i}:{rod['team'][0].upper()}/{rod['expected']}",
                    (rx - 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    players, dbg = detect_players(processed, rods, colors_cfg)

    for p in players:
        rx = p["rod_x"]
        ya, yb = p["y0"], p["y1"]
        x0s = max(0, rx - STRIPE_HALF_W)
        x1s = min(w - 1, rx + STRIPE_HALF_W)
        cv2.rectangle(processed, (x0s, ya), (x1s, yb), (0, 0, 255), 2)
        cv2.putText(processed, f"{p['team']} ({p['score']})",
                    (x0s, max(20, ya - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    if SHOW_MASKS and dbg is not None:
        cv2.imshow("TeamMask", dbg)

    # HUD
    cv2.putText(processed, f"Speed: {smoothed_speed:.1f} px/s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(processed, f"FPS: {video_fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(processed, "M: mask | C: recolor ROI | ESC: quit", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Video Analyse", processed)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord('m'), ord('M')):
        SHOW_MASKS = not SHOW_MASKS
        if not SHOW_MASKS:
            try:
                cv2.destroyWindow("TeamMask")
            except:
                pass
    if key in (ord('c'), ord('C')):
        new_cfg = calibrate_colors_roi(processed)
        if new_cfg is not None:
            colors_cfg = new_cfg
            save_json(PLAYERS_COLOR_FILE, colors_cfg)
            print(f"✅ players_color.config aktualisiert: {PLAYERS_COLOR_FILE}")

cap.release()
cv2.destroyAllWindows()
