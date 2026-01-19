import cv2
import numpy as np
import json
import math
import os
import datetime

cv2.setUseOptimized(True)

# ================================================================
# FILES
# ================================================================
VIDEO_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos\Video2.mp4"
CALIBRATION_FILE = "table_calibration.json"

# ================================================================
# VIDEO
# ================================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Konnte Video nicht öffnen!")

video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎥 Video FPS erkannt: {video_fps}")

# ================================================================
# STARTMODUS
# ================================================================
print("\n=====================================")
print("STARTMODUS")
print("1 = Neue Gesamtkalibrierung")
print("2 = Vorhandene Kalibrierung laden")
print("=====================================")
mode = input("Auswahl (1/2): ").strip()
DO_CALIBRATION = (mode == "1")

# ================================================================
# ROD LAYOUT
# ================================================================
ROD_LAYOUT = [
    {"team": "black", "expected": 1},
    {"team": "black", "expected": 2},
    {"team": "white", "expected": 2},
    {"team": "black", "expected": 5},
    {"team": "white", "expected": 5},
    {"team": "black", "expected": 3},
    {"team": "white", "expected": 2},
    {"team": "white", "expected": 1},
]

# ================================================================
# JSON HELPERS
# ================================================================
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ================================================================
# COLOR CALIBRATION (ROI / LAB)
# ================================================================
def roi_to_model(lab_roi, min_sigma=10.0):
    A = lab_roi.reshape(-1, 3).astype(np.float32)
    mu = A.mean(axis=0)
    sigma = np.maximum(A.std(axis=0), min_sigma)
    d2 = np.sum(((A - mu) / sigma) ** 2, axis=1)
    thr = float(np.percentile(d2, 99.5) * 1.6 + 1.0)
    return mu.tolist(), sigma.tolist(), thr

def calibrate_colors_roi(frame):
    win = "FARB-KALIBRIERUNG (ENTER bestätigt)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("🟦 WHITE ROI")
    r1 = cv2.selectROI(win, frame, False)
    print("⬛ BLACK ROI")
    r2 = cv2.selectROI(win, frame, False)
    print("🟩 BG ROI")
    r3 = cv2.selectROI(win, frame, False)
    cv2.destroyWindow(win)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    def crop(r):
        x,y,w,h = r
        return lab[y:y+h, x:x+w]

    mu_w, sg_w, thr_w = roi_to_model(crop(r1))
    mu_b, sg_b, thr_b = roi_to_model(crop(r2), min_sigma=14)
    mu_bg, sg_bg, thr_bg = roi_to_model(crop(r3))

    return {
        "white": {"mu": mu_w, "sigma": sg_w, "thr": thr_w},
        "black": {"mu": mu_b, "sigma": sg_b, "thr": thr_b},
        "bg": {"mu": mu_bg, "sigma": sg_bg, "thr": thr_bg},
        "bg_margin": 0.92
    }

# ================================================================
# ROD CALIBRATION
# ================================================================
def calibrate_rods(frame):
    h, w = frame.shape[:2]
    clicks = []

    def on_mouse(e,x,y,f,p):
        if e == cv2.EVENT_LBUTTONDOWN:
            clicks.append(x)

    win = "STANGEN (8 Klicks links->rechts)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = frame.copy()
        for i,x in enumerate(sorted(clicks)):
            cv2.line(vis,(x,0),(x,h),(0,255,0),2)
            cv2.putText(vis,str(i+1),(x+5,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow(win,vis)
        k=cv2.waitKey(15)&0xFF
        if k==27: return None
        if k==ord('s') and len(clicks)>=8:
            cv2.destroyWindow(win)
            xs=sorted(clicks)[:8]
            return [x/w for x in xs]

# ================================================================
# GOAL CALIBRATION (2D ROI)
# ================================================================
def calibrate_goals(frame):
    h,w = frame.shape[:2]
    clicks=[]

    def on_mouse(e,x,y,f,p):
        if e==cv2.EVENT_LBUTTONDOWN:
            clicks.append((x,y))

    win="TOR-KALIBRIERUNG"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win,on_mouse)

    while True:
        vis=frame.copy()
        if len(clicks)>=2:
            cv2.rectangle(vis,clicks[0],clicks[1],(0,0,255),2)
        if len(clicks)>=4:
            cv2.rectangle(vis,clicks[2],clicks[3],(255,0,0),2)
        cv2.imshow(win,vis)
        k=cv2.waitKey(15)&0xFF
        if k==27: return None
        if k==ord('s') and len(clicks)==4:
            cv2.destroyWindow(win)
            (x0,y0),(x1,y1),(x2,y2),(x3,y3)=clicks
            return {
                "left":{
                    "x0":min(x0,x1)/w,"y0":min(y0,y1)/h,
                    "x1":max(x0,x1)/w,"y1":max(y0,y1)/h},
                "right":{
                    "x0":min(x2,x3)/w,"y0":min(y2,y3)/h,
                    "x1":max(x2,x3)/w,"y1":max(y2,y3)/h}
            }

# ================================================================
# BALL DETECTION
# ================================================================
prev_ball=None
smoothed_speed=0

def detect_ball(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(18,80,110),(38,255,255))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best=None
    for c in cnts:
        a=cv2.contourArea(c)
        if 80<a<8000:
            (x,y),r=cv2.minEnclosingCircle(c)
            best=(int(x),int(y),int(r))
    return best

# ================================================================
# FIRST FRAME
# ================================================================
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Kein Frame")

# ================================================================
# CALIBRATION OR LOAD
# ================================================================
if DO_CALIBRATION:
    colors_cfg = calibrate_colors_roi(first_frame)
    rods_norm = calibrate_rods(first_frame)
    goals_cfg = calibrate_goals(first_frame)

    rods_cfg = [
        {"x_norm": rods_norm[i], **ROD_LAYOUT[i]}
        for i in range(8)
    ]

    full_cfg = {
        "created": datetime.datetime.now().isoformat(),
        "colors": colors_cfg,
        "rods": rods_cfg,
        "goals": goals_cfg
    }

    save_json(CALIBRATION_FILE, full_cfg)
    print("✅ Kalibrierung gespeichert")

else:
    cfg = load_json(CALIBRATION_FILE)
    colors_cfg = cfg["colors"]
    rods_cfg = cfg["rods"]
    goals_cfg = cfg["goals"]
    print("📁 Kalibrierung geladen")

# ================================================================
# MAIN LOOP
# ================================================================
cap.set(cv2.CAP_PROP_POS_FRAMES,0)

while True:
    ret,frame=cap.read()
    if not ret: break
    h,w=frame.shape[:2]

    ball=detect_ball(frame)
    if ball:
        x,y,r=ball
        if prev_ball:
            dx=x-prev_ball[0]
            dy=y-prev_ball[1]
            smoothed_speed=0.8*smoothed_speed+0.2*math.hypot(dx,dy)*video_fps
        prev_ball=(x,y)
        cv2.circle(frame,(x,y),r,(0,255,255),3)

        xn,yn=x/w,y/h
        gL,gR=goals_cfg["left"],goals_cfg["right"]

        if gL["x0"]<=xn<=gL["x1"] and gL["y0"]<=yn<=gL["y1"]:
            cv2.putText(frame,"GOAL LEFT",(w//2-120,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)

        if gR["x0"]<=xn<=gR["x1"] and gR["y0"]<=yn<=gR["y1"]:
            cv2.putText(frame,"GOAL RIGHT",(w//2-140,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),4)

    cv2.putText(frame,f"Speed: {smoothed_speed:.1f}px/s",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("Kicker Analyse",frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
