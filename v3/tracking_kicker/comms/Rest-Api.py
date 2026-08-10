from fastapi import FastAPI
import base64
import cv2
import time

from v3.tracking_kicker.analysis import KickerAnalyzer
from v3.tracking_kicker.detection import BallDetector

app = FastAPI()

@app.get("/replay-analysis")
def replay_analysis():
    now = time.perf_counter()
    seconds = 3.0

    # Frames der letzten 3 Sekunden holen
    frames = [f for f in replay_buffer if now - f[1] <= seconds]

    # Wenn keine Frames → Fehler
    if len(frames) == 0:
        return {"error": "No frames in replay buffer"}

    # Analyzer neu starten (damit alte Werte nicht stören)
    analyzer = KickerAnalyzer(goals_cfg=vision.goals, field_cfg=vision.field)
    detector = BallDetector(debug=False)

    max_speed = 0.0

    # Analyse aller Frames
    for img, ts in frames:
        fake_frame = SimpleNamespace(
            frame=img.copy(),
            timestamp=ts,
            ball=None
        )

        meas = detector.detect(fake_frame)
        if meas is True:
            analyzer.process_frame(fake_frame)
            max_speed = max(max_speed, analyzer.speed_kmh_s)

    # Jetzt Frames auf 60 FPS sampeln
    target_fps = 60
    dt_target = 1.0 / target_fps

    sampled_frames = []
    last_ts = None

    for img, ts in frames:
        if last_ts is None or ts - last_ts >= dt_target:
            # Bild nach JPEG → Base64
            _, jpeg = cv2.imencode(".jpg", img)
            b64 = base64.b64encode(jpeg).decode("utf-8")
            sampled_frames.append(b64)
            last_ts = ts

    return {
        "max_speed_kmh": round(max_speed, 2),
        "fps": target_fps,
        "frames": sampled_frames
    }
