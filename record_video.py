# record_video.py
import time
import cv2
from videosource import VideoSource, BaslerSource


# ==============================
# CONFIG
# ==============================
OUTPUT_PATH = r"C:\Users\DKleis\PycharmProjects\TrackingKicker\videos\test2.mp4"
USE_BASLER = True          # False = Video/Webcam
VIDEO_SOURCE = 0           # Webcam oder Pfad bei USE_BASLER=False
SHOW_PREVIEW = True


# ==============================
# Quelle wählen
# ==============================
if USE_BASLER:
    src = BaslerSource(fps_hint=220.0)
else:
    src = VideoSource(VIDEO_SOURCE)

# Erstes Frame holen → Größe bestimmen
ret, frame = src.read()
if not ret or frame is None:
    raise RuntimeError("❌ Kein Frame erhalten")

h, w = frame.shape[:2]

# ==============================
# VideoWriter
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    src.fps,       # FPS der Quelle
    (w, h),
)

print("🎥 Aufnahme gestartet →", OUTPUT_PATH)

# ==============================
# Loop
# ==============================
while True:
    ret, frame = src.read()
    if not ret or frame is None:
        break

    # FPS Overlay
    cv2.putText(
        frame,
        f"FPS: {src.fps:6.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Frame speichern
    writer.write(frame)

    if SHOW_PREVIEW:
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

# ==============================
# Cleanup
# ==============================
writer.release()
src.release()
cv2.destroyAllWindows()

print("✅ Aufnahme beendet")
