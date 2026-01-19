"""
Kicker Tracking Test
Testet Ball-Tracking und Torerkennung mit Video
"""

import cv2
from ball_tracker_v2 import BallTracker
from goal_detector import GoalDetector

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = r"C:\\Users\\DKleis\\PycharmProjects\\TrackingKicker\\videos\\Video2.mp4"

FPS = 186.6  # Aus deinem Original-Code

# === INITIALISIERUNG ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Konnte Video nicht öffnen: {VIDEO_PATH}")

# Erste Frame-Größe ermitteln
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Konnte ersten Frame nicht lesen")
h, w = first_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Zurück zum Anfang

print(f"Video: {w}x{h} @ {FPS} FPS")

# Tracker initialisieren
ball_tracker = BallTracker(fps=FPS)
goal_detector = GoalDetector(frame_width=w, frame_height=h, fps=FPS)

# Versuche Config zu laden, sonst Standard-Zonen
try:
    goal_detector.load_config("goal_config.json")
except:
    print("Keine goal_config.json gefunden - nutze Standard-Zonen")

# === HAUPTSCHLEIFE ===
frame_count = 0
last_goal = None
goal_anim_frame = 0
paused = False
frame = first_frame.copy()  # Starte mit erstem Frame

print("\nSteuerung:")
print("  SPACE = Pause/Play")
print("  M = Ball-Maske anzeigen")
print("  G = Torzonen anzeigen/verstecken")
print("  R = Score zurücksetzen")
print("  S = Screenshot speichern")
print("  ESC = Beenden")

show_mask = False
show_goals = True

while True:
    if not paused:
        ret, new_frame = cap.read()
        if not ret:
            print("\nVideo fertig. Drücke ESC zum Beenden oder SPACE zum Neustart.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ball_tracker.reset()
            paused = True
            # frame bleibt das letzte gültige Frame
        else:
            frame = new_frame
            frame_count += 1
            # Ball tracken
            ball_state = ball_tracker.update(frame)
            # Tor prüfen
            goal_event = goal_detector.update_from_tracker(ball_state)
            if goal_event:
                print(f"\n⚽ TOR! Frame {frame_count}")
                print(f"   {goal_event.scoring_team.value.upper()} trifft!")
                print(f"   Speed: {goal_event.ball_speed:.0f} px/s")
                print(f"   Score: White {goal_detector.score_white} : {goal_detector.score_black} Black")
                last_goal = goal_event
                goal_anim_frame = 0
    # === VISUALISIERUNG ===
    vis = frame.copy()
    # Ball zeichnen
    vis = ball_tracker.draw(vis, show_velocity=True)
    # Torzonen + Score
    if show_goals:
        vis = goal_detector.draw(vis, show_zones=True, show_score=True)
    # Tor-Animation
    if last_goal and goal_anim_frame < 60:
        vis = goal_detector.draw_goal_animation(vis, last_goal, goal_anim_frame)
        if not paused:
            goal_anim_frame += 1
    # Status-Anzeige
    status = "PAUSED" if paused else "PLAYING"
    cv2.putText(vis, f"Frame: {frame_count} | {status}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Ball-Info
    if ball_state.state.name != "LOST":
        info = f"Ball: ({ball_state.x:.0f}, {ball_state.y:.0f}) | Speed: {ball_state.speed_px_s:.0f} px/s"
        cv2.putText(vis, info, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Anzeigen
    cv2.imshow("Kicker Tracking Test", vis)
    # Maske anzeigen
    if show_mask:
        mask = ball_tracker.get_debug_mask()
        if mask is not None:
            cv2.imshow("Ball Mask", mask)
    # === TASTATUREINGABE ===
    key = cv2.waitKey(1 if not paused else 30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE
        paused = not paused
    elif key == ord('m') or key == ord('M'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("Ball Mask")
    elif key == ord('g') or key == ord('G'):
        show_goals = not show_goals
    elif key == ord('r') or key == ord('R'):
        goal_detector.reset_score()
        print("Score zurückgesetzt")
    elif key == ord('s') or key == ord('S'):
        filename = f"screenshot_{frame_count:06d}.png"
        cv2.imwrite(filename, vis)
        print(f"Screenshot gespeichert: {filename}")

# === AUFRÄUMEN ===
cap.release()
cv2.destroyAllWindows()

print(f"\nEndstand: White {goal_detector.score_white} : {goal_detector.score_black} Black")
print(f"Frames verarbeitet: {frame_count}")