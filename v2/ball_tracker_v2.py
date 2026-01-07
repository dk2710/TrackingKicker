"""
Kicker Ball Tracker - Verbesserte Version
==========================================

Integration in bestehendes Projekt:
1. Diese Datei ins Projektverzeichnis kopieren
2. Im Hauptskript importieren:
   from ball_tracker_v2 import BallTracker
3. Alte detect_ball() Funktion ersetzen

Änderungen gegenüber Original:
- HSV-Bereich optimiert für gelben Kickerball
- Kalman-Filter für robustes Tracking bei Verdeckung
- Multi-Frame-Validierung gegen False Positives  
- Geschwindigkeitsberechnung mit konfigurierbarer Kalibrierung
- Debug-Visualisierung

Autor: Claude (für Johannes)
Datum: 2025-01
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class TrackingState(Enum):
    """Status des Ball-Trackings"""
    LOST = 0        # Kein Ball, keine Vorhersage
    TRACKING = 1    # Ball aktiv detektiert
    PREDICTED = 2   # Ball verdeckt, Position vorhergesagt
    SEARCHING = 3   # Suche nach Ball nach Verlust


@dataclass
class BallState:
    """Aktueller Zustand des Balls"""
    x: float = 0.0
    y: float = 0.0
    radius: float = 20.0
    vx: float = 0.0  # Geschwindigkeit X (Pixel/Frame)
    vy: float = 0.0  # Geschwindigkeit Y (Pixel/Frame)
    speed_px_s: float = 0.0  # Geglättete Geschwindigkeit (Pixel/Sekunde)
    confidence: float = 0.0
    state: TrackingState = TrackingState.LOST
    frames_since_detection: int = 0


class KalmanTracker:
    """
    Kalman-Filter für Positions- und Geschwindigkeits-Tracking.
    Optimiert für schnelle Ballbewegungen.
    """
    
    def __init__(self, 
                 process_noise: float = 200.0,   # Höher = mehr Vertrauen in Messungen
                 measurement_noise: float = 3.0,  # Niedriger = mehr Vertrauen in Messungen
                 max_predict_frames: int = 15):   # Max Frames ohne Detektion
        
        # Kalman Filter: State = [x, y, vx, vy], Measurement = [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State Transition (konstante Geschwindigkeit)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement Matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Noise Covariances
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        
        self.max_predict_frames = max_predict_frames
        self.initialized = False
        self.predict_count = 0
    
    def reset(self):
        """Reset tracker state"""
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        self.initialized = False
        self.predict_count = 0
    
    def init_position(self, x: float, y: float):
        """Initialisiere mit bekannter Position"""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.predict_count = 0
    
    def predict(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Vorhersage ohne Messung.
        Returns: (x, y, vx, vy) oder None wenn zu lange ohne Update
        """
        if not self.initialized:
            return None
        
        self.predict_count += 1
        if self.predict_count > self.max_predict_frames:
            return None
        
        pred = self.kf.predict()
        return (float(pred[0, 0]), float(pred[1, 0]), float(pred[2, 0]), float(pred[3, 0]))
    
    def update(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """
        Update mit Messung.
        Returns: (x, y, vx, vy) - geglättete Werte
        """
        if not self.initialized:
            self.init_position(x, y)
            return (x, y, 0.0, 0.0)
        
        # Predict + Correct
        self.kf.predict()
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        
        self.predict_count = 0
        
        return (
            float(corrected[0, 0]),
            float(corrected[1, 0]),
            float(corrected[2, 0]),
            float(corrected[3, 0])
        )


class BallDetector:
    """
    HSV-basierte Ball-Detektion mit optimierten Parametern.
    """
    
    def __init__(self):
        # Primäre HSV-Range (aus Analyse: gelber Kickerball)
        # H: 13-19 (erweitert auf 8-28 für Toleranz)
        # S: 139-170 (erweitert auf 70-255)  
        # V: 103+ (erweitert auf 60+)
        self.hsv_ranges = [
            # Primär: Optimiert für den analysierten Ball
            (np.array([8, 80, 70]), np.array([28, 255, 255])),
            # Fallback: Breiterer Bereich für schwierige Lichtverhältnisse
            (np.array([5, 50, 50]), np.array([35, 255, 255])),
        ]
        
        # Morphologie-Kernel
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Größenfilter
        self.min_area = 250
        self.max_area = 8000
        self.min_radius = 8
        self.max_radius = 50
        
        # Formfilter
        self.min_circularity = 0.50
        
        # Debug
        self.last_mask = None
    
    def detect(self, frame: np.ndarray, 
               search_region: Optional[Tuple[int, int, int, int]] = None,
               use_fallback: bool = False) -> List[Tuple[float, float, float, float, int]]:
        """
        Erkenne Ball im Frame.
        
        Args:
            frame: BGR-Bild
            search_region: Optional (x, y, w, h) - eingeschränkte Suche
            use_fallback: Breiteren HSV-Bereich nutzen
            
        Returns:
            Liste von (x, y, radius, confidence, area), sortiert nach confidence
        """
        # ROI extrahieren
        if search_region:
            rx, ry, rw, rh = search_region
            h, w = frame.shape[:2]

            # Grenzen validieren und clampen
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))

            roi = frame[ry:ry+rh, rx:rx+rw]
            offset = (rx, ry)

            # Fallback wenn ROI zu klein oder leer
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                roi = frame
                offset = (0, 0)
        else:
            roi = frame
            offset = (0, 0)
        
        # HSV konvertieren
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # HSV-Range wählen
        hsv_range = self.hsv_ranges[1 if use_fallback else 0]
        
        # Maske erstellen
        mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
        
        # Morphologie
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        
        self.last_mask = mask.copy()
        
        # Konturen finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (self.min_area < area < self.max_area):
                continue
            
            # Minimum Enclosing Circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            if not (self.min_radius < radius < self.max_radius):
                continue
            
            # Zirkularität
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Fill Ratio (wie viel des Kreises ist gefüllt)
            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 0
            
            # Confidence berechnen
            confidence = self._calc_confidence(circularity, fill_ratio, area)
            
            # Position mit Offset
            candidates.append((
                cx + offset[0],
                cy + offset[1],
                radius,
                confidence,
                int(area)
            ))
        
        # Sortiere nach Confidence
        candidates.sort(key=lambda c: c[3], reverse=True)
        
        return candidates
    
    def _calc_confidence(self, circularity: float, fill_ratio: float, area: int) -> float:
        """Confidence-Score berechnen"""
        # Zirkularität: 0.85+ ist ideal
        circ_score = min(circularity / 0.85, 1.0)
        
        # Fill: 0.7+ ist ideal
        fill_score = min(fill_ratio / 0.70, 1.0)
        
        # Größe: ~1500 ist ideal
        ideal_area = 1500
        area_diff = abs(area - ideal_area) / ideal_area
        area_score = max(0, 1.0 - area_diff * 0.5)
        
        return 0.50 * circ_score + 0.30 * fill_score + 0.20 * area_score


class BallTracker:
    """
    Hauptklasse: Kombiniert Detektion + Kalman-Filter.
    
    Verwendung:
        tracker = BallTracker(fps=180)
        
        while video_running:
            frame = get_frame()
            state = tracker.update(frame)
            
            if state.state != TrackingState.LOST:
                draw_ball(state.x, state.y, state.radius)
                print(f"Speed: {state.speed_px_s:.0f} px/s")
    """
    
    def __init__(self, 
                 fps: float = 180.0,
                 pixels_per_meter: float = 800.0,  # Kalibrierung für km/h
                 play_area: Optional[Tuple[int, int, int, int]] = None):  # x, y, w, h
        """
        Args:
            fps: Video-Framerate
            pixels_per_meter: Kalibrierung (z.B. Spielfeldbreite_px / 1.20m)
            play_area: Optional Spielbereich-Einschränkung
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.play_area = play_area
        
        # Komponenten
        self.detector = BallDetector()
        self.kalman = KalmanTracker(
            process_noise=200.0,
            measurement_noise=3.0,
            max_predict_frames=int(fps * 0.08)  # ~80ms Vorhersage
        )
        
        # State
        self.state = BallState()
        
        # Geschwindigkeits-Glättung
        self.speed_history = deque(maxlen=8)
        
        # Validierung
        self.max_jump_distance = 150  # Max Pixel-Sprung zwischen Frames
        self.search_radius = 200  # Suchbereich um letzte Position
    
    def update(self, frame: np.ndarray) -> BallState:
        """
        Verarbeite Frame und update Ball-State.
        
        Returns:
            BallState mit aktueller Position/Geschwindigkeit/Status
        """
        h, w = frame.shape[:2]
        
        # Suchbereich bestimmen
        search_region = None
        if self.state.state in (TrackingState.TRACKING, TrackingState.PREDICTED):
            # Suche um letzte bekannte Position
            cx, cy = int(self.state.x), int(self.state.y)
            r = self.search_radius
            x0 = max(0, cx - r)
            y0 = max(0, cy - r)
            x1 = min(w, cx + r)
            y1 = min(h, cy + r)
            search_region = (x0, y0, x1 - x0, y1 - y0)
        elif self.play_area:
            search_region = self.play_area
        
        # Detektion
        candidates = self.detector.detect(frame, search_region, use_fallback=False)
        
        # Fallback wenn nichts gefunden
        if not candidates:
            candidates = self.detector.detect(frame, search_region, use_fallback=True)
        
        # Kandidaten validieren
        best = self._select_best_candidate(candidates)
        
        if best:
            x, y, radius, confidence, area = best
            
            # Kalman Update
            kx, ky, vx, vy = self.kalman.update(x, y)
            
            # State updaten
            self.state.x = kx
            self.state.y = ky
            self.state.radius = radius
            self.state.vx = vx
            self.state.vy = vy
            self.state.confidence = confidence
            self.state.state = TrackingState.TRACKING
            self.state.frames_since_detection = 0
            
        else:
            # Keine Detektion - Kalman vorhersagen
            pred = self.kalman.predict()
            
            if pred:
                kx, ky, vx, vy = pred
                self.state.x = kx
                self.state.y = ky
                self.state.vx = vx
                self.state.vy = vy
                self.state.confidence *= 0.9  # Confidence sinkt
                self.state.state = TrackingState.PREDICTED
                self.state.frames_since_detection += 1
            else:
                # Tracking verloren
                self.state.state = TrackingState.LOST
                self.state.confidence = 0.0
                self.kalman.reset()
        
        # Geschwindigkeit berechnen
        self._update_speed()
        
        return self.state
    
    def _select_best_candidate(self, 
                               candidates: List[Tuple[float, float, float, float, int]]
                               ) -> Optional[Tuple[float, float, float, float, int]]:
        """Wähle besten Kandidaten mit Validierung"""
        if not candidates:
            return None
        
        # Wenn wir bereits tracken: Sprung validieren
        if self.state.state in (TrackingState.TRACKING, TrackingState.PREDICTED):
            for cand in candidates:
                x, y = cand[0], cand[1]
                dist = np.sqrt((x - self.state.x)**2 + (y - self.state.y)**2)
                
                if dist < self.max_jump_distance:
                    return cand
            
            # Kein Kandidat in akzeptabler Distanz
            return None
        
        # Erstes Tracking: Besten nehmen
        return candidates[0] if candidates else None
    
    def _update_speed(self):
        """Geglättete Geschwindigkeit berechnen"""
        if self.state.state == TrackingState.LOST:
            self.speed_history.clear()
            self.state.speed_px_s = 0.0
            return
        
        # Momentangeschwindigkeit
        speed_frame = np.sqrt(self.state.vx**2 + self.state.vy**2)
        speed_sec = speed_frame * self.fps
        
        self.speed_history.append(speed_sec)
        
        # Geglätteter Wert (Median ist robuster als Mean)
        if self.speed_history:
            self.state.speed_px_s = float(np.median(self.speed_history))
    
    def get_speed_kmh(self) -> float:
        """Geschwindigkeit in km/h (benötigt Kalibrierung)"""
        if self.pixels_per_meter <= 0:
            return 0.0
        m_per_s = self.state.speed_px_s / self.pixels_per_meter
        return m_per_s * 3.6
    
    def reset(self):
        """Tracker zurücksetzen"""
        self.state = BallState()
        self.kalman.reset()
        self.speed_history.clear()
    
    def draw(self, frame: np.ndarray, 
             show_velocity: bool = True,
             show_search_area: bool = False) -> np.ndarray:
        """
        Zeichne Ball-Visualisierung auf Frame.
        
        Returns:
            Frame mit Overlay
        """
        vis = frame.copy()
        
        if self.state.state == TrackingState.LOST:
            return vis
        
        x, y = int(self.state.x), int(self.state.y)
        r = int(self.state.radius)
        
        # Farbe nach Status
        if self.state.state == TrackingState.TRACKING:
            color = (0, 255, 255)  # Gelb
        else:  # PREDICTED
            color = (0, 165, 255)  # Orange
        
        # Äußerer Ring
        cv2.circle(vis, (x, y), int(r * 1.6), color, 3)
        
        # Innerer Punkt
        cv2.circle(vis, (x, y), 5, color, -1)
        
        # Geschwindigkeitsvektor
        if show_velocity and (abs(self.state.vx) > 2 or abs(self.state.vy) > 2):
            scale = 8
            end_x = int(x + self.state.vx * scale)
            end_y = int(y + self.state.vy * scale)
            cv2.arrowedLine(vis, (x, y), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)
        
        # Suchbereich (Debug)
        if show_search_area and self.state.state != TrackingState.LOST:
            sr = self.search_radius
            cv2.rectangle(vis, (x - sr, y - sr), (x + sr, y + sr), (100, 100, 100), 1)
        
        return vis
    
    def get_debug_mask(self) -> Optional[np.ndarray]:
        """Letzte HSV-Maske für Debug"""
        return self.detector.last_mask


# ============================================================
# Drop-in Replacement für alten Code
# ============================================================

# Globaler Tracker (für einfache Integration)
_global_tracker: Optional[BallTracker] = None


def init_ball_tracker(fps: float = 180.0, pixels_per_meter: float = 800.0):
    """Initialisiere globalen Tracker"""
    global _global_tracker
    _global_tracker = BallTracker(fps=fps, pixels_per_meter=pixels_per_meter)


def detect_ball(frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Drop-in Replacement für alte detect_ball() Funktion.
    
    Returns:
        (x, y, radius) oder None
    """
    global _global_tracker
    
    if _global_tracker is None:
        init_ball_tracker()
    
    state = _global_tracker.update(frame)
    
    if state.state != TrackingState.LOST:
        return (int(state.x), int(state.y), int(state.radius))
    return None


def get_ball_speed() -> float:
    """Hole aktuelle Geschwindigkeit in px/s"""
    global _global_tracker
    if _global_tracker is None:
        return 0.0
    return _global_tracker.state.speed_px_s


def get_ball_speed_kmh() -> float:
    """Hole aktuelle Geschwindigkeit in km/h"""
    global _global_tracker
    if _global_tracker is None:
        return 0.0
    return _global_tracker.get_speed_kmh()


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import sys
    import glob
    
    # Test mit Frames
    frames = sorted(glob.glob("/mnt/user-data/uploads/frame_*.png"))
    
    if not frames:
        print("Keine Test-Frames gefunden")
        sys.exit(1)
    
    print(f"=== BallTracker Test ({len(frames)} Frames) ===\n")
    
    tracker = BallTracker(fps=186.6)
    
    results = []
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            continue
        
        state = tracker.update(img)
        name = path.split("/")[-1]
        
        status = state.state.name
        if state.state != TrackingState.LOST:
            print(f"{name}: [{status:10}] pos=({state.x:.0f}, {state.y:.0f}) "
                  f"v=({state.vx:.1f}, {state.vy:.1f}) speed={state.speed_px_s:.0f} px/s "
                  f"conf={state.confidence:.2f}")
        else:
            print(f"{name}: [{status:10}]")
        
        results.append((name, state))
    
    # Visualisierung speichern
    print("\n=== Erstelle Visualisierung ===")
    
    for path in frames[:4]:
        img = cv2.imread(path)
        if img is None:
            continue
        
        tracker.reset()  # Fresh start für jedes Bild
        tracker.update(img)
        
        vis = tracker.draw(img, show_velocity=True, show_search_area=True)
        
        name = path.split("/")[-1]
        out_path = f"/mnt/user-data/uploads/tracked_{name}"
        cv2.imwrite(out_path, vis)
        print(f"  Saved: {out_path}")
    
    print("\n✅ Test abgeschlossen")
