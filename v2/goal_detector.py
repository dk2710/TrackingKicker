"""
Kicker Torerkennung
===================

Erkennt Tore basierend auf:
1. Ball verlässt Spielfeld durch Torzone
2. Ball hat Mindestgeschwindigkeit Richtung Tor
3. Ball bleibt für mehrere Frames verschwunden

Konfiguration:
- Torzonen werden beim Start kalibriert oder aus Config geladen
- Links = Team "White" Tor (White kassiert)
- Rechts = Team "Black" Tor (Black kassiert)

Integration:
    from goal_detector import GoalDetector
    
    goal_detector = GoalDetector(frame_width=1920, frame_height=1080)
    # oder: goal_detector.load_config("goal_config.json")
    
    while running:
        state = ball_tracker.update(frame)
        goal_event = goal_detector.update(state)
        
        if goal_event:
            print(f"TOR! {goal_event.scoring_team} hat getroffen!")

Autor: Claude (für Johannes)
Datum: 2025-01
"""

import cv2
import numpy as np
import json
import os
import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
from collections import deque
from enum import Enum


class Team(Enum):
    """Team-Identifikation"""
    WHITE = "white"
    BLACK = "black"
    NONE = "none"


@dataclass
class GoalZone:
    """Definition einer Torzone"""
    x_min: int      # Linke Grenze
    x_max: int      # Rechte Grenze
    y_min: int      # Obere Grenze
    y_max: int      # Untere Grenze
    team: Team      # Welches Team KASSIERT hier (Tor gehört dem Gegner)
    
    def contains(self, x: float, y: float) -> bool:
        """Prüfe ob Punkt in Zone liegt"""
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def distance_to(self, x: float, y: float) -> float:
        """Distanz zum Zonenzentrum"""
        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        return np.sqrt((x - cx)**2 + (y - cy)**2)


@dataclass
class GoalEvent:
    """Ein erkanntes Tor"""
    timestamp: float            # Zeit in Sekunden (seit Start)
    frame_number: int           # Frame-Nummer
    scoring_team: Team          # Team das GETROFFEN hat
    conceding_team: Team        # Team das KASSIERT hat
    ball_entry_pos: Tuple[float, float]  # Letzte Ball-Position vor Verschwinden
    ball_velocity: Tuple[float, float]   # Geschwindigkeit beim Eintritt
    ball_speed: float           # Geschwindigkeit in px/s
    confidence: float           # Sicherheit der Erkennung (0-1)
    goal_side: str              # "left" oder "right"


@dataclass 
class BallHistory:
    """Historische Ball-Daten für Analyse"""
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    frame: int
    tracked: bool  # True wenn detektiert, False wenn predicted


class GoalDetector:
    """
    Hauptklasse für Torerkennung.
    
    Logik:
    1. Überwache Ball-Position kontinuierlich
    2. Wenn Ball in Torzone UND dann verschwindet → potentielles Tor
    3. Validiere: Geschwindigkeit muss Richtung Tor zeigen
    4. Wenn Ball für N Frames weg bleibt → Tor bestätigt
    """
    
    # Standard-Konfiguration (wird überschrieben durch Kalibrierung)
    DEFAULT_GOAL_WIDTH = 80      # Breite der Torzone in Pixeln
    DEFAULT_GOAL_MARGIN_Y = 100  # Abstand vom Rand oben/unten
    
    # Erkennungs-Parameter
    MIN_GOAL_SPEED = 200         # Mindest-Geschwindigkeit für Tor (px/s)
    MIN_FRAMES_LOST = 5          # Frames ohne Ball für Tor-Bestätigung
    MAX_FRAMES_LOST = 30         # Max Frames - danach wird nicht mehr als Tor gewertet
    MIN_VELOCITY_TOWARDS_GOAL = 3.0  # Mindest-Geschwindigkeit Richtung Tor (px/frame)
    
    def __init__(self, 
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 fps: float = 180.0):
        """
        Args:
            frame_width: Bildbreite
            frame_height: Bildhöhe  
            fps: Framerate für Zeitberechnung
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Torzonen (werden durch calibrate() oder load_config() gesetzt)
        self.goal_left: Optional[GoalZone] = None
        self.goal_right: Optional[GoalZone] = None
        
        # Spielfeld-Grenzen
        self.play_area: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        
        # State
        self.frame_count = 0
        self.ball_history: Deque[BallHistory] = deque(maxlen=30)
        self.frames_ball_lost = 0
        self.potential_goal: Optional[dict] = None
        
        # Events
        self.goal_events: List[GoalEvent] = []
        self.last_goal_frame = -100  # Cooldown zwischen Toren
        
        # Score
        self.score_white = 0
        self.score_black = 0
        
        # Standard-Zonen initialisieren
        self._init_default_zones()
    
    def _init_default_zones(self):
        """Initialisiere Standard-Torzonen basierend auf Bildgröße"""
        w, h = self.frame_width, self.frame_height
        
        # Linkes Tor (White kassiert hier, Black trifft)
        self.goal_left = GoalZone(
            x_min=0,
            x_max=self.DEFAULT_GOAL_WIDTH,
            y_min=self.DEFAULT_GOAL_MARGIN_Y,
            y_max=h - self.DEFAULT_GOAL_MARGIN_Y,
            team=Team.WHITE
        )
        
        # Rechtes Tor (Black kassiert hier, White trifft)
        self.goal_right = GoalZone(
            x_min=w - self.DEFAULT_GOAL_WIDTH,
            x_max=w,
            y_min=self.DEFAULT_GOAL_MARGIN_Y,
            y_max=h - self.DEFAULT_GOAL_MARGIN_Y,
            team=Team.BLACK
        )
        
        # Spielfeld (ohne Torzonen)
        self.play_area = (
            self.DEFAULT_GOAL_WIDTH,
            0,
            w - 2 * self.DEFAULT_GOAL_WIDTH,
            h
        )
    
    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Interaktive Kalibrierung der Torzonen.
        
        Args:
            frame: Referenz-Frame für Kalibrierung
            
        Returns:
            True wenn erfolgreich, False wenn abgebrochen
        """
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h
        
        zones = []
        current_zone = []
        
        win_name = "TORERKENNUNG KALIBRIERUNG"
        help_text = [
            "Klicke 4 Punkte fuer linkes Tor (oben-links, oben-rechts, unten-rechts, unten-links)",
            "Dann 4 Punkte fuer rechtes Tor",
            "[R] = Reset  [S] = Speichern  [ESC] = Abbrechen"
        ]
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_zone
            if event == cv2.EVENT_LBUTTONDOWN:
                current_zone.append((x, y))
                if len(current_zone) == 4:
                    zones.append(current_zone.copy())
                    current_zone = []
        
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, mouse_callback)
        
        while True:
            vis = frame.copy()
            
            # Bereits definierte Zonen zeichnen
            colors = [(255, 0, 0), (0, 0, 255)]  # Blau=Links, Rot=Rechts
            labels = ["LEFT (White kassiert)", "RIGHT (Black kassiert)"]
            
            for i, zone in enumerate(zones):
                pts = np.array(zone, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, colors[i], 3)
                cv2.putText(vis, labels[i], (zone[0][0], zone[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            # Aktuelle Punkte zeichnen
            for pt in current_zone:
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
            if len(current_zone) >= 2:
                for i in range(len(current_zone) - 1):
                    cv2.line(vis, current_zone[i], current_zone[i+1], (0, 255, 0), 2)
            
            # Hilfe-Text
            for i, text in enumerate(help_text):
                cv2.putText(vis, text, (20, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            status = f"Zonen: {len(zones)}/2 | Aktuelle Punkte: {len(current_zone)}/4"
            cv2.putText(vis, status, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyWindow(win_name)
                return False
            
            if key in (ord('r'), ord('R')):
                zones = []
                current_zone = []
            
            if key in (ord('s'), ord('S')) and len(zones) >= 2:
                break
        
        cv2.destroyWindow(win_name)
        
        # Zonen aus Punkten erstellen
        def points_to_zone(pts, team):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return GoalZone(
                x_min=min(xs),
                x_max=max(xs),
                y_min=min(ys),
                y_max=max(ys),
                team=team
            )
        
        self.goal_left = points_to_zone(zones[0], Team.WHITE)
        self.goal_right = points_to_zone(zones[1], Team.BLACK)
        
        # Spielfeld aktualisieren
        self.play_area = (
            self.goal_left.x_max,
            0,
            self.goal_right.x_min - self.goal_left.x_max,
            h
        )
        
        print(f"✅ Torzonen kalibriert:")
        print(f"   Links: x={self.goal_left.x_min}-{self.goal_left.x_max}")
        print(f"   Rechts: x={self.goal_right.x_min}-{self.goal_right.x_max}")
        
        return True
    
    def calibrate_simple(self, frame: np.ndarray, 
                         left_x: int, right_x: int,
                         goal_y_top: int, goal_y_bottom: int) -> None:
        """
        Einfache Kalibrierung mit direkten Werten.
        
        Args:
            left_x: X-Position der linken Tor-Außenkante
            right_x: X-Position der rechten Tor-Außenkante
            goal_y_top: Y-Position Oberkante Torbereich
            goal_y_bottom: Y-Position Unterkante Torbereich
        """
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h
        
        goal_width = 80  # Annahme
        
        self.goal_left = GoalZone(
            x_min=0,
            x_max=left_x + goal_width,
            y_min=goal_y_top,
            y_max=goal_y_bottom,
            team=Team.WHITE
        )
        
        self.goal_right = GoalZone(
            x_min=right_x - goal_width,
            x_max=w,
            y_min=goal_y_top,
            y_max=goal_y_bottom,
            team=Team.BLACK
        )
        
        self.play_area = (left_x, 0, right_x - left_x, h)
    
    def save_config(self, path: str = "goal_config.json"):
        """Speichere Konfiguration"""
        config = {
            "version": 1,
            "created": datetime.datetime.now().isoformat(),
            "frame_size": [self.frame_width, self.frame_height],
            "goal_left": {
                "x_min": self.goal_left.x_min,
                "x_max": self.goal_left.x_max,
                "y_min": self.goal_left.y_min,
                "y_max": self.goal_left.y_max,
                "team": self.goal_left.team.value
            },
            "goal_right": {
                "x_min": self.goal_right.x_min,
                "x_max": self.goal_right.x_max,
                "y_min": self.goal_right.y_min,
                "y_max": self.goal_right.y_max,
                "team": self.goal_right.team.value
            },
            "play_area": list(self.play_area) if self.play_area else None,
            "parameters": {
                "min_goal_speed": self.MIN_GOAL_SPEED,
                "min_frames_lost": self.MIN_FRAMES_LOST,
                "max_frames_lost": self.MAX_FRAMES_LOST,
                "min_velocity_towards_goal": self.MIN_VELOCITY_TOWARDS_GOAL
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Goal-Config gespeichert: {path}")
    
    def load_config(self, path: str = "goal_config.json") -> bool:
        """Lade Konfiguration"""
        if not os.path.exists(path):
            print(f"⚠️ Config nicht gefunden: {path}")
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.frame_width, self.frame_height = config["frame_size"]
            
            gl = config["goal_left"]
            self.goal_left = GoalZone(
                x_min=gl["x_min"], x_max=gl["x_max"],
                y_min=gl["y_min"], y_max=gl["y_max"],
                team=Team(gl["team"])
            )
            
            gr = config["goal_right"]
            self.goal_right = GoalZone(
                x_min=gr["x_min"], x_max=gr["x_max"],
                y_min=gr["y_min"], y_max=gr["y_max"],
                team=Team(gr["team"])
            )
            
            if config.get("play_area"):
                self.play_area = tuple(config["play_area"])
            
            params = config.get("parameters", {})
            self.MIN_GOAL_SPEED = params.get("min_goal_speed", self.MIN_GOAL_SPEED)
            self.MIN_FRAMES_LOST = params.get("min_frames_lost", self.MIN_FRAMES_LOST)
            self.MAX_FRAMES_LOST = params.get("max_frames_lost", self.MAX_FRAMES_LOST)
            
            print(f"✅ Goal-Config geladen: {path}")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return False
    
    def update(self, ball_x: Optional[float], ball_y: Optional[float],
               ball_vx: float = 0, ball_vy: float = 0,
               ball_speed: float = 0, ball_tracked: bool = False) -> Optional[GoalEvent]:
        """
        Update mit aktuellen Ball-Daten.
        
        Args:
            ball_x, ball_y: Ball-Position (None wenn nicht sichtbar)
            ball_vx, ball_vy: Geschwindigkeit (Pixel/Frame)
            ball_speed: Geschwindigkeit (Pixel/Sekunde)
            ball_tracked: True wenn Ball aktiv detektiert (nicht predicted)
            
        Returns:
            GoalEvent wenn Tor erkannt, sonst None
        """
        self.frame_count += 1
        
        # Ball-History aktualisieren
        if ball_x is not None and ball_y is not None:
            self.ball_history.append(BallHistory(
                x=ball_x, y=ball_y,
                vx=ball_vx, vy=ball_vy,
                speed=ball_speed,
                frame=self.frame_count,
                tracked=ball_tracked
            ))
            self.frames_ball_lost = 0
        else:
            self.frames_ball_lost += 1
        
        # Cooldown nach letztem Tor
        if self.frame_count - self.last_goal_frame < 30:
            return None
        
        # Tor-Erkennung
        goal = self._check_for_goal()
        
        if goal:
            self.goal_events.append(goal)
            self.last_goal_frame = self.frame_count
            
            # Score aktualisieren
            if goal.scoring_team == Team.WHITE:
                self.score_white += 1
            elif goal.scoring_team == Team.BLACK:
                self.score_black += 1
            
            # Reset für nächstes Tor
            self.potential_goal = None
            self.ball_history.clear()
        
        return goal
    
    def update_from_tracker(self, ball_state) -> Optional[GoalEvent]:
        """
        Convenience-Methode für Integration mit BallTracker.
        
        Args:
            ball_state: BallState-Objekt vom BallTracker
            
        Returns:
            GoalEvent wenn Tor erkannt
        """
        from ball_tracker_v2 import TrackingState
        
        if ball_state.state == TrackingState.LOST:
            return self.update(None, None, 0, 0, 0, False)
        else:
            tracked = (ball_state.state == TrackingState.TRACKING)
            return self.update(
                ball_state.x, ball_state.y,
                ball_state.vx, ball_state.vy,
                ball_state.speed_px_s,
                tracked
            )
    
    def _check_for_goal(self) -> Optional[GoalEvent]:
        """Interne Tor-Prüfung"""
        
        # Brauchen Historie
        if len(self.ball_history) < 3:
            return None
        
        # Ball muss verloren sein
        if self.frames_ball_lost < self.MIN_FRAMES_LOST:
            return None
        
        if self.frames_ball_lost > self.MAX_FRAMES_LOST:
            # Zu lange weg - kein Tor, Ball wurde rausgenommen o.ä.
            self.ball_history.clear()
            return None
        
        # Letzte bekannte Positionen analysieren
        recent = list(self.ball_history)[-10:]  # Letzte 10 Einträge
        
        if not recent:
            return None
        
        # Letzte Position und Durchschnitts-Geschwindigkeit
        last = recent[-1]
        
        # Durchschnittliche Geschwindigkeit der letzten Frames
        vx_avg = np.mean([h.vx for h in recent])
        vy_avg = np.mean([h.vy for h in recent])
        speed_avg = np.mean([h.speed for h in recent])
        
        # Prüfe: War Ball in Torzone oder auf dem Weg dorthin?
        goal_zone = None
        goal_side = None
        
        # Linkes Tor (Ball bewegt sich nach links, vx < 0)
        if vx_avg < -self.MIN_VELOCITY_TOWARDS_GOAL:
            if (last.x < self.goal_left.x_max + 50 and
                self.goal_left.y_min < last.y < self.goal_left.y_max):
                goal_zone = self.goal_left
                goal_side = "left"
        
        # Rechtes Tor (Ball bewegt sich nach rechts, vx > 0)
        elif vx_avg > self.MIN_VELOCITY_TOWARDS_GOAL:
            if (last.x > self.goal_right.x_min - 50 and
                self.goal_right.y_min < last.y < self.goal_right.y_max):
                goal_zone = self.goal_right
                goal_side = "right"
        
        if goal_zone is None:
            return None
        
        # Mindest-Geschwindigkeit prüfen
        if speed_avg < self.MIN_GOAL_SPEED:
            return None
        
        # Confidence berechnen
        confidence = self._calc_goal_confidence(last, vx_avg, speed_avg, goal_zone)
        
        if confidence < 0.5:
            return None
        
        # Tor bestätigt!
        scoring_team = Team.BLACK if goal_zone.team == Team.WHITE else Team.WHITE
        
        return GoalEvent(
            timestamp=self.frame_count / self.fps,
            frame_number=self.frame_count,
            scoring_team=scoring_team,
            conceding_team=goal_zone.team,
            ball_entry_pos=(last.x, last.y),
            ball_velocity=(vx_avg, vy_avg),
            ball_speed=speed_avg,
            confidence=confidence,
            goal_side=goal_side
        )
    
    def _calc_goal_confidence(self, last_pos: BallHistory, 
                              vx: float, speed: float,
                              goal_zone: GoalZone) -> float:
        """Berechne Konfidenz der Tor-Erkennung"""
        confidence = 0.0
        
        # Faktor 1: Geschwindigkeit (höher = sicherer)
        speed_factor = min(speed / 500, 1.0)  # Max bei 500 px/s
        confidence += 0.3 * speed_factor
        
        # Faktor 2: Richtung stimmt
        if goal_zone == self.goal_left and vx < -3:
            direction_factor = min(abs(vx) / 10, 1.0)
        elif goal_zone == self.goal_right and vx > 3:
            direction_factor = min(abs(vx) / 10, 1.0)
        else:
            direction_factor = 0.0
        confidence += 0.3 * direction_factor
        
        # Faktor 3: Position war nah am Tor
        dist = goal_zone.distance_to(last_pos.x, last_pos.y)
        max_dist = 200
        position_factor = max(0, 1 - dist / max_dist)
        confidence += 0.25 * position_factor
        
        # Faktor 4: Ball war getrackt (nicht nur predicted)
        if last_pos.tracked:
            confidence += 0.15
        else:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def reset_score(self):
        """Setze Spielstand zurück"""
        self.score_white = 0
        self.score_black = 0
        self.goal_events.clear()
        print("🔄 Spielstand zurückgesetzt")
    
    def reset(self):
        """Komplett-Reset"""
        self.reset_score()
        self.frame_count = 0
        self.ball_history.clear()
        self.frames_ball_lost = 0
        self.potential_goal = None
        self.last_goal_frame = -100
    
    def get_score(self) -> Tuple[int, int]:
        """Returns: (white_score, black_score)"""
        return (self.score_white, self.score_black)
    
    def draw(self, frame: np.ndarray, 
             show_zones: bool = True,
             show_score: bool = True) -> np.ndarray:
        """
        Zeichne Torzonen und Score auf Frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        if show_zones:
            # Linke Torzone (halbtransparent blau)
            if self.goal_left:
                overlay = vis.copy()
                cv2.rectangle(overlay, 
                             (self.goal_left.x_min, self.goal_left.y_min),
                             (self.goal_left.x_max, self.goal_left.y_max),
                             (255, 100, 0), -1)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                cv2.rectangle(vis,
                             (self.goal_left.x_min, self.goal_left.y_min),
                             (self.goal_left.x_max, self.goal_left.y_max),
                             (255, 100, 0), 2)
            
            # Rechte Torzone (halbtransparent rot)
            if self.goal_right:
                overlay = vis.copy()
                cv2.rectangle(overlay,
                             (self.goal_right.x_min, self.goal_right.y_min),
                             (self.goal_right.x_max, self.goal_right.y_max),
                             (0, 100, 255), -1)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                cv2.rectangle(vis,
                             (self.goal_right.x_min, self.goal_right.y_min),
                             (self.goal_right.x_max, self.goal_right.y_max),
                             (0, 100, 255), 2)
        
        if show_score:
            # Score-Anzeige oben Mitte
            score_text = f"WHITE {self.score_white} : {self.score_black} BLACK"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            
            # Hintergrund
            cv2.rectangle(vis, (text_x - 15, 5), 
                         (text_x + text_size[0] + 15, 50), (0, 0, 0), -1)
            
            # Text
            cv2.putText(vis, score_text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return vis
    
    def draw_goal_animation(self, frame: np.ndarray, 
                           goal_event: GoalEvent,
                           frames_since_goal: int) -> np.ndarray:
        """
        Zeichne Tor-Animation.
        
        Args:
            frame: Basis-Frame
            goal_event: Das Tor-Event
            frames_since_goal: Frames seit dem Tor (für Animation)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Animation dauert 60 Frames (~0.3s bei 180fps)
        if frames_since_goal > 60:
            return vis
        
        # Pulsierender Effekt
        pulse = abs(np.sin(frames_since_goal * 0.3)) * 0.3 + 0.2
        
        # Overlay
        overlay = vis.copy()
        
        # Team-Farbe
        if goal_event.scoring_team == Team.WHITE:
            color = (255, 255, 255)
            text = "GOAL WHITE!"
        else:
            color = (50, 50, 50)
            text = "GOAL BLACK!"
        
        # Hintergrund-Flash
        cv2.rectangle(overlay, (0, h//3), (w, 2*h//3), color, -1)
        cv2.addWeighted(overlay, pulse, vis, 1-pulse, 0, vis)
        
        # Text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 + text_size[1] // 2
        
        # Schatten
        cv2.putText(vis, text, (text_x + 3, text_y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        # Text
        cv2.putText(vis, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
        
        return vis


# ============================================================
# Convenience-Funktionen für einfache Integration
# ============================================================

_global_detector: Optional[GoalDetector] = None


def init_goal_detector(frame_width: int = 1920, 
                       frame_height: int = 1080,
                       fps: float = 180.0,
                       config_path: Optional[str] = "goal_config.json"):
    """Initialisiere globalen Goal-Detector"""
    global _global_detector
    _global_detector = GoalDetector(frame_width, frame_height, fps)
    
    if config_path and os.path.exists(config_path):
        _global_detector.load_config(config_path)


def check_goal(ball_x: Optional[float], ball_y: Optional[float],
               ball_vx: float = 0, ball_vy: float = 0,
               ball_speed: float = 0) -> Optional[GoalEvent]:
    """Prüfe auf Tor mit globalem Detector"""
    global _global_detector
    if _global_detector is None:
        init_goal_detector()
    return _global_detector.update(ball_x, ball_y, ball_vx, ball_vy, ball_speed, ball_x is not None)


def get_score() -> Tuple[int, int]:
    """Hole aktuellen Spielstand (white, black)"""
    global _global_detector
    if _global_detector is None:
        return (0, 0)
    return _global_detector.get_score()


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=== GoalDetector Test ===\n")
    
    # Erstelle Detector
    detector = GoalDetector(frame_width=1920, frame_height=1080, fps=180)
    
    print(f"Linkes Tor: x={detector.goal_left.x_min}-{detector.goal_left.x_max}")
    print(f"Rechtes Tor: x={detector.goal_right.x_min}-{detector.goal_right.x_max}")
    
    # Simuliere Ball-Bewegung Richtung linkes Tor
    print("\n--- Simulation: Ball fliegt ins linke Tor ---")
    
    # Ball startet in der Mitte und fliegt nach links
    x, y = 960, 540
    vx, vy = -15, 0  # Schnell nach links
    
    for frame in range(50):
        x += vx
        y += vy
        speed = abs(vx) * 180  # px/s
        
        # Ball verschwindet bei x < 50
        if x > 50:
            goal = detector.update(x, y, vx, vy, speed, ball_tracked=True)
        else:
            goal = detector.update(None, None, 0, 0, 0, ball_tracked=False)
        
        if goal:
            print(f"\n🎯 TOR in Frame {frame}!")
            print(f"   Scoring Team: {goal.scoring_team.value}")
            print(f"   Position: ({goal.ball_entry_pos[0]:.0f}, {goal.ball_entry_pos[1]:.0f})")
            print(f"   Speed: {goal.ball_speed:.0f} px/s")
            print(f"   Confidence: {goal.confidence:.2f}")
            break
    
    print(f"\nEndstand: White {detector.score_white} : {detector.score_black} Black")
    
    # Test mit Bild
    print("\n--- Visualisierungs-Test ---")
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    test_frame[:] = (40, 80, 40)  # Grün
    
    vis = detector.draw(test_frame)
    cv2.imwrite("/mnt/user-data/outputs/goal_zones_test.png", vis)
    print("Gespeichert: goal_zones_test.png")
    
    print("\n✅ Test abgeschlossen")
