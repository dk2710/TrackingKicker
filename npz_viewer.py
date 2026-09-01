"""Play a saved NPZ frame sequence as a local OpenCV video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORIES = (
    PROJECT_ROOT / "v3" / "videos",
    PROJECT_ROOT / "v3" / "tracking_kicker" / "videos",
)


def _default_npz() -> Path | None:
    candidates = []
    for directory in DEFAULT_DIRECTORIES:
        candidates.extend(directory.glob("goal_pre_*.npz"))
        candidates.extend(directory.glob("session_raw_*.npz"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _as_scalar(data, name: str, default=""):
    if name not in data:
        return default
    value = data[name]
    return value.item() if value.shape == () else value


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data:
            raise ValueError("Die NPZ-Datei enthält kein 'frames'-Array.")

        frames = np.asarray(data["frames"])
        if frames.ndim not in (3, 4) or len(frames) == 0:
            raise ValueError(
                f"Unerwartete Frame-Form {frames.shape}; erwartet N,H,W oder N,H,W,C."
            )

        if "times" in data:
            times = np.asarray(data["times"], dtype=np.float64).reshape(-1)
        else:
            times = np.arange(len(frames), dtype=np.float64) / 120.0

        if len(times) != len(frames):
            raise ValueError("'times' und 'frames' enthalten unterschiedlich viele Einträge.")

        # Absolute Kamerazeiten sind für die Wiedergabe nicht nötig. Die erste
        # Zeit wird daher wie bei RawReplaySource auf null gesetzt.
        times = times - times[0]
        metadata = {
            "goal_position": _as_scalar(data, "goal_position", ""),
            "speed_kmh": _as_scalar(data, "speed_kmh", None),
            "goal_time": _as_scalar(data, "goal_time", None),
        }

    return frames, times, metadata


def _display_frame(frame: np.ndarray, raw: bool) -> np.ndarray:
    if frame.ndim == 2:
        if raw:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Die Kamera speichert BayerRG8. OpenCV erzeugt daraus ein normales
        # BGR-Bild für die Anzeige.
        return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)

    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 3:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Nicht unterstützte Farbkanalzahl: {frame.shape}")


def _wait_ms(times: np.ndarray, index: int, speed: float, default_fps: float) -> int:
    if index + 1 < len(times):
        delta = max(0.001, float(times[index + 1] - times[index]))
    else:
        delta = 1.0 / default_fps
    return max(1, min(1000, round(delta * 1000.0 / speed)))


def play(path: Path, raw: bool, speed: float, loop: bool, info_only: bool) -> None:
    frames, times, metadata = _load_npz(path)
    duration = float(times[-1]) if len(times) else 0.0
    measured_fps = (len(times) - 1) / duration if duration > 0 else 120.0

    print(f"Datei: {path}")
    print(f"Frames: {len(frames)} | Auflösung: {frames.shape[1]} x {frames.shape[2]}")
    print(f"Dauer: {duration:.3f} s | gemessene FPS: {measured_fps:.1f}")
    if metadata["goal_position"]:
        print(f"Tor: {metadata['goal_position']}")
    if metadata["speed_kmh"] is not None:
        print(f"Geschwindigkeit: {float(metadata['speed_kmh']):.2f} km/h")

    if info_only:
        return

    cv2.namedWindow("NPZ Replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NPZ Replay", min(1280, frames.shape[2]), min(800, frames.shape[1]))

    index = 0
    paused = False
    while True:
        image = _display_frame(frames[index], raw).copy()
        text = f"Frame {index + 1}/{len(frames)} | t={times[index]:.3f}s"
        cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if metadata["goal_position"]:
            goal_text = f"Tor: {metadata['goal_position']}"
            if metadata["speed_kmh"] is not None:
                goal_text += f" | {float(metadata['speed_kmh']):.2f} km/h"
            cv2.putText(image, goal_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if paused:
            cv2.putText(image, "PAUSE (Leertaste zum Fortsetzen)", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("NPZ Replay", image)
        key = cv2.waitKey(0 if paused else _wait_ms(times, index, speed, measured_fps)) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("r"):
            index = 0
            paused = False
            continue
        if key in (ord("n"), 83):
            index = min(index + 1, len(frames) - 1)
            paused = True
            continue

        if not paused:
            if index + 1 < len(frames):
                index += 1
            elif loop:
                index = 0
            else:
                # Nach dem letzten Frame offen stehen bleiben, damit das
                # Ergebnis inspiziert werden kann.
                paused = True

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ-Rohframes als Video abspielen")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="NPZ-Datei; ohne Angabe wird die neueste goal_pre/session_raw-Datei verwendet",
    )
    parser.add_argument("--raw", action="store_true", help="Bayerbild als Graustufen anzeigen")
    parser.add_argument("--speed", type=float, default=1.0, help="Wiedergabegeschwindigkeit, z. B. 0.5 oder 2")
    parser.add_argument("--loop", action="store_true", help="Nach dem letzten Frame wieder von vorne beginnen")
    parser.add_argument("--info", action="store_true", help="Nur Metadaten anzeigen, kein Fenster öffnen")
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed muss größer als null sein.")

    path = args.file or _default_npz()
    if path is None:
        parser.error("Keine NPZ-Datei gefunden.")
    if not path.is_file():
        parser.error(f"Datei nicht gefunden: {path}")

    try:
        play(path.resolve(), args.raw, args.speed, args.loop, args.info)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
