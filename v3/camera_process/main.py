"""Headless camera producer with a small localhost control API."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request

from basler_camera_tuner.settings import load_settings
from v3.tracking_kicker.logging_support import (
    configure_runtime_logging,
    latest_log_id,
    runtime_logs,
)

from .camera import CameraProcess


def create_app(process: CameraProcess) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        status = process.status()
        return jsonify({"status": "ok", "camera_open": status["camera_open"]})

    @app.get("/api/status")
    def status():
        return jsonify(process.status())

    @app.get("/api/logs")
    def logs():
        after = request.args.get("after", default=0, type=int)
        limit = request.args.get("limit", default=200, type=int)
        return jsonify({
            "logs": runtime_logs(after=after or 0, limit=limit or 200),
            "latest_id": latest_log_id(),
        })

    @app.post("/api/settings")
    def settings():
        payload = request.get_json(silent=True) or {}
        values = payload.get("camera", payload)
        if not isinstance(values, dict):
            return jsonify({"error": "camera muss ein Objekt sein"}), 400
        try:
            opened = process.update_settings(values)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"camera_open": opened, "settings": process.status()["settings"]})

    @app.post("/api/reconnect")
    def reconnect():
        opened = process.reconnect()
        return jsonify({
            "camera_open": opened,
            "message": "Kamera neu verbunden" if opened else "Kamera nicht verfuegbar",
        }), (200 if opened else 503)

    return app


def main() -> None:
    configure_runtime_logging()
    parser = argparse.ArgumentParser(description="TrackingKicker CameraProcess")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "basler_camera_tuner"
        / "settings.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument(
        "--skip-frame-publish",
        action="store_true",
        help="Nur pylon/USB testen; Frames nicht in Shared Memory schreiben",
    )
    args = parser.parse_args()

    camera_settings, ball_settings, server_settings = load_settings(args.settings)
    process = CameraProcess(
        camera_settings,
        settings_path=args.settings,
        frame_ipc_name=str(
            server_settings.get("frame_ipc_name", "tracking_kicker_camera_frames")
        ),
        frame_ipc_slots=int(server_settings.get("frame_ipc_slots", 3)),
        ball_settings=ball_settings,
        publish_frames=not args.skip_frame_publish,
    )
    process.start()
    app = create_app(process)
    print(f"[CAMERA-PROCESS] Control-API: http://{args.host}:{args.port}/")
    try:
        app.run(
            host=str(args.host),
            port=int(args.port),
            threaded=True,
            use_reloader=False,
        )
    finally:
        process.stop()


if __name__ == "__main__":
    main()
