"""HTTP API compatible with the v3 tracker, without video endpoints."""

from __future__ import annotations

import math
import logging
import time
from dataclasses import asdict
from flask import Flask, jsonify, request


def _number(value, name, minimum=None, maximum=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} muss eine Zahl sein") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} muss endlich sein")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} muss mindestens {minimum} sein")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} darf hoechstens {maximum} sein")
    return result


def _calibration_payload(vision):
    return {
        "goals": asdict(vision.goals) if vision.goals is not None else None,
        "field": asdict(vision.field) if vision.field is not None else None,
    }


def create_app(service, config, vision) -> Flask:
    # Request access logs write synchronously to stderr and can briefly hold
    # the GIL while the USB receiver is draining 209 FPS. Keep API errors
    # available but suppress routine request lines in the high-rate process.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = Flask(__name__)

    @app.get("/")
    def dashboard():
        return jsonify({
            "service": "v4-ball-detection",
            "status_url": "/api/status",
            "parameters_url": "/api/parameters",
            "logs_url": "/api/logs",
            "video_enabled": False,
        })

    @app.get("/api/status")
    @app.get("/status")
    def status():
        return jsonify(service.status(vision=vision))

    @app.get("/api/logs")
    def logs():
        after = request.args.get("after", default=0, type=int) or 0
        limit = request.args.get("limit", default=200, type=int) or 200
        entries, latest = service.logs(after=after, limit=limit)
        return jsonify({"logs": entries, "latest_id": latest})

    @app.get("/api/parameters")
    def get_parameters():
        return jsonify({
            "ball_detection": service.detection_parameters(),
            "enabled": bool(config.ball_detection_enabled),
            "camera": asdict(config.camera) if config.camera else {},
            "calibration": _calibration_payload(vision),
            "video_enabled": False,
        })

    @app.post("/api/parameters")
    def update_parameters():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON-Objekt erwartet"}), 400
        updates = payload.get("ball_detection", payload)
        if not isinstance(updates, dict):
            return jsonify({"error": "ball_detection muss ein Objekt sein"}), 400
        try:
            values = service.validate_detection_update(updates)
            service.update_detection(values)
            config.ball_detection = dict(values)
            if bool(payload.get("persist", False)):
                config.save()
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "status": "ok",
            "persisted": bool(payload.get("persist", False)),
            "enabled": bool(config.ball_detection_enabled),
            "ball_detection": service.detection_parameters(),
        })

    @app.route("/api/calibration", methods=["GET", "POST"])
    def calibration():
        if request.method == "GET":
            return jsonify(_calibration_payload(vision))
        # v4 intentionally performs no video calibration while running the
        # high-rate detector. Keep the old endpoint explicit for clients.
        return jsonify({
            "error": "Video-Kalibrierung ist im reinen Erkennungsmodus deaktiviert"
        }), 409

    @app.get("/api/test")
    @app.get("/test")
    def test():
        return jsonify({"status": "ok", "service": "v4-ball-detection"})

    @app.route("/live.mjpg", methods=["GET"])
    @app.route("/api/live.mjpg", methods=["GET"])
    def live_disabled():
        return jsonify({
            "error": "Video-Stream ist im reinen Erkennungsmodus deaktiviert"
        }), 404

    return app
