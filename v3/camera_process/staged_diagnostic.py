"""Run the camera pipeline in isolated startup stages.

The diagnostic deliberately starts and stops each stage independently. This
lets a transport failure be assigned to camera acquisition, frame handoff,
the web consumer, or ball detection instead of mixing all four timelines.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen

from flask import Flask, jsonify

from basler_camera_tuner.settings import load_settings, load_tracker_ball_settings
from v3.tracking_kicker.logging_support import configure_runtime_logging

from .camera import CameraProcess
from .main import _run_website_process


STAGES = ("camera", "handoff", "publish", "consumer", "web", "detection")


def _stage_trace(stage: str, event: str, **fields: Any) -> None:
    payload = {"stage": stage, "event": event, **fields}
    print("[STAGE-TRACE] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))


class _ProcessHolder:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: Optional[CameraProcess] = None


def _create_control_app(holder: _ProcessHolder) -> Flask:
    app = Flask("staged_camera_control")

    @app.get("/health")
    def health():
        with holder.lock:
            process = holder.process
        if process is None:
            return jsonify({"status": "idle", "camera_open": False})
        status = process.status()
        return jsonify({"status": "ok", "camera_open": status["camera_open"]})

    @app.get("/api/status")
    def status():
        with holder.lock:
            process = holder.process
        return jsonify(process.status() if process is not None else {
            "camera_open": False,
            "grabbing": False,
            "last_error": "diagnostic stage is idle",
        })

    return app


def _run_control_server(holder: _ProcessHolder, host: str, port: int) -> None:
    app = _create_control_app(holder)
    _stage_trace("control", "server_start", host=host, port=port)
    app.run(host=host, port=port, threaded=True, use_reloader=False)


def _fetch_json(url: str) -> Optional[dict[str, Any]]:
    try:
        with urlopen(url, timeout=0.8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, URLError):
        return None


def _camera_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    frame_buffer = status.get("frame_buffer") or {}
    return {
        "camera_open": status.get("camera_open"),
        "grabbing": status.get("grabbing"),
        "camera_fps": status.get("camera_fps"),
        "resulting_fps": status.get("resulting_fps"),
        "camera_frames_total": status.get("camera_frames_total"),
        "failed_grabs": status.get("failed_grabs"),
        "payload_errors": status.get("payload_errors"),
        "reconnects": status.get("reconnects"),
        "last_failure_phase": status.get("last_failure_phase"),
        "last_result_diagnostic": status.get("last_result_diagnostic"),
        "last_failure_age_seconds": status.get("last_failure_age_seconds"),
        "capture_handoff_dropped": status.get("capture_handoff_dropped"),
        "last_error": status.get("last_error"),
        "capture_aborted": status.get("capture_aborted"),
        "frames_published": frame_buffer.get("frames_published"),
        "frames_replaced": frame_buffer.get("frames_replaced"),
        "frame_buffer_errors": frame_buffer.get("errors"),
    }


def _website_snapshot(status: Optional[dict[str, Any]]) -> dict[str, Any]:
    if status is None:
        return {"reachable": False}
    return {
        "reachable": True,
        "detection_enabled": status.get("ball_detection_enabled"),
        "detection_fps": status.get("detection_fps"),
        "detection_frames": status.get("detection_frames"),
        "received_frames": status.get("received_frames"),
        "dropped_frames": status.get("dropped_frames"),
        "ipc_errors": status.get("ipc_errors"),
        "preview_fps": status.get("preview_fps"),
        "preview_enabled": status.get("preview_enabled"),
        "last_error": status.get("last_error"),
    }


def _wait_for_camera_frame(process: CameraProcess, stage: str, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        status = process.status()
        _stage_trace(stage, "camera_startup_status", **_camera_snapshot(status))
        if status.get("camera_open") and int(status.get("camera_frames_total", 0)) > 0:
            return True
        if status.get("capture_aborted"):
            return False
        time.sleep(0.25)
    return False


def _run_stage(
    stage: str,
    *,
    seconds: float,
    camera_settings,
    ball_settings,
    tracker_detection_enabled: bool,
    server_settings: dict[str, Any],
    settings_path: Path,
    tracker_config_path: Path,
    context,
    holder: _ProcessHolder,
    control_url: str,
    web_port: int,
) -> bool:
    publish = stage in {"handoff", "publish", "consumer", "web", "detection"}
    queue_publish = stage in {"publish", "consumer", "web", "detection"}
    website = stage in {"consumer", "web", "detection"}
    detection = stage == "detection" and tracker_detection_enabled
    preview = stage in {"web", "detection"}
    frame_queue = context.Queue(maxsize=1) if queue_publish else None
    website_process = None
    process = CameraProcess(
        camera_settings,
        settings_path=settings_path,
        frame_ipc_name=str(server_settings.get("frame_ipc_name", "tracking_kicker_camera_frames")),
        frame_ipc_slots=int(server_settings.get("frame_ipc_slots", 3)),
        ball_settings=ball_settings,
        publish_frames=publish,
        frame_queue=frame_queue,
    )
    with holder.lock:
        holder.process = process

    initial_status: dict[str, Any] = {}
    result = False
    started = time.monotonic()
    _stage_trace(
        stage,
        "start",
        publish=publish,
        website=website,
        detection=detection,
        preview=preview,
        queue=queue_publish,
        duration_seconds=seconds,
    )
    try:
        process.start()
        _stage_trace(stage, "camera_process_started")
        ready = _wait_for_camera_frame(process, stage, timeout=5.0)
        if not ready:
            _stage_trace(stage, "camera_ready_failed", **_camera_snapshot(process.status()))
            return False
        initial_status = process.status()
        _stage_trace(stage, "camera_ready", **_camera_snapshot(initial_status))

        if website:
            website_process = context.Process(
                target=_run_website_process,
                args=(
                    camera_settings,
                    ball_settings,
                    detection,
                    settings_path,
                    {**server_settings, "ball_detection_enabled": detection},
                    tracker_config_path,
                    frame_queue,
                    "127.0.0.1",
                    web_port,
                    control_url,
                    preview,
                ),
                name=f"tracking-kicker-website-{stage}",
                daemon=True,
            )
            _stage_trace(stage, "website_start_requested", web_port=web_port)
            website_process.start()
            _stage_trace(stage, "website_process_started", pid=website_process.pid)

        deadline = started + max(0.1, seconds)
        last_failure_count = int(initial_status.get("failed_grabs", 0) or 0)
        while time.monotonic() < deadline:
            camera_status = process.status()
            web_status = (
                _fetch_json(f"http://127.0.0.1:{web_port}/api/status")
                if website
                else None
            )
            camera_snapshot = _camera_snapshot(camera_status)
            website_snapshot = _website_snapshot(web_status)
            _stage_trace(
                stage,
                "heartbeat",
                elapsed_seconds=round(time.monotonic() - started, 3),
                camera=camera_snapshot,
                website=website_snapshot,
            )
            failure_count = int(camera_status.get("failed_grabs", 0) or 0)
            if failure_count > last_failure_count:
                _stage_trace(
                    stage,
                    "camera_failure_observed",
                    previous_failed_grabs=last_failure_count,
                    current_failed_grabs=failure_count,
                    **camera_snapshot,
                )
                return False
            last_failure_count = failure_count
            if not camera_status.get("camera_open") or camera_status.get("capture_aborted"):
                _stage_trace(stage, "camera_stopped", **camera_snapshot)
                return False
            time.sleep(1.0)

        final_camera_status = process.status()
        final_camera_snapshot = _camera_snapshot(final_camera_status)
        final_failure_count = int(final_camera_status.get("failed_grabs", 0) or 0)
        final_web_status = (
            _fetch_json(f"http://127.0.0.1:{web_port}/api/status")
            if website
            else None
        )
        if final_failure_count > last_failure_count:
            _stage_trace(
                stage,
                "camera_failure_observed",
                previous_failed_grabs=last_failure_count,
                current_failed_grabs=final_failure_count,
                **final_camera_snapshot,
            )
            return False
        if not final_camera_status.get("camera_open") or final_camera_status.get("capture_aborted"):
            _stage_trace(stage, "camera_stopped", **final_camera_snapshot)
            return False
        result = True
        _stage_trace(
            stage,
            "completed",
            elapsed_seconds=round(time.monotonic() - started, 3),
            camera=final_camera_snapshot,
            website=_website_snapshot(final_web_status),
        )
        return True
    finally:
        _stage_trace(stage, "stop_requested")
        if website_process is not None:
            if website_process.is_alive():
                website_process.terminate()
            website_process.join(timeout=5.0)
            _stage_trace(
                stage,
                "website_stopped",
                exitcode=website_process.exitcode,
            )
        process.stop()
        with holder.lock:
            holder.process = None
        if frame_queue is not None:
            frame_queue.close()
            frame_queue.cancel_join_thread()
        _stage_trace(stage, "stopped", result=result)


def main() -> None:
    configure_runtime_logging()
    parser = argparse.ArgumentParser(description="Stufenweiser CameraProcess-Diagnosetest")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "basler_camera_tuner"
        / "settings.json",
    )
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Kommagetrennt: camera,handoff,publish,consumer,web,detection",
    )
    parser.add_argument("--control-port", type=int, default=5021)
    parser.add_argument("--web-port", type=int, default=5030)
    args = parser.parse_args()

    stages = [item.strip().lower() for item in str(args.stages).split(",") if item.strip()]
    invalid = [item for item in stages if item not in STAGES]
    if invalid or not stages:
        parser.error(f"Ungueltige Stufen: {invalid or 'keine'}; erlaubt: {', '.join(STAGES)}")

    camera_settings, _unused_ball_settings, server_settings = load_settings(args.settings)
    tracker_config_path = Path(__file__).parents[1] / "tracking_kicker" / "config.json"
    ball_settings, tracker_detection_enabled = load_tracker_ball_settings(
        tracker_config_path
    )
    context = multiprocessing.get_context("spawn")
    holder = _ProcessHolder()
    control_thread = threading.Thread(
        target=_run_control_server,
        args=(holder, "127.0.0.1", int(args.control_port)),
        name="staged-control-api",
        daemon=True,
    )
    control_thread.start()
    time.sleep(0.3)
    control_url = f"http://127.0.0.1:{int(args.control_port)}"

    _stage_trace(
        "run",
        "start",
        stages=stages,
        seconds=args.seconds,
        requested_fps=camera_settings.force_fps,
        pixel_format=camera_settings.pixel_format,
        tracker_detection_enabled=tracker_detection_enabled,
    )
    overall = True
    for index, stage in enumerate(stages):
        if index:
            time.sleep(1.0)
        stage_ok = _run_stage(
            stage,
            seconds=float(args.seconds),
            camera_settings=camera_settings,
            ball_settings=ball_settings,
            tracker_detection_enabled=tracker_detection_enabled,
            server_settings=server_settings,
            settings_path=args.settings,
            tracker_config_path=tracker_config_path,
            context=context,
            holder=holder,
            control_url=control_url,
            web_port=int(args.web_port) + index,
        )
        _stage_trace("run", "stage_result", stage_name=stage, ok=stage_ok)
        if not stage_ok:
            overall = False
            _stage_trace("run", "stopped_after_failure", failed_stage=stage)
            break
    _stage_trace("run", "complete", ok=overall)
    raise SystemExit(0 if overall else 2)


if __name__ == "__main__":
    main()
