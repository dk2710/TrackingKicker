"""HTTP endpoints for status, goal metadata and generated videos."""

import time
from pathlib import Path

from flask import jsonify, request, send_from_directory, url_for

from ..runtime import ANALYSIS_VIDEO_PATH, GOAL_VIDEO_FRAME_LIMIT


class ApiMixin:
    """Flask handlers kept separate from camera and analysis workers."""

    def api_test(self):
        print("TEST")
        return jsonify({"status": "ok"})

    def api_loop(self):
        self.api.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False,
        )

    def _api_goal_fast(self):
        """Start/poll the latest three-second goal video without blocking."""
        refresh = request.args.get("refresh", "").lower() in {
            "1", "true", "yes", "on"
        }
        requested_generation = request.args.get("generation", type=int)

        with self.goal_video_lock:
            video_path = self.goal_video_path
            preparing = (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            )
            generation = self.goal_video_generation
            result_delivered = self.goal_video_result_delivered
            error = self.goal_video_error

        should_start = refresh or (
            requested_generation is None
            and not preparing
            and (video_path is None or result_delivered)
        )
        if should_start:
            generation = self._start_goal_video_preparation(force=True)
            preparing = True

        if requested_generation is not None and requested_generation != generation:
            requested_generation = generation

        if preparing or video_path is None:
            with self.last_analysis_lock:
                latest_goal = dict(self.last_analysis)
            response = {
                "status": "preparing",
                "video_ready": False,
                "video_generation": generation,
                "ball_speed_kmh": float(self.goal_video_speed),
                "retry_after_ms": 100,
                "status_url": url_for("goal", generation=generation),
            }
            if latest_goal:
                response["goal_position"] = latest_goal.get("goal_position")
                response["goal_side"] = latest_goal.get("side")
            if error:
                response["error"] = error
            return jsonify(response), 202

        if error:
            return jsonify({
                "status": "error",
                "video_ready": False,
                "video_generation": generation,
                "message": error,
            }), 503

        with self.goal_video_lock:
            self.goal_video_result_delivered = True

        speed = float(self.goal_video_speed)
        frame_count = self.goal_video_frame_count or GOAL_VIDEO_FRAME_LIMIT
        with self.last_analysis_lock:
            latest_goal = dict(self.last_analysis)
        video_filename = Path(video_path).name
        try:
            video_version = Path(video_path).stat().st_mtime_ns
        except OSError:
            video_version = time.time_ns()

        response = {
            "status": "ready",
            "video_ready": True,
            "video_generation": generation,
            "ball_speed_kmh": speed,
            "video_url": url_for(
                "goal_video",
                filename=video_filename,
                v=video_version,
            ),
            "video_filename": video_filename,
            "video_frame_count": frame_count,
            "video_fps": self.goal_video_fps,
        }
        if latest_goal:
            response["goal_position"] = latest_goal.get("goal_position")
            response["goal_side"] = latest_goal.get("side")
        return jsonify(response), 200

    def api_goal(self):
        return self._api_goal_fast()

    def _api_goal_legacy(self):
        with self.goal_video_lock:
            video_path = self.goal_video_path
            preparing = (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            )

        if not preparing:
            self._start_goal_video_preparation(force=True)
            preparing = True

        if preparing:
            self.goal_video_ready.wait(timeout=6.0)
            with self.goal_video_lock:
                video_path = self.goal_video_path

        speed = float(self.goal_video_speed)
        frame_count = GOAL_VIDEO_FRAME_LIMIT
        if video_path is None:
            return jsonify({
                "status": "error",
                "message": "Die letzten 3 Sekunden konnten nicht als Video erstellt werden",
            }), 503

        video_filename = Path(video_path).name
        video_url = url_for("goal_video", filename=video_filename)
        return jsonify({
            "ball_speed_kmh": speed,
            "video_url": video_url,
            "video_filename": video_filename,
            "video_frame_count": frame_count,
            "video_fps": self.goal_video_fps,
        })

    def api_goal_video(self, filename):
        response = send_from_directory(
            str(ANALYSIS_VIDEO_PATH),
            Path(filename).name,
            mimetype="video/webm",
            as_attachment=False,
        )
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
