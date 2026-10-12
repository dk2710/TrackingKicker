import threading
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from flask import Flask

from tracking_kicker.comms.http_api import ApiMixin
from tracking_kicker.goal_media import GoalMediaMixin
from tracking_kicker.goal_replay import (
    calculate_three_point_speed_kmh,
    select_goal_frames,
)
from tracking_kicker.runtime import (
    GOAL_VIDEO_FRAME_LIMIT,
    GOAL_VIDEO_FPS,
)
from tracking_kicker.Objects.Frames import Frame


FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class ApiHarness(ApiMixin):
    """Small API-only harness; no camera or encoder is started."""

    def __init__(self):
        self.api = Flask(__name__)
        self.api.add_url_rule("/goal", "goal", self.api_goal, methods=["GET"])
        self.api.add_url_rule(
            "/goal/video/<path:filename>",
            "goal_video",
            self.api_goal_video,
            methods=["GET"],
        )
        self.goal_video_lock = threading.Lock()
        self.goal_video_path = None
        self.goal_video_thread = None
        self.goal_video_generation = 0
        self.goal_video_error = None
        self.goal_video_result = None


class RetroHarness(GoalMediaMixin):
    def __init__(self):
        from tracking_kicker.camera.distortion import DistortionCorrector
        from tracking_kicker.runtime import config

        self.dist = DistortionCorrector(config.camera)
        self.calibration_lock = threading.RLock()

        class Analyzer:
            @staticmethod
            def _ensure_homography(_shape):
                return None

            @staticmethod
            def _img_to_cm(x, y):
                return float(x), float(y)

        self.analyzer = Analyzer()


class GoalReplayTests(unittest.TestCase):
    def test_api_waits_for_a_confirmed_goal(self):
        harness = ApiHarness()
        response = harness.api.test_client().get("/goal")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["status"], "no_goal")

    def test_selects_90_frames_from_three_seconds(self):
        source = [
            (np.zeros((2, 2), dtype=np.uint8), index / 180.0)
            for index in range(540)
        ]

        selected = select_goal_frames(source, 3.0)

        self.assertEqual(len(selected), GOAL_VIDEO_FRAME_LIMIT)
        self.assertAlmostEqual(selected[0][1], 0.0, places=3)
        self.assertLessEqual(selected[-1][1], 3.0)
        self.assertEqual(len({round(timestamp, 6) for _, timestamp in selected}), 90)

    def test_speed_uses_three_positions_and_timestamps(self):
        points = [
            {"timestamp": 0.0, "x_cm": 0.0, "y_cm": 0.0},
            {"timestamp": 0.1, "x_cm": 10.0, "y_cm": 0.0},
            {"timestamp": 0.2, "x_cm": 20.0, "y_cm": 0.0},
        ]

        self.assertAlmostEqual(
            calculate_three_point_speed_kmh(points),
            3.6,
            places=5,
        )

    def test_api_returns_goal_data_and_video_url(self):
        harness = ApiHarness()
        with tempfile.TemporaryDirectory() as temporary:
            video_dir = Path(temporary)
            video_path = video_dir / "goal_latest.webm"
            video_path.write_bytes(b"test-video")
            harness.goal_video_generation = 1
            harness.goal_video_path = str(video_path)
            harness.goal_video_result = {
                "event": "goal",
                "event_id": 1,
                "side": "left",
                "goal_timestamp": 3.0,
                "speed_kmh": 12.5,
                "analysis": {
                    "direction": "backward",
                    "point_count": 3,
                    "points": [],
                },
                "video_frame_count": GOAL_VIDEO_FRAME_LIMIT,
                "video_fps": GOAL_VIDEO_FPS,
            }

            with patch(
                "tracking_kicker.comms.http_api.ANALYSIS_VIDEO_PATH",
                video_dir,
            ):
                client = harness.api.test_client()
                response = client.get("/goal")
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["video_frame_count"], 90)
                self.assertEqual(payload["video_fps"], 30.0)
                self.assertEqual(payload["analysis"]["point_count"], 3)

                video_response = client.get(payload["video_url"])
                self.assertEqual(video_response.status_code, 200)
                self.assertEqual(video_response.mimetype, "video/webm")
                self.assertEqual(video_response.data, b"test-video")
                video_response.close()

    def test_reverse_analysis_stops_after_three_points(self):
        harness = RetroHarness()
        source = [
            (np.zeros((10, 10), dtype=np.uint8), timestamp)
            for timestamp in (0.8, 0.9, 1.0)
        ]

        class Detector:
            @classmethod
            def from_config(cls, *_args):
                return cls()

            def _reset_track(self):
                return None

            def detect(self, frame):
                frame.ball = (int(round(frame.timestamp * 100.0)), 2, 3)

        with patch("tracking_kicker.goal_media.BallDetector", Detector):
            result = harness._retro_analyze_last_second(source, 1.0)

        self.assertEqual(result["direction"], "backward")
        self.assertEqual(result["point_count"], 3)
        self.assertEqual(len(result["points"]), 3)
        self.assertEqual(
            result["points"],
            sorted(result["points"], key=lambda point: point["timestamp"]),
        )
        self.assertAlmostEqual(result["speed_kmh"], 3.6, places=3)

    @unittest.skipUnless(FFMPEG_AVAILABLE, "FFmpeg/ffprobe ist nicht installiert")
    def test_creates_three_second_30_fps_video(self):
        harness = GoalMediaMixin()
        frames = []
        for index in range(GOAL_VIDEO_FRAME_LIMIT):
            frame = Frame()
            frame.frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            frame.timestamp = index / GOAL_VIDEO_FPS
            frames.append(frame)

        with tempfile.TemporaryDirectory() as temporary:
            with patch("tracking_kicker.goal_media.ANALYSIS_VIDEO_PATH", Path(temporary)):
                video_path = Path(harness.create_goal_video(frames))
                self.assertTrue(video_path.exists())

                probe = subprocess.run(
                    [
                        shutil.which("ffprobe"),
                        "-v", "error",
                        "-count_frames",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames,r_frame_rate",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()

                self.assertIn("30/1", probe)
                self.assertIn("90", probe)

    @unittest.skipUnless(FFMPEG_AVAILABLE, "FFmpeg/ffprobe ist nicht installiert")
    def test_confirmed_goal_prepares_analysis_and_video_result(self):
        harness = RetroHarness()
        harness.goal_video_lock = threading.Lock()
        harness.goal_video_generation = 1
        harness.goal_video_path = None
        harness.goal_video_speed = 0.0
        harness.goal_video_frame_count = 0
        harness.goal_video_fps = 0.0
        harness.goal_video_result = None
        harness.goal_video_error = None
        harness.last_analysis_lock = threading.Lock()
        harness.last_analysis = {}
        harness._last_posted_goal_event_id = 0
        harness._post_goal_result = lambda _payload: None

        source = []
        for index in range(GOAL_VIDEO_FRAME_LIMIT):
            image = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            source.append((image, index / GOAL_VIDEO_FPS))

        event = {
            "event": "goal",
            "event_id": 7,
            "side": "left",
            "goal_position": "left",
            "goal_timestamp": 3.0,
            "candidate_id": 4,
        }
        analysis = {
            "direction": "backward",
            "window_seconds": 1.0,
            "point_count": 3,
            "points": [
                {"timestamp": 2.8, "x_cm": 0.0, "y_cm": 0.0},
                {"timestamp": 2.9, "x_cm": 10.0, "y_cm": 0.0},
                {"timestamp": 3.0, "x_cm": 20.0, "y_cm": 0.0},
            ],
            "speed_kmh": 3.6,
        }

        with tempfile.TemporaryDirectory() as temporary:
            with patch("tracking_kicker.goal_media.ANALYSIS_VIDEO_PATH", Path(temporary)):
                with patch.object(
                    harness,
                    "_retro_analyze_last_second",
                    return_value=analysis,
                ):
                    harness._prepare_goal_media(
                        event,
                        {"frames": source},
                        1,
                    )

            self.assertIsNone(harness.goal_video_error)
            self.assertEqual(harness.goal_video_speed, 3.6)
            self.assertEqual(harness.goal_video_frame_count, 90)
            self.assertEqual(harness.goal_video_result["status"], "ready")
            self.assertEqual(
                harness.goal_video_result["analysis"]["point_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
