# CameraProcess

This is the headless producer and the only process that opens the Basler camera:

```bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m v3.camera_process.main
```

It runs with `LatestImageOnly` and publishes raw `BayerRG8` frames through the
shared-memory ring `tracking_kicker_camera_frames`. It has a localhost control
API on port 5011 for status, settings and reconnects, but it does not encode
JPEGs or run ball detection.

Start the BallTuner separately to provide the website, 30 FPS preview,
timestamped logs and ball detection:

```bash
.venv/bin/python -u -m basler_camera_tuner.main
```
