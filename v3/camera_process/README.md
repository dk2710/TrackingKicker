# CameraProcess

This process owns the Basler camera and starts the website in a second OS
process:

```bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m v3.camera_process.main
```

It runs with `LatestImageOnly` and hands the newest raw `BayerRG8` frame
through a one-element `multiprocessing.Queue`. The website, JPEG preview and
optional ball detection run in the separate website process. No SharedMemory
frame ring is created by this combined start.

The website is available at `http://192.168.10.63:5010/`. The localhost
control API remains available on port 5011 for status, settings and reconnects.

For a staged transport diagnosis, run:

```bash
.venv/bin/python -m v3.camera_process.staged_diagnostic --seconds 15
```

The stages are `camera` (pylon only), `handoff` (camera plus the internal
latest-frame handoff), `publish` (handoff plus the multiprocessing queue),
`consumer` (website frame consumer without preview or detection), `web`
(handoff plus website without detection), and `detection` (the complete
path). Each stage writes JSON lines tagged with `[STAGE-TRACE]`; camera and
website internals use `[CAMERA-TRACE]`, `[PROCESS-TRACE]`, and
`[BALL-TUNER-TRACE]`. The run stops at the first stage that loses the camera.
