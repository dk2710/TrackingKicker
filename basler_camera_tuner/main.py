from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, render_template_string, request

from .camera import BaslerCamera
from .settings import load_settings, save_settings


PAGE = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Basler Kamera-Tuner</title>
  <style>
    body { margin: 0; background: #111827; color: #e5e7eb;
           font-family: system-ui, sans-serif; }
    main { max-width: 1250px; margin: auto; padding: 18px; }
    .layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
              gap: 18px; align-items: start; }
    .previews { display: grid; grid-template-columns: 1fr; gap: 14px; }
    h3 { font-size: .9rem; margin: 0 0 6px; color: #bfdbfe; }
    .card { background: #1f2937; border-radius: 10px; padding: 16px;
            box-shadow: 0 5px 20px #0005; }
    img { width: 100%; min-height: 240px; object-fit: contain; background: #000;
          border-radius: 6px; }
    h1, h2 { margin-top: 0; }
    h1 { font-size: 1.35rem; }
    h2 { font-size: 1.05rem; margin-bottom: 12px; }
    .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 9px 12px; }
    label { display: flex; flex-direction: column; gap: 4px; font-size: .85rem; }
    input, select, button { box-sizing: border-box; border: 1px solid #4b5563;
      border-radius: 5px; padding: 8px; background: #111827; color: #f9fafb; }
    button { background: #2563eb; border: 0; cursor: pointer; margin-top: 14px; }
    button.secondary { background: #4b5563; margin-left: 6px; }
    .status { white-space: pre-wrap; font-family: ui-monospace, monospace;
              font-size: .8rem; line-height: 1.45; color: #bfdbfe; }
    .hint { font-size: .8rem; color: #9ca3af; line-height: 1.4; }
    #message { min-height: 1.3em; margin-top: 10px; color: #a7f3d0; }
    @media (max-width: 850px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Basler Kamera-Tuner</h1>
  <div class="layout">
    <section class="card">
      <h2>Livebild und Ballerkennung</h2>
      <div class="previews">
        <div><h3>Rohbild</h3>
          <img src="/video_feed" alt="Livebild der Basler-Kamera"></div>
        <div><h3>Ball-Overlay</h3>
          <img src="/ball_feed" alt="Livebild mit erkannter Ballkontur"></div>
        <div><h3>Farbmaske</h3>
          <img src="/mask_feed" alt="HSV-Farbmaske der Ballerkennung"></div>
      </div>
      <p class="hint">Die Seite kann vom Laptop aus geöffnet werden. Der Tuner zeigt
        den zuletzt empfangenen Frame als Live-MJPEG-Stream.</p>
      <div id="status" class="status">Status wird geladen ...</div>
    </section>
    <section class="card">
      <h2>Kameraeinstellungen</h2>
      <div class="fields">
        <label>FPS <input id="force_fps" type="number" step="1" min="0"></label>
        <label>PixelFormat <select id="pixel_format">
          <option>BayerRG8</option><option>Mono8</option>
        </select></label>
        <label>Breite <input id="width" type="number" min="1"></label>
        <label>Höhe <input id="height" type="number" min="1"></label>
        <label>Offset X <input id="offset_x" type="number" min="0"></label>
        <label>Offset Y <input id="offset_y" type="number" min="0"></label>
        <label>Belichtungszeit (µs) <input id="exposure_time" type="number" step=".1" min="0"></label>
        <label>Gain <input id="gain" type="number" step=".1" min="0"></label>
        <label>ExposureAuto <select id="exposure_auto">
          <option>Off</option><option>Continuous</option><option>Once</option>
        </select></label>
        <label>GainAuto <select id="gain_auto">
          <option>Off</option><option>Continuous</option><option>Once</option>
        </select></label>
        <label>Grab-Strategie <select id="grab_strategy">
          <option value="latest_image_only">LatestImageOnly</option>
          <option value="one_by_one">OneByOne</option>
        </select></label>
        <label>MaxNumBuffer <input id="max_num_buffer" type="number" min="1"></label>
        <label>Packet Size (0=default) <input id="packet_size" type="number" min="0"></label>
        <label>Inter-Packet Delay <input id="inter_packet_delay" type="number" min="0"></label>
      </div>
      <button onclick="applySettings()">Einstellungen anwenden</button>
      <button class="secondary" onclick="reconnect()">Neu verbinden</button>
      <button class="secondary" onclick="exportTracker()">Für Tracker übernehmen</button>
      <div id="message"></div>
      <p class="hint">Für das Tuning zuerst <b>LatestImageOnly</b> verwenden. Für die
        Tracking-Anwendung erst auf <b>OneByOne</b> umstellen, wenn FPS und Netzwerk
        stabil sind. Änderungen werden in <code>settings.json</code> gespeichert und
        nach jedem Reconnect erneut angewendet.</p>
      <h2 style="margin-top: 22px">Ballerkennung (HSV)</h2>
      <div class="fields">
        <label>H Min <input id="hsv_low_h" type="number" min="0" max="179"></label>
        <label>H Max <input id="hsv_high_h" type="number" min="0" max="179"></label>
        <label>S Min <input id="hsv_low_s" type="number" min="0" max="255"></label>
        <label>S Max <input id="hsv_high_s" type="number" min="0" max="255"></label>
        <label>V Min <input id="hsv_low_v" type="number" min="0" max="255"></label>
        <label>V Max <input id="hsv_high_v" type="number" min="0" max="255"></label>
        <label>Fläche Min <input id="area_min" type="number" step=".1" min="0"></label>
        <label>Fläche Max <input id="area_max" type="number" step=".1" min="0"></label>
        <label>Erkennungsskalierung <input id="detection_scale" type="number" step=".05" min=".05" max="1"></label>
        <label>Radius Min (px) <input id="radius_min_px" type="number" step=".1" min="0"></label>
        <label>Radius Max (px) <input id="radius_max_px" type="number" step=".1" min="0"></label>
        <label>Rundheit Min <input id="circularity_min" type="number" step=".01" min="0" max="1"></label>
        <label>Solidität Min <input id="solidity_min" type="number" step=".01" min="0" max="1"></label>
      </div>
      <p class="hint">HSV bedeutet Farbton (H), Sättigung (S) und Helligkeit (V).
        Weiße Bereiche in der Farbmaske erfüllen den Farbbereich. Der gelbe Kreis
        im Overlay ist die daraus ausgewählte Ballkontur.</p>
    </section>
  </div>
</main>
<script>
const names = ["force_fps", "width", "height", "offset_x", "offset_y",
  "exposure_time", "gain", "pixel_format", "exposure_auto", "gain_auto",
  "grab_strategy", "max_num_buffer", "packet_size", "inter_packet_delay"];
const ballNames = ["hsv_low_h", "hsv_high_h", "hsv_low_s", "hsv_high_s",
  "hsv_low_v", "hsv_high_v", "area_min", "area_max", "detection_scale",
  "radius_min_px", "radius_max_px", "circularity_min", "solidity_min"];
let settingsLoaded = false;

function fill(settings) {
  for (const name of names) {
    const element = document.getElementById(name);
    if (element && settings[name] !== undefined) element.value = settings[name];
  }
}

function values() {
  const result = {};
  for (const name of names) {
    const element = document.getElementById(name);
    result[name] = element.type === "number" ? Number(element.value) : element.value;
  }
  return result;
}

function ballValues() {
  const result = {};
  for (const name of ballNames) {
    const element = document.getElementById(name);
    result[name] = Number(element.value);
  }
  return result;
}

function fillBall(settings) {
  for (const name of ballNames) {
    const element = document.getElementById(name);
    if (element && settings[name] !== undefined) element.value = settings[name];
  }
}

async function refresh() {
  const response = await fetch("/api/status");
  const data = await response.json();
  // Polling must not overwrite values while the user is editing the form.
  if (!settingsLoaded) {
    fill(data.settings);
    fillBall(data.ball_detection);
    settingsLoaded = true;
  }
  document.getElementById("status").textContent = JSON.stringify(data, null, 2);
}

async function applySettings() {
  const message = document.getElementById("message");
  message.textContent = "Wende Einstellungen an ...";
  const response = await fetch("/api/settings", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({camera: values(), ball_detection: ballValues()})
  });
  const data = await response.json();
  message.textContent = data.message || (response.ok ? "Gespeichert" : "Fehler");
  await refresh();
}

async function reconnect() {
  const response = await fetch("/api/reconnect", {method: "POST"});
  const data = await response.json();
  document.getElementById("message").textContent = data.message || "Neu verbunden";
  await refresh();
}

async function exportTracker() {
  const response = await fetch("/api/export_tracker", {method: "POST"});
  const data = await response.json();
  document.getElementById("message").textContent = data.message || "Tracker-Konfiguration aktualisiert";
}

refresh().catch(error => document.getElementById("message").textContent = error);
setInterval(() => refresh().catch(() => {}), 1500);
</script>
</body>
</html>
"""


def _export_tracker_ball_settings(
    tracker_config_path: Optional[Path],
    camera: BaslerCamera,
) -> None:
    if tracker_config_path is None or not tracker_config_path.exists():
        raise FileNotFoundError("v3/tracking_kicker/config.json nicht gefunden")

    with tracker_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["ball_detection"] = camera.ball_settings.as_tracker_dict()

    temporary = tracker_config_path.with_suffix(tracker_config_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    temporary.replace(tracker_config_path)


def create_app(
    camera: BaslerCamera,
    settings_path: Path,
    server_settings: dict,
    tracker_config_path: Optional[Path] = None,
) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "camera": camera.status()})

    @app.get("/api/status")
    def status():
        return jsonify(camera.status())

    @app.get("/snapshot.jpg")
    def snapshot():
        image = camera.latest_jpeg()
        if image is None:
            return jsonify({"message": "Noch kein Kamerabild vorhanden"}), 503
        return Response(image, mimetype="image/jpeg")

    @app.get("/video_feed")
    def video_feed():
        def stream():
            while True:
                image = camera.latest_jpeg()
                if image is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" + image + b"\r\n"
                    )
                time.sleep(0.04)

        return Response(
            stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/ball_feed")
    def ball_feed():
        def stream():
            while True:
                image = camera.latest_ball_jpeg()
                if image is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" + image + b"\r\n"
                    )
                time.sleep(0.04)

        return Response(
            stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/mask_feed")
    def mask_feed():
        def stream():
            while True:
                image = camera.latest_mask_jpeg()
                if image is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" + image + b"\r\n"
                    )
                time.sleep(0.04)

        return Response(
            stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/settings")
    def update_settings():
        payload = request.get_json(silent=True) or {}
        camera_values = payload.get("camera", payload)
        ball_values = payload.get("ball_detection", {})
        if not isinstance(camera_values, dict) or not isinstance(ball_values, dict):
            return jsonify({"message": "camera und ball_detection müssen Objekte sein"}), 400
        try:
            ok = camera.update_settings(camera_values) if camera_values else True
            if ball_values:
                camera.update_ball_settings(ball_values)
            save_settings(
                settings_path,
                camera.settings,
                camera.ball_settings,
                server_settings,
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"message": str(exc)}), 400
        return jsonify({
            "message": "Einstellungen gespeichert und Kamera neu verbunden",
            "camera_open": ok,
            "settings": camera.status()["settings"],
            "ball_detection": camera.status()["ball_detection"],
        })

    @app.post("/api/reconnect")
    def reconnect():
        ok = camera.reconnect()
        return jsonify({
            "message": "Kamera neu verbunden" if ok else "Kamera nicht verfuegbar",
            "camera_open": ok,
        }), (200 if ok else 503)

    @app.post("/api/export_tracker")
    def export_tracker():
        try:
            _export_tracker_ball_settings(tracker_config_path, camera)
        except (OSError, TypeError, ValueError) as exc:
            return jsonify({"message": str(exc)}), 500
        return jsonify({
            "message": "Ballerkennungswerte in die Tracker-Konfiguration übernommen",
            "ball_detection": camera.ball_settings.as_tracker_dict(),
        })

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Basler Livebild und Kamera-Tuner")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).with_name("settings.json"),
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    camera_settings, ball_settings, server_settings = load_settings(args.settings)
    if args.host is not None:
        server_settings["host"] = args.host
    if args.port is not None:
        server_settings["port"] = args.port

    camera = BaslerCamera(camera_settings, ball_settings)
    camera.start()
    tracker_config_path = Path(__file__).parent.parent / "v3" / "tracking_kicker" / "config.json"
    app = create_app(camera, args.settings, server_settings, tracker_config_path)
    print(
        "Basler Tuner erreichbar unter "
        f"http://<PI-IP>:{server_settings['port']}/"
    )
    try:
        app.run(
            host=str(server_settings["host"]),
            port=int(server_settings["port"]),
            threaded=True,
            use_reloader=False,
        )
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
