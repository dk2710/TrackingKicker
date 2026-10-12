# TrackingKicker

TrackingKicker ist jetzt eine einzige Softwarebasis. Der produktive Tracker,
der CameraProcess, der Browser-Tuner und die Konfiguration liegen gemeinsam
unter `tracking_kicker/`. Die frühere Aufteilung in `v3` und `v4` ist entfernt.

## Start auf dem Pi

Aus dem Projektverzeichnis starten:

```bash
.venv/bin/python -u -m tracking_kicker.camera_process.main
```

Der CameraProcess besitzt die Basler-Kamera und stellt seine lokale
Steuerungs-API auf Port `5011` bereit. Den Tracker in einem zweiten Prozess
starten:

```bash
.venv/bin/python -u -m tracking_kicker.app
```

Der Tracker stellt das Dashboard und die API auf Port `5000` bereit. Der
optionale Browser-Tuner läuft zusätzlich auf Port `5010`:

```bash
.venv/bin/python -u -m tracking_kicker.tuner.main --host 0.0.0.0 --port 5010
```

Die Kamera- und Tunerwerte liegen in
`tracking_kicker/tuner/settings.json`. Die Tracker-, Tor- und Spielfeldwerte
liegen in `tracking_kicker/config.json` und `tracking_kicker/vision_config.json`.

## Goal-API

Nach einem bestätigten Tor bereitet der Tracker die letzten drei Sekunden als
90 Farbframes mit 30 FPS auf und analysiert das letzte Sekundenfenster rückwärts.
Das Ergebnis kann über `GET /goal` auf Port `5000` abgefragt werden. Während der
Aufbereitung antwortet die API mit `202`; danach enthält die Antwort die drei
Ballpunkte, die Geschwindigkeit und die Video-URL.

## Entwicklung

Abhängigkeiten installieren:

```bash
python -m pip install -r requirements.txt
```

Tests und Syntaxprüfung:

```bash
python -m unittest discover -s tests -v
python -m compileall -q tracking_kicker tests tools
```

`tools/npz_viewer.py` dient zum lokalen Abspielen gespeicherter Aufnahmen.
`tools/transport_monitor.py` ist ein optionales Transportdiagnose-Skript für
die CameraProcess-API.
