# Basler Kamera-Tuner

Dieses eigenständige Projekt öffnet die erste Basler-Kamera, zeigt das aktuelle
Bild als Live-Stream im Browser und erlaubt das Testen der für den
Tischkicker-Tracker wichtigen Einstellungen.

## Start auf dem Orange Pi

Aus dem Projektverzeichnis:

```bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m basler_camera_tuner.main
```

Danach auf dem Laptop öffnen:

```text
http://192.168.10.63:5010/
```

Der Port kann mit `--port` geändert werden. Die Einstellungen werden in
`basler_camera_tuner/settings.json` gespeichert und bei jedem Reconnect erneut
angewendet.

## Wichtige Einstellungen

- `force_fps`: Auflösung und FPS müssen zusammen in die verfügbare
  GigE-Bandbreite passen. Für die aktuelle ROI-Konfiguration ist 120 FPS ein
  stabiler Startwert.
- `grab_strategy`: `latest_image_only` ist für den Live-Tuner sicherer;
  `one_by_one` liefert für die Tracking-Anwendung jedes empfangene Bild.
- `width`, `height`, `offset_x`, `offset_y`: Kamera-ROI.
- `exposure_time`, `gain`, `exposure_auto`, `gain_auto`: Bildhelligkeit und
  Bewegungsunschärfe.
- `packet_size`, `inter_packet_delay`: optionale GigE-Transportparameter;
  `0` lässt den Kamerawert unverändert.

Jede Änderung stoppt den Stream, öffnet die Kamera neu, ruft
`_configure_reconnected_camera()` auf und startet den Stream mit den neuen
Werten. Dadurch entspricht das Verhalten auch dem Reconnect im Tracker.
