# Basler Kamera-Tuner

Dieses eigenständige Projekt öffnet die erste Basler-Kamera, zeigt das aktuelle
Bild als Live-Stream im Browser und erlaubt das Testen der für den
Tischkicker-Tracker wichtigen Einstellungen.

## Start auf dem Orange Pi

Aus dem Projektverzeichnis in zwei Terminals:

```bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m v3.camera_process.main
```

Danach den BallTuner starten:

```bash
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

- `force_fps`: Die Kamera wird für die aktuelle ROI-Konfiguration mit mindestens
  180 FPS angefordert. Die Oberfläche zeigt zusätzlich die tatsächlich
  resultierende pylon-Rate und die erfolgreich empfangenen Kamera-Frames an.
- `grab_strategy`: `latest_image_only` ist die aktive Strategie für Tuner und
  Tracking. Ältere fertige Frames dürfen dabei verworfen werden.
- `max_transfer_size` und `num_max_queued_urbs`: optionale USB-
  Stream-Grabber-Parameter; `0` lässt den pylon-Standardwert unverändert.
- `width`, `height`, `offset_x`, `offset_y`: Kamera-ROI.
- `exposure_time`, `gain`, `exposure_auto`, `gain_auto`: Bildhelligkeit und
  Bewegungsunschärfe.
- `packet_size`, `inter_packet_delay`: optionale GigE-Transportparameter;
  `0` lässt den Kamerawert unverändert.

Der CameraProcess spricht nur mit pylon und schreibt rohe Frames in den
Shared-Memory-Ring. Der BallTuner liest diesen Ring, erzeugt die auf 30 FPS
begrenzte JPEG-Vorschau und führt optional die Ballerkennung aus. Dadurch wird
die Kameraaufnahme nicht durch die Browserdarstellung ausgebremst. Mit **Ballerkennung: EIN/AUS** lassen sich
Ballkontur, Farbmaske und Kalibrieranzeige ausblenden bzw. wieder aktivieren;
die Kameraaufnahme läuft dabei weiter. Der Zustand wird in `settings.json`
gespeichert.

Der Tracker und der BallTuner lesen die Rohframes über den Shared-Memory-Ring
`tracking_kicker_camera_frames`. Nur der CameraProcess öffnet die
Basler-Kamera; BallTuner und Tracker greifen ohne JPEG-Umweg auf die Frames zu.
Die Kameraeinstellungen werden vom BallTuner über die lokale Control-API auf
Port 5011 an den CameraProcess gesendet.

## Ballerkennung einstellen

Im Bereich **Ballerkennung (HSV)** werden die Farbgrenzen der Ballmaske
eingestellt:

- `H Min`/`H Max` (**0–179**) bestimmen den Farbton. Blau liegt typischerweise
  ungefähr zwischen 100 und 120, kann aber durch Licht und Weißabgleich
  abweichen.
- `S Min`/`S Max` (**0–255**) bestimmen die Farbsättigung. Ein höheres `S Min`
  blendet graue Tischbereiche aus.
- `V Min`/`V Max` (**0–255**) bestimmen die Helligkeit. Ein höheres `V Min`
  blendet dunkle Schatten aus; ein niedrigeres `V Min` hilft bei dunklem Licht.

In der **Farbmaske** sind weiße Pixel diejenigen, die den HSV-Bereich erfüllen.
Im **Ball-Overlay** werden daraus nur Konturen akzeptiert, die zusätzlich zu
den geometrischen Filtern passen. Die gelbe Markierung zeigt die ausgewählte
Ballkontur.

- `Fläche Min`/`Fläche Max`: Größe der Kontur in Pixeln im Originalbild.
- `Radius Min`/`Radius Max`: erwarteter Ballradius in Pixeln.
- `Rundheit Min`: filtert längliche oder unregelmäßige Konturen aus.
- `Solidität Min`: filtert Konturen mit großen Löchern oder Einbuchtungen aus.
- `Erkennungsskalierung`: verkleinert das Bild nur für die Erkennung. `0.5`
  ist schneller; `1.0` ist genauer, benötigt aber mehr CPU.

Zum Einstellen zuerst nur den HSV-Bereich verändern, bis der Ball weiß und
der übrige Tisch möglichst schwarz in der Farbmaske erscheint. Danach können
Fläche, Radius, Rundheit und Solidität angepasst werden. Die Werte werden in
`settings.json` gespeichert. Der Tracker verwendet dieselben Werte in
`v3/tracking_kicker/config.json` unter `ball_detection`.

Mit **Für Tracker übernehmen** werden die aktuellen Ballwerte direkt in
`v3/tracking_kicker/config.json` im Format `hsv_low: [H, S, V]` und
`hsv_high: [H, S, V]` gespeichert. Danach muss der Tracker neu gestartet
werden, damit er die neuen Werte lädt.

Jede Änderung stoppt den Stream, öffnet die Kamera neu, ruft
`_configure_reconnected_camera()` auf und startet den Stream mit den neuen
Werten. Dadurch entspricht das Verhalten auch dem Reconnect im Tracker.
