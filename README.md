# TrackingKicker – Team-Dokumentation

Diese Dokumentation beschreibt den Stand des Branches team/tracking-kicker.
Der Branch enthält eine einzige TrackingKicker-Software mit CameraProcess,
Tracker, Goal-API und optionalem Browser-Tuner.

Der produktive Ablauf besteht aus zwei notwendigen Prozessen:

1. Der CameraProcess besitzt die Basler-Kamera, liest die pylon-Grab-Ergebnisse
   und veröffentlicht die aktuellen Rohbilder über Shared Memory.
2. Der Tracker liest diese Bilder, hält den Drei-Sekunden-Puffer, erkennt den
   Tor-Kandidaten und erstellt nach der Torbestätigung Analyse und Video.

Der Browser-Tuner ist optional. Er zeigt Kamerastatus und Livebilder und kann
Kamera- oder Ballerkennungswerte über die APIs ändern.

## 1. Voraussetzungen

Auf dem Pi werden benötigt:

- Linux mit Python 3
- eine installierte Basler-pylon-Laufzeit mit funktionierendem pypylon
- eine per USB verbundene Basler-Kamera
- FFmpeg und FFprobe für die Goal-Videos
- ein Python-Virtualenv im Projektverzeichnis

FFmpeg prüfen:

~~~bash
command -v ffmpeg
command -v ffprobe
~~~

Falls beide Programme fehlen:

~~~bash
sudo apt-get update
sudo apt-get install ffmpeg
~~~

Die Python-Abhängigkeiten werden aus der zentralen Datei installiert:

~~~bash
.venv/bin/python -m pip install -r requirements.txt
~~~

Die Kamera muss dabei bereits mit pylon erreichbar sein. Ein Python-Paket
allein installiert nicht den Basler-pylon-Treiber.

## 2. Branch auf dem Pi verwenden

Das Projekt liegt auf dem Pi normalerweise unter
/home/dkleis/dev/tischkicker.

Beim ersten Auschecken:

~~~bash
cd /home/dkleis/dev/tischkicker
git fetch origin
git switch --track origin/team/tracking-kicker
~~~

Bei einer bereits vorhandenen lokalen Kopie:

~~~bash
cd /home/dkleis/dev/tischkicker
git switch team/tracking-kicker
git pull --ff-only origin team/tracking-kicker
~~~

Vor einem Update müssen die laufenden Prozesse beendet werden. So wird
verhindert, dass ein alter Prozess die Kamera oder den Shared-Memory-Namen
weiter verwendet.

## 3. Prozesse starten

Die folgenden Befehle werden aus drei Terminals oder aus drei tmux-Fenstern
ausgeführt. Der CameraProcess muss zuerst laufen.

### Terminal 1: CameraProcess

~~~bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m tracking_kicker.camera_process.main \
  --settings tracking_kicker/tuner/settings.json \
  --host 127.0.0.1 \
  --port 5011
~~~

Der Prozess öffnet die Kamera und stellt seine lokale Steuerungs-API auf Port
5011 bereit. Er muss genau einmal laufen, weil nur dieser Prozess pylon und
den USB-Transport besitzt.

### Terminal 2: Tracker

~~~bash
cd /home/dkleis/dev/tischkicker
HEADLESS=1 .venv/bin/python -u -m tracking_kicker.app
~~~

Der Tracker verbindet sich mit dem CameraProcess über den Shared-Memory-Namen
tracking_kicker_camera_frames. Die Tracker-API und das Dashboard laufen auf
Port 5000.

### Terminal 3: optionaler Browser-Tuner

~~~bash
cd /home/dkleis/dev/tischkicker
.venv/bin/python -u -m tracking_kicker.tuner.main \
  --settings tracking_kicker/tuner/settings.json \
  --host 0.0.0.0 \
  --port 5010
~~~

Der Tuner läuft absichtlich auf Port 5010. Port 5011 wird bereits von der
CameraProcess-Steuerungs-API verwendet. Der Tuner öffnet die Kamera nicht
selbst, sondern liest die vom CameraProcess veröffentlichten Bilder.

Auf dem Laptop sind danach erreichbar:

~~~text
http://192.168.10.63:5000/       Tracker-Dashboard
http://192.168.10.63:5010/       Browser-Tuner
~~~

Wenn der Tuner nicht benötigt wird, genügt der CameraProcess zusammen mit dem
Tracker.

### Prozesse beenden

In jedem Terminal mit Ctrl+C beenden. Der CameraProcess stoppt dabei das
Grabbing, schließt die Kamera und entfernt seinen Shared-Memory-Ring.

## 4. Startprüfung

Die CameraProcess-API ist standardmäßig nur auf 127.0.0.1 gebunden. Die
Prüfungen daher direkt auf dem Pi ausführen:

~~~bash
curl -fsS http://127.0.0.1:5011/health | python3 -m json.tool
curl -fsS http://127.0.0.1:5011/api/status | python3 -m json.tool
curl -fsS http://127.0.0.1:5000/api/status | python3 -m json.tool
~~~

Beim CameraProcess sollten mindestens diese Werte stimmen:

~~~text
camera_open: true
grabbing: true
capture_aborted: false
camera_frames_total: zunehmend
last_success_block_id: zunehmend
~~~

Beim Tracker sollte die Antwort unter /api/status kommen und die Quelle als
CameraProcess sichtbar sein. Ein Tracker-Start vor dem CameraProcess führt
typischerweise zu einer fehlenden Shared-Memory-Verbindung; zuerst den
CameraProcess starten und danach den Tracker neu starten.

## 5. APIs

### CameraProcess auf Port 5011

| Methode | Pfad | Zweck |
| --- | --- | --- |
| GET | /health | kurzer Erreichbarkeits- und Kameratest |
| GET | /api/status | Kamera-, Transport-, FPS- und Fehlerzähler |
| GET | /api/logs?after=0&limit=200 | aktuelle Diagnose-Logs |
| POST | /api/settings | Kameraeinstellungen aktualisieren und reconnecten |
| POST | /api/reconnect | Kamera neu verbinden |

Beispiel für einen Reconnect:

~~~bash
curl -fsS -X POST http://127.0.0.1:5011/api/reconnect \
  -H 'Content-Type: application/json' \
  -d '{}'
~~~

### Tracker auf Port 5000

| Methode | Pfad | Zweck |
| --- | --- | --- |
| GET | / | Tracker-Dashboard |
| GET | /live.mjpg | MJPEG-Livebild |
| GET | /api/status | Tracker-, Buffer- und Goalstatus |
| GET | /api/logs?after=0&limit=200 | Tracker-Logs |
| GET | /goal | letztes Goal-Ergebnis oder Verarbeitungsstatus |
| GET | /goal/video/<filename> | fertiges Goal-Video |

### Tuner auf Port 5010

| Methode | Pfad | Zweck |
| --- | --- | --- |
| GET | / | Browser-Oberfläche |
| GET | /health | Tuner- und Kameraerreichbarkeit |
| GET | /api/status | Tuner- und CameraProcess-Status |
| GET | /snapshot.jpg | aktuelles JPEG-Bild |
| GET | /video_feed | Farb-Livebild |
| GET | /ball_feed | Livebild mit Ballerkennung |
| POST | /api/settings | Werte ändern und Kamera reconnecten |
| POST | /api/reconnect | CameraProcess reconnecten |
| POST | /api/ball_detection | Ballerkennung ein- oder ausschalten |
| POST | /api/export_tracker | Ballwerte in die Tracker-Konfiguration übernehmen |

## 6. Goal-Ablauf und Goal-API

Der Tracker hält laufend die letzten drei Sekunden der Rohbilder anhand ihrer
Kamera-Zeitstempel. Bei etwa 180 FPS entspricht das ungefähr 540 Rohframes;
entscheidend ist die Zeitspanne, nicht eine feste Framezahl.

Nach dem Eintritt des Balles in eine Tor-ROI wird der Puffer eingefroren. Nach
der bestehenden Torbestätigung startet die Hintergrundaufbereitung:

1. Die letzten drei Sekunden werden als 90 Farbframes mit 30 FPS ausgewählt.
2. Das letzte Sekundenfenster wird mit der ursprünglichen Kamerarate rückwärts
   durchsucht.
3. Die letzten Ballpositionen werden in umgekehrter Zeitrichtung gesucht, bis
   drei gültige Punkte vorliegen.
4. Aus den drei Zeit-/Weltkoordinaten wird die Geschwindigkeit in km/h berechnet.
5. Das WebM-Video wird kodiert und erst danach optional an DIGIKICKER_POST_URL
   gesendet.

Die laufende Positions- und Geschwindigkeitsanalyse ist deaktiviert. Die
Ballerkennung bleibt aktiv, weil sie den Tor-Kandidaten auslöst.

GET /goal startet die Aufbereitung nicht selbst. Ein bestätigtes Tor startet
die Verarbeitung; der Client fragt anschließend den Status ab:

~~~bash
curl -i http://127.0.0.1:5000/goal
~~~

Die möglichen normalen Zustände:

| HTTP | status | Bedeutung |
| --- | --- | --- |
| 404 | no_goal | Noch kein Tor bestätigt; beim Start normal |
| 202 | preparing | Rückwärtsanalyse oder Video-Kodierung läuft |
| 200 | ready | Analyse, Geschwindigkeit und Video-URL sind fertig |
| 503 | error | Aufbereitung fehlgeschlagen; message prüfen |
| 409 | gone | Angeforderte alte Goal-Generation ist nicht mehr aktuell |

Eine fertige Antwort enthält unter anderem:

- video_url
- video_frame_count: 90
- video_fps: 30
- ball_speed_kmh
- analysis.direction: "backward"
- analysis.points mit timestamp, Pixel- und Weltkoordinaten

Beispiel zum Polling:

~~~bash
while true; do
  curl -s http://127.0.0.1:5000/goal | python3 -m json.tool
  sleep 1
done
~~~

## 7. Payload- und Transportfehler

### Was bedeutet die Meldung?

Die Meldung

~~~text
Payload data has been discarded
~~~

entsteht im CameraProcess, wenn pylon ein ungültiges Grab-Ergebnis meldet
(IsValid() == false). Das ist ein Fehler beim Kamera-/Transportergebnis. Die
Meldung bedeutet nicht automatisch, dass LatestImageOnly falsch ist.

LatestImageOnly darf ältere bereits fertige Frames verwerfen, wenn der
Verbraucher langsamer ist. Das ist in dieser Software die vorgegebene
Produktionsstrategie. Ein ungültiges Payload-Ergebnis wird dagegen als
payload_errors gezählt und der Capture-Loop wird zum Schutz abgebrochen.

### Diagnose in der richtigen Reihenfolge

1. Status und Logs vom CameraProcess abrufen:

   ~~~bash
   curl -s http://127.0.0.1:5011/api/status | python3 -m json.tool
   curl -s 'http://127.0.0.1:5011/api/logs?after=0&limit=200' | python3 -m json.tool
   ~~~

2. In /api/status besonders diese Felder prüfen:

   | Feld | Bedeutung |
   | --- | --- |
   | camera_open | Kamera ist geöffnet |
   | grabbing | pylon-Grabbing läuft |
   | capture_aborted | Capture-Loop wurde nach einem Fehler beendet |
   | failed_grabs | fehlgeschlagene Grab-Versuche |
   | payload_errors | erkannte Payload-/Transportfehler |
   | last_error | letzte vollständige Fehlermeldung |
   | last_failure_phase | Stelle des letzten Fehlers |
   | last_result_diagnostic | pylon-Werte wie error code, block id und Payload-Größe |
   | resulting_fps | von pylon gemeldete tatsächliche Kamerarate |
   | camera_fps | im Capture-Loop gemessene Rate |
   | dropped_frames | Lücken in den pylon-Block-IDs |

3. Einen kontrollierten Reconnect versuchen:

   ~~~bash
   curl -s -X POST http://127.0.0.1:5011/api/reconnect \
     -H 'Content-Type: application/json' -d '{}'
   sleep 2
   curl -s http://127.0.0.1:5011/api/status | python3 -m json.tool
   ~~~

4. Wenn capture_aborted danach wieder true ist, den CameraProcess mit
   Ctrl+C beenden und allein neu starten. Sicherstellen, dass kein zweiter
   CameraProcess, Tuner mit eigenem pylon-Zugriff oder altes Tracker-Programm
   dieselbe Kamera öffnet.

5. Wenn der Fehler wiederkommt, die Hardware und den Transport prüfen:

   - USB-3-Port und USB-3-Kabel verwenden
   - Kamera und Pi ausreichend versorgen
   - pylon-Gerät und Seriennummer prüfen
   - ROI und PixelFormat nicht zwischen CameraProcess und Tracker mischen
   - CPU- und USB-Auslastung während des Fehlers notieren
   - last_result_diagnostic mit Zeitpunkt und resulting_fps sichern

Die Werte max_transfer_size, num_max_queued_urbs und weitere
Transportparameter sollten nicht blind geändert werden. Erst den Status und
die Logs sichern, dann jeweils nur einen Wert ändern und erneut testen.

### Was bei einem Payloadfehler nicht geändert werden soll

- grab_strategy nicht auf OneByOne umstellen. Die Produktionslogik verlangt
  latest_image_only; die Einstellungs-API weist andere Werte zurück.
- Die tatsächliche Kamerarate nicht aus force_fps ableiten. requested_fps ist
  der Konfigurationswert, resulting_fps ist der von pylon gemeldete Wert.
- Den Tracker nicht mehrfach starten. Mehrere Tracker können zwar denselben
  Frame-Ring lesen, erschweren aber die Diagnose und können doppelte Goal-Events
  erzeugen.

### Häufige verwandte Fehler

| Meldung oder Zustand | Maßnahme |
| --- | --- |
| Keine Basler-Kamera gefunden | USB, pylon, Zugriffsrechte und Seriennummer prüfen |
| camera_open: false | CameraProcess-Log lesen, danach /api/reconnect |
| capture_aborted: true | CameraProcess vollständig neu starten |
| SharedMemory nicht gefunden | CameraProcess zuerst starten, gleichen IPC-Namen verwenden |
| Framegröße passt nicht | width/height in den beiden Konfigurationen abgleichen |
| /goal liefert 404 | Noch kein bestätigtes Tor; das ist beim Start normal |
| /goal liefert 202 | Aufbereitung läuft; weiter pollen |
| /goal liefert 503 | message, Tracker-Logs und FFmpeg-Installation prüfen |

## 8. Konfiguration

| Datei | Verantwortlich für |
| --- | --- |
| tracking_kicker/tuner/settings.json | CameraProcess, Kamera-ROI, PixelFormat, USB- und Tunerwerte |
| tracking_kicker/config.json | Tracker-Laufzeit, Ballerkennung und CameraProcess-IPC |
| tracking_kicker/vision_config.json | Tor-ROIs, Spielfeldgröße, Spielfeldecken und Stangen |
| tracking_kicker/table_calibration.json | gespeicherte Tischkalibrierung |
| tracking_kicker/lens_config.json | Objektivkorrekturwerte |

Die wichtige Kamera-Basis ist:

- pixel_format: BayerRG8
- ROI 1100 x 650
- Offset x=240, y=266
- grab_strategy: latest_image_only
- Shared-Memory-Name: tracking_kicker_camera_frames

Die tatsächliche Kamerarate immer über /api/status mit resulting_fps und
camera_fps prüfen. force_fps allein ist kein Nachweis für die resultierende
Rate; bei deaktivierter Acquisition-Rate-Regelung wird dieser Wert nicht als
harte Kamerabegrenzung geschrieben.

Generierte Videos liegen unter tracking_kicker/videos/ und werden nicht in
Git eingecheckt. Die aktuelle Goal-Aufbereitung erzeugt genau 90 Farbframes in
30 FPS.

## 9. Optionaler digiKicker-POST

Nach fertiger Analyse und Videokodierung kann der Tracker das Ergebnis an eine
externe URL senden. Die Werte werden über Umgebungsvariablen gesetzt:

~~~bash
export DIGIKICKER_POST_URL="https://example.invalid/goal"
export DIGIKICKER_POST_VERIFY_TLS=1
export DIGIKICKER_POST_TIMEOUT=2
~~~

Der POST läuft in einem eigenen Thread und blockiert die Kamera nicht. Ein
fehlgeschlagener POST macht die lokale Goal-Analyse nicht rückgängig; die
Meldung steht im Tracker-Log. Keine Passwörter, Tokens oder privaten Schlüssel
in README, JSON-Dateien oder Git speichern.

## 10. Entwicklung und Tests

Aus dem Projektverzeichnis:

~~~bash
python -m unittest discover -s tests -v
python -m compileall -q tracking_kicker tests tools
git diff --check
~~~

Der Goal-API-Test prüft:

- 404 ohne bestätigtes Tor
- 202 während der Aufbereitung
- 200 mit Geschwindigkeit, drei Analysepunkten und Video-URL
- 90 Frames bei 30 FPS
- Rückwärtsanalyse im letzten Sekundenfenster

## 11. Änderungen mit dem Team teilen

Der Team-Branch wird aktualisiert mit:

~~~bash
git switch team/tracking-kicker
git pull --ff-only origin team/tracking-kicker
~~~

Eigene Änderungen teilen:

~~~bash
git add .
git commit -m "Kurze Beschreibung der Änderung"
git push origin team/tracking-kicker
~~~

Für die produktive Live-Version ist ein separater, geprüfter Push nach live
erforderlich. Der Team-Branch und live sollten deshalb nicht verwechselt
werden.
