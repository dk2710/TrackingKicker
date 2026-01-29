import socket
import struct
from queue import Queue
import threading

class SocketJpegStreamer:
    def __init__(self, host="0.0.0.0", port=9999, jpeg_quality=75, send_fps=60):
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.send_fps = float(send_fps)

    def run(self, display_queue: Queue, stop_event: threading.Event):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"🌐 Streamer wartet auf Client: {self.host}:{self.port}")

        srv.settimeout(1.0)
        conn = None
        try:
            while not stop_event.is_set():
                try:
                    conn, addr = srv.accept()
                    print("✅ Stream-Client verbunden:", addr)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    break
                except socket.timeout:
                    continue

            if conn is None:
                return

            dt = 1.0 / self.send_fps
            t_next = time.perf_counter()

            while not stop_event.is_set():
                # fps pacing
                now = time.perf_counter()
                if now < t_next:
                    time.sleep(max(0, t_next - now))
                t_next += dt

                # latest-only aus display_queue ziehen (alles alte wegwerfen)
                frame = None
                try:
                    frame = display_queue.get(timeout=0.2)
                    while True:
                        frame = display_queue.get_nowait()
                except:
                    pass

                if frame is None:
                    continue

                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if not ok:
                    continue

                payload = jpg.tobytes()
                conn.sendall(struct.pack("!I", len(payload)))
                conn.sendall(payload)

        except Exception as e:
            print("⚠️ Streamer stopped:", e)
        finally:
            try:
                if conn:
                    conn.close()
            except:
                pass
            srv.close()
            print("🛑 Streamer beendet")
