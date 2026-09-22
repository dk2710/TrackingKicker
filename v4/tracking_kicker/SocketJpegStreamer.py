import socket
import struct
from queue import Queue
import threading
import time
import cv2

class SocketJpegStreamer:
    def __init__(self, host="0.0.0.0", port=9999, jpeg_quality=75, send_fps=60):
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.send_fps = float(send_fps)

    def run(self, display_queue: Queue, stop_event: threading.Event):
        import socket

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)

        print(f"🌐 HTTP MJPEG Stream läuft: http://localhost:{self.port}")

        conn, addr = srv.accept()
        print("✅ Client verbunden:", addr)

        # ✅ HTTP HEADER!!!!
        conn.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        )

        while not stop_event.is_set():
            try:
                frame = display_queue.get(timeout=1)
            except:
                continue

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue

            conn.sendall(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
