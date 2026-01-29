import socket, struct, threading
import numpy as np
import cv2
import time

PI_IP = "172.20.10.6"  # <-- Pi-IP
PORT = 9999

latest = None
lock = threading.Lock()
stop = False

def recvn(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Disconnected")
        buf += chunk
    return buf

def reader(sock):
    global latest, stop
    try:
        while not stop:
            (ln,) = struct.unpack("!I", recvn(sock, 4))
            data = recvn(sock, ln)
            with lock:
                latest = data
    except Exception as e:
        print("Reader stopped:", e)
        stop = True

def main():
    global stop
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, PORT))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    threading.Thread(target=reader, args=(sock,), daemon=True).start()

    while not stop:
        with lock:
            data = latest
        if data is None:
            cv2.waitKey(1)
            continue

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        cv2.imshow("Kicker Overlay (Socket)", img)
        if cv2.waitKey(1) == 27:  # ESC
            stop = True

    sock.close()

if __name__ == "__main__":
    main()
