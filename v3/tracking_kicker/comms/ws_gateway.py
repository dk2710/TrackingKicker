# -*- coding: utf-8 -*-
import asyncio
import json
from typing import Dict, Set, Any
import websockets

class WSGateway:
    """
    Minimaler WebSocket-Gateway:
    - Textnachrichten (JSON) für Status/Signale (replay_start/replay_end)
    - Binärnachrichten (bytes, JPEG) für Frames
    Authentifizierung per Query-Parameter ?token=
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8777, token: str = "SUPERSECRET"):
        self.host = host
        self.port = port
        self.token = token
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def _handler(self, ws: websockets.WebSocketServerProtocol):
        # Simple Auth via Query-Parameter
        if "?" in ws.path:
            try:
                q = ws.path.split("?", 1)[1]
                pairs = [p.split("=", 1) for p in q.split("&") if "=" in p]
                qs = {k: v for k, v in pairs}
            except Exception:
                qs = {}
        else:
            qs = {}

        if qs.get("token") != self.token:
            await ws.close(code=4401, reason="unauthorized")
            return

        self.clients.add(ws)
        try:
            async for _ in ws:
                # Dieser Server ist primär "Server -> Client".
                # Eingehende Nachrichten können hier optional ausgewertet werden.
                pass
        finally:
            self.clients.discard(ws)

    async def start(self):
        async with websockets.serve(self._handler, self.host, self.port, max_size=16 * 1024 * 1024):
            print(f"[WS] listening on ws://{self.host}:{self.port}/  (token protected)")
            await asyncio.Future()  # run forever

    async def broadcast_json(self, payload: Dict[str, Any]):
        if not self.clients:
            return
        msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.clients.discard(d)
            await self._safe_close(d)

    async def broadcast_frame(self, jpg_bytes: bytes):
        if not self.clients:
            return
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(jpg_bytes)  # Binary-Message
            except Exception:
                dead.append(ws)
        for d in dead:
            self.clients.discard(d)
            await self._safe_close(d)

    async def _safe_close(self, ws):
        try:
            await ws.close()
        except:
            pass