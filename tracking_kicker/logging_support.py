"""Timestamped runtime logs shared by the tracker and its web dashboard."""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Any


_LOCK = threading.RLock()
_RECORDS: deque[dict[str, Any]] = deque(maxlen=500)
_NEXT_ID = 1
_HANDLER: logging.Handler | None = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None

_ERROR_MARKERS = (
    "-ERR]",
    "ERROR",
    "FEHLER",
    "FEHLGESCHLAGEN",
    "ABGEBROCHEN",
    "PAYLOAD DATA HAS BEEN DISCARDED",
    "PAYLOAD-/TRANSPORTFEHLER",
)


class _MemoryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _NEXT_ID
        try:
            message = self.format(record)
            with _LOCK:
                _RECORDS.append(
                    {
                        "id": _NEXT_ID,
                        "text": message,
                        "time": datetime.fromtimestamp(
                            record.created
                        ).astimezone().isoformat(timespec="seconds"),
                        "level": record.levelname,
                    }
                )
                _NEXT_ID += 1
        except Exception:
            self.handleError(record)


class _StreamToLogger:
    """Turn existing print/traceback output into timestamped log records."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level

    def write(self, value: str) -> int:
        for line in str(value).splitlines():
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(str(value))

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class _SafeTextStream:
    """Keep logging usable when a Windows console cannot encode an emoji."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, value: str) -> int:
        try:
            return self._stream.write(value)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "utf-8"
            safe = value.encode(encoding, errors="replace").decode(encoding)
            return self._stream.write(safe)

    def flush(self) -> None:
        self._stream.flush()


def configure_runtime_logging() -> None:
    """Install one process-wide handler and capture legacy print output."""

    global _HANDLER, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    with _LOCK:
        if _HANDLER is not None:
            return

        _ORIGINAL_STDOUT = sys.stdout
        _ORIGINAL_STDERR = sys.stderr
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        memory = _MemoryHandler()
        memory.setFormatter(formatter)
        stream = logging.StreamHandler(_SafeTextStream(_ORIGINAL_STDOUT))
        stream.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(memory)
        root.addHandler(stream)
        _HANDLER = memory

        stdout_logger = logging.getLogger("tracking_kicker.stdout")
        stderr_logger = logging.getLogger("tracking_kicker.stderr")
        sys.stdout = _StreamToLogger(stdout_logger, logging.INFO)
        sys.stderr = _StreamToLogger(stderr_logger, logging.ERROR)

        logging.getLogger("tracking_kicker").info(
            "Zeitstempel-Logging aktiviert; Web-Logpuffer bereit"
        )


def runtime_logs(
    *,
    after: int = 0,
    limit: int = 200,
    errors_only: bool = False,
    include_diagnostics: bool = False,
) -> list[dict[str, Any]]:
    """Return new dashboard log entries after an optional sequence number.

    The web dashboard can request a compact view. Legacy ``print`` output is
    recorded at INFO, so the explicit error markers are included there as
    well as normal ERROR and CRITICAL records.
    """

    after = max(0, int(after))
    limit = max(1, min(int(limit), 500))
    with _LOCK:
        entries = []
        for item in _RECORDS:
            if int(item["id"]) <= after:
                continue
            if errors_only and not _is_dashboard_error(item):
                if not (include_diagnostics and _is_diagnostic(item)):
                    continue
            entries.append(item.copy())
    return entries[-limit:]


def _is_dashboard_error(item: dict[str, Any]) -> bool:
    level = str(item.get("level", "")).upper()
    if level in {"ERROR", "CRITICAL"}:
        return True
    text = str(item.get("text", "")).upper()
    return any(marker in text for marker in _ERROR_MARKERS)


def _is_diagnostic(item: dict[str, Any]) -> bool:
    return "[CAMERA-DIAG]" in str(item.get("text", ""))


def latest_log_id() -> int:
    with _LOCK:
        return int(_RECORDS[-1]["id"]) if _RECORDS else 0
