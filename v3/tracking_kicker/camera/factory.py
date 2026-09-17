"""Factories for physical camera sources."""

from __future__ import annotations

import time


def create_camera_with_retry(
    camera_config,
    *,
    attempts: int = 15,
    retry_delay_seconds: float = 2.0,
):
    """Open the configured Basler camera, retrying transient USB failures."""

    if camera_config is None:
        raise ValueError("Eine Kamerakonfiguration ist erforderlich")

    # Lazy import keeps replay-only mode independent of the pypylon package.
    source = str(getattr(camera_config, "source", "camera_process")).lower()
    if source in {"camera_process", "camera-process", "shared_memory"}:
        source_type = "CameraProcessSource"
        try:
            from .CameraProcessSource import CameraProcessSource
        except ModuleNotFoundError as exc:
            if exc.name == "numpy":
                raise RuntimeError(
                    "numpy ist fuer den CameraProcess-Framezugriff nicht installiert"
                ) from exc
            raise
        source_class = CameraProcessSource
    else:
        source_type = "BaslerSource"
        try:
            from .BaslerSource import BaslerSource
        except ModuleNotFoundError as exc:
            if exc.name == "pypylon":
                raise RuntimeError(
                    "pypylon ist nicht installiert; Basler-Kamera kann nicht verwendet werden"
                ) from exc
            raise
        source_class = BaslerSource

    attempts = max(1, int(attempts))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return source_class(camera_config)
        except Exception as exc:
            last_error = exc
            print(
                f"[CAM-START] Kamera nicht verfuegbar "
                f"{source_type} nicht verfuegbar "
                f"(Versuch {attempt}/{attempts}): {exc}"
            )
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_delay_seconds)))

    raise RuntimeError(
        f"Basler-Kamera konnte nach {attempts} Versuchen nicht geoeffnet werden: "
        f"{last_error}"
    ) from last_error
