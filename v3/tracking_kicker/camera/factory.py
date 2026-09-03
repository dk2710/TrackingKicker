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
    try:
        from .BaslerSource import BaslerSource
    except ModuleNotFoundError as exc:
        if exc.name == "pypylon":
            raise RuntimeError(
                "pypylon ist nicht installiert; Basler-Kamera kann nicht verwendet werden"
            ) from exc
        raise

    attempts = max(1, int(attempts))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return BaslerSource(camera_config)
        except ModuleNotFoundError as exc:
            # BaslerSource checks pypylon lazily because replay mode must stay
            # usable without the vendor package. Do not spend all retry
            # attempts on a package that is definitely not installed.
            if exc.name == "pypylon" or "pypylon" in str(exc).lower():
                raise RuntimeError(
                    "pypylon ist nicht installiert; Basler-Kamera kann nicht verwendet werden"
                ) from exc
            last_error = exc
            print(
                f"[CAM-START] Kamera nicht verfuegbar "
                f"(Versuch {attempt}/{attempts}): {exc}"
            )
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_delay_seconds)))
        except Exception as exc:
            last_error = exc
            print(
                f"[CAM-START] Kamera nicht verfuegbar "
                f"(Versuch {attempt}/{attempts}): {exc}"
            )
            # A USB2 fallback is deterministic and cannot deliver the
            # requested high-rate stream. Retrying pylon against that same
            # link only adds noise and can disturb a marginal USB state.
            message = str(exc).lower()
            if "usb2" in message and "usb3" in message:
                raise RuntimeError(str(exc)) from exc
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_delay_seconds)))

    raise RuntimeError(
        f"Basler-Kamera konnte nach {attempts} Versuchen nicht geoeffnet werden: "
        f"{last_error}"
    ) from last_error
