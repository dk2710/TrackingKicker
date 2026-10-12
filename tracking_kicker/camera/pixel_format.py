"""Helpers for validating Basler pixel-format enum values."""

from __future__ import annotations

from typing import Any


def normalize_pixel_format(value: Any) -> str:
    """Normalize a pylon pixel-format name for comparisons."""

    return str(value or "").strip().replace("_", "").upper()


def pixel_format_family(value: Any) -> str:
    """Return the sensor/color family, independent of bit depth."""

    normalized = normalize_pixel_format(value)
    for family in (
        "BAYERRG",
        "BAYERBG",
        "BAYERGR",
        "BAYERGB",
        "MONO",
        "RGB",
        "BGR",
        "YUV",
    ):
        if normalized.startswith(family):
            return family
    return normalized


def available_pixel_formats(camera: Any) -> tuple[str, ...]:
    """Read the PixelFormat enum values exposed by a pylon camera.

    pypylon versions expose the values either as ``Symbolics`` or through
    ``GetSymbolics()``.  An empty result means that this particular pypylon
    wrapper did not expose the enum list; configuration can still proceed and
    pylon will perform its normal validation when the value is written.
    """

    try:
        node = getattr(camera, "PixelFormat")
    except Exception:
        return ()

    for accessor in ("Symbolics", "GetSymbolics"):
        try:
            values = getattr(node, accessor)
            values = values() if callable(values) else values
            if values is None:
                continue
            if isinstance(values, str):
                return (values,)
            return tuple(str(value) for value in values)
        except Exception:
            continue
    return ()


def resolve_pixel_format(camera: Any, requested: Any) -> tuple[str, tuple[str, ...]]:
    """Validate and return the camera's spelling of a requested format.

    A Bayer format with another bit depth is deliberately not selected
    automatically.  The raw frame's format is also used later for OpenCV
    conversion, so silently changing it would make the capture and consumer
    disagree about the image data.
    """

    requested_name = str(requested or "BayerRG8").strip() or "BayerRG8"
    available = available_pixel_formats(camera)
    if not available:
        return requested_name, available

    normalized_requested = normalize_pixel_format(requested_name)
    selected = next(
        (
            value
            for value in available
            if normalize_pixel_format(value) == normalized_requested
        ),
        None,
    )
    if selected is not None:
        return selected, available

    family = pixel_format_family(requested_name)
    same_family = [
        value for value in available if pixel_format_family(value) == family
    ]
    details = f"Verfuegbar: {', '.join(available)}"
    if same_family and family.startswith("BAYER"):
        details += (
            ". Gleiches Bayer-Muster mit anderer Bittiefe vorhanden: "
            + ", ".join(same_family)
        )
    raise RuntimeError(
        f"PixelFormat {requested_name!r} wird von der Kamera nicht unterstuetzt. "
        f"{details}"
    )
