"""Camera input package.

The Basler implementation is loaded lazily so replay-only installations stay
importable even when pypylon is not installed.
"""

from .Camera import Camera
from .conversion import to_bgr
__all__ = [
    "BaslerSource",
    "Camera",
    "CameraProcessSource",
    "to_bgr",
]


def __getattr__(name):
    if name == "BaslerSource":
        # Keep the historical import path while exposing the one maintained
        # direct Basler implementation in the package root.
        from ..BaslerSource import BaslerSource

        return BaslerSource
    if name == "CameraProcessSource":
        from .CameraProcessSource import CameraProcessSource

        return CameraProcessSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
