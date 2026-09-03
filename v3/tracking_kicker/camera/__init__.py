"""Camera input package.

The Basler implementation is loaded lazily so replay-only installations stay
importable even when pypylon is not installed.
"""

from .Camera import Camera
from .conversion import to_bgr
from .RawReplaySource import RawReplaySource

__all__ = ["BaslerSource", "Camera", "RawReplaySource", "to_bgr"]


def __getattr__(name):
    if name == "BaslerSource":
        from .BaslerSource import BaslerSource

        return BaslerSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
