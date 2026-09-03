"""Convert camera pixel formats to the BGR images used by OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def to_bgr(image: np.ndarray, pixel_format: str = "BayerRG8") -> np.ndarray:
    """Return an OpenCV BGR image for a configured pylon pixel format.

    Basler Bayer formats arrive as a two-dimensional array.  Replay files may
    already contain BGR images, so three-channel input is kept unchanged
    unless the configured format explicitly says RGB.
    """

    if image is None:
        return image

    fmt = str(pixel_format or "BayerRG8").replace("_", "").upper()
    if image.ndim == 2:
        conversion_codes = {
            "BAYERRG": cv2.COLOR_BayerRG2BGR,
            "BAYERBG": cv2.COLOR_BayerBG2BGR,
            "BAYERGB": cv2.COLOR_BayerGB2BGR,
            "BAYERGR": cv2.COLOR_BayerGR2BGR,
        }
        for prefix, code in conversion_codes.items():
            if fmt.startswith(prefix):
                return cv2.cvtColor(image, code)

        if fmt.startswith("MONO"):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Unbekanntes zweidimensionales Pixel-Format: {pixel_format!r}")

    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3 and fmt.startswith("RGB"):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 3 and image.shape[2] == 4 and fmt.startswith("RGBA"):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image
