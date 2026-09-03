"""Small OpenCV GUI adapter that is safe in headless deployments."""

import cv2

from ..runtime import HEADLESS


def named_window(name: str) -> None:
    if not HEADLESS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)


def imshow(name: str, frame) -> None:
    if not HEADLESS:
        cv2.imshow(name, frame)


def wait_key(ms: int = 1) -> int:
    if not HEADLESS:
        return cv2.waitKey(ms) & 0xFF
    return 255


def destroy_window(name: str) -> None:
    if not HEADLESS:
        cv2.destroyWindow(name)


def destroy_all() -> None:
    if not HEADLESS:
        cv2.destroyAllWindows()
