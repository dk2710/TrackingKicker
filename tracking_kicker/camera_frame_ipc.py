"""A latest-frame mailbox shared by the camera and tuner processes.

The mailbox deliberately keeps only the current frame. Three fixed slots are
used only so a producer can write one slot while a consumer copies another.
Slot state changes use a short inter-process ``flock``; the lock is never held
while copying the image, so a slow consumer cannot stall the USB grab loop.
"""

from __future__ import annotations

import re
import struct
import threading
import time
from contextlib import contextmanager
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
import tempfile
from typing import Iterator, Optional

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - local Windows fallback
    fcntl = None


MAGIC = b"TKCAMR01"
VERSION = 2
DEFAULT_NAME = "tracking_kicker_camera_frames"
DEFAULT_SLOTS = 3
HEADER_BYTES = 64
HEADER = struct.Struct("<8sIIIIIIQ")
SLOT_META = struct.Struct("<IQd")  # state, sequence, timestamp
SLOT_META_BYTES = SLOT_META.size

FREE = 0
WRITING = 1
READY = 2
READING = 3


def _untrack_reader(shm: shared_memory.SharedMemory) -> None:
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except (AttributeError, KeyError, ModuleNotFoundError, OSError, RuntimeError):
        pass


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name, create=False)
        _untrack_reader(shm)
        return shm


def _lock_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))
    lock_directory = Path("/tmp") if fcntl is not None else Path(tempfile.gettempdir())
    return lock_directory / f"{safe}.mailbox.lock"


class SharedFrameMailbox:
    """Single-producer/latest-frame mailbox backed by shared memory."""

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        *,
        name: str,
        width: int,
        height: int,
        slots: int,
        owner: bool,
    ) -> None:
        self.shm = shm
        self.name = str(name)
        self.width = int(width)
        self.height = int(height)
        self.slots = 3
        self.slot_bytes = self.width * self.height
        self.owner = bool(owner)
        self._closed = False
        self._local_lock = threading.RLock()
        self._lock_handle = open(_lock_path(self.name), "a+b")
        self._publish_drops = 0
        self._read_claims = 0
        self._read_releases = 0

    @classmethod
    def create_or_replace(
        cls,
        name: str = DEFAULT_NAME,
        *,
        width: int,
        height: int,
        slots: int = DEFAULT_SLOTS,
    ) -> "SharedFrameMailbox":
        width = int(width)
        height = int(height)
        slots = 3
        size = HEADER_BYTES + slots * (SLOT_META_BYTES + width * height)
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            stale = _attach_shared_memory(name)
            try:
                stale.unlink()
            finally:
                stale.close()
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        mailbox = cls(
            shm,
            name=name,
            width=width,
            height=height,
            slots=slots,
            owner=True,
        )
        HEADER.pack_into(
            shm.buf, 0, MAGIC, VERSION, width, height, slots, mailbox.slot_bytes, 0, 0
        )
        for slot in range(slots):
            mailbox._write_slot_meta(slot, FREE, 0, 0.0)
        return mailbox

    @classmethod
    def open(cls, name: str = DEFAULT_NAME) -> "SharedFrameMailbox":
        shm = _attach_shared_memory(name)
        try:
            magic, version, width, height, slots, slot_bytes, _reserved, _latest = (
                HEADER.unpack_from(shm.buf, 0)
            )
            if magic != MAGIC or version != VERSION:
                raise RuntimeError("Unbekanntes CameraProcess-Frameformat")
            if width <= 0 or height <= 0 or slots != 3 or slot_bytes != width * height:
                raise RuntimeError("Ungueltige CameraProcess-Mailboxgroesse")
            return cls(
                shm,
                name=name,
                width=width,
                height=height,
                slots=slots,
                owner=False,
            )
        except Exception:
            shm.close()
            raise

    def _slot_offset(self, slot: int) -> int:
        return HEADER_BYTES + int(slot) * (SLOT_META_BYTES + self.slot_bytes)

    def _payload_offset(self, slot: int) -> int:
        return self._slot_offset(slot) + SLOT_META_BYTES

    def _read_slot_meta(self, slot: int):
        return SLOT_META.unpack_from(self.shm.buf, self._slot_offset(slot))

    def _write_slot_meta(self, slot: int, state: int, sequence: int, timestamp: float) -> None:
        SLOT_META.pack_into(
            self.shm.buf,
            self._slot_offset(slot),
            int(state),
            int(sequence),
            float(timestamp),
        )

    def _read_latest(self) -> int:
        return int(struct.unpack_from("<Q", self.shm.buf, HEADER.size - 8)[0])

    def _write_latest(self, sequence: int) -> None:
        struct.pack_into("<Q", self.shm.buf, HEADER.size - 8, int(sequence))

    @contextmanager
    def _metadata_lock(self) -> Iterator[None]:
        """Protect metadata only; never hold this while copying pixels."""

        if fcntl is None:  # pragma: no cover - local Windows fallback
            with self._local_lock:
                yield
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)

    def _find_writable_slot(self, latest: int) -> Optional[int]:
        superseded = []
        for slot in range(self.slots):
            state, sequence, _timestamp = self._read_slot_meta(slot)
            if state == FREE:
                return slot
            if state == READY and sequence != latest:
                superseded.append((sequence, slot))
        return min(superseded)[1] if superseded else None

    def publish(self, image: np.ndarray, timestamp: Optional[float] = None) -> Optional[int]:
        """Publish without waiting; ``None`` means an intentional latest-only drop."""

        if self._closed:
            raise RuntimeError("Frame-Mailbox ist geschlossen")
        array = np.asarray(image)
        if array.shape != (self.height, self.width) or array.dtype != np.uint8:
            raise ValueError(
                f"Erwartet uint8-Frame {(self.height, self.width)}, "
                f"erhalten {array.shape} {array.dtype}"
            )
        now = float(timestamp or time.monotonic())
        with self._metadata_lock():
            latest = self._read_latest()
            slot = self._find_writable_slot(latest)
            if slot is None:
                self._publish_drops += 1
                return None
            sequence = latest + 1
            self._write_slot_meta(slot, WRITING, sequence, now)

        # The expensive frame copy is outside the inter-process lock.
        target = np.ndarray(
            (self.height, self.width),
            dtype=np.uint8,
            buffer=self.shm.buf,
            offset=self._payload_offset(slot),
        )
        target[:, :] = array

        with self._metadata_lock():
            state, current_sequence, _ = self._read_slot_meta(slot)
            if state != WRITING or current_sequence != sequence:
                self._publish_drops += 1
                return None
            for old_slot in range(self.slots):
                old_state, old_sequence, old_timestamp = self._read_slot_meta(old_slot)
                if old_state == READY and old_sequence == latest:
                    self._write_slot_meta(old_slot, FREE, 0, old_timestamp)
            self._write_slot_meta(slot, READY, sequence, now)
            self._write_latest(sequence)
        return sequence

    def latest_sequence(self) -> int:
        with self._metadata_lock():
            return self._read_latest()

    def _claim_latest(self, last_sequence: int):
        with self._metadata_lock():
            latest = self._read_latest()
            if latest <= int(last_sequence):
                return None
            for slot in range(self.slots):
                state, sequence, timestamp = self._read_slot_meta(slot)
                if state == READY and sequence == latest:
                    self._write_slot_meta(slot, READING, sequence, timestamp)
                    self._read_claims += 1
                    return slot, sequence, timestamp
        return None

    def read_latest(self, last_sequence: int):
        """Copy the newest frame while preventing producer overwrites."""

        claim = self._claim_latest(last_sequence)
        if claim is None:
            return None
        slot, sequence, timestamp = claim
        try:
            source = np.ndarray(
                (self.height, self.width),
                dtype=np.uint8,
                buffer=self.shm.buf,
                offset=self._payload_offset(slot),
            )
            return sequence, timestamp, source.copy()
        finally:
            with self._metadata_lock():
                state, current_sequence, current_timestamp = self._read_slot_meta(slot)
                if state == READING and current_sequence == sequence:
                    self._write_slot_meta(slot, FREE, 0, current_timestamp)
                    self._read_releases += 1

    def read_next(self, last_sequence: int):
        return self.read_latest(last_sequence)

    def stats(self):
        with self._metadata_lock():
            states = [self._read_slot_meta(slot)[0] for slot in range(self.slots)]
        return {
            "slots": self.slots,
            "states": states,
            "publish_drops": self._publish_drops,
            "read_claims": self._read_claims,
            "read_releases": self._read_releases,
        }

    def close(self, *, unlink: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        finally:
            self._lock_handle.close()
        if unlink and self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


# Compatibility name for existing imports during migration.
SharedFrameRing = SharedFrameMailbox
