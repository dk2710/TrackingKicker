"""pypylon capture process used by :mod:`BaslerSource`.

The pylon transport layer is deliberately kept in a separate process.  A
blocked ``RetrieveResult`` or ``StopGrabbing`` can therefore be terminated by
the parent without leaving the application worker in a half-closed state.
Frames are written to a shared-memory ring supplied by the parent; control
messages never travel through that high-rate path.
"""

from __future__ import annotations

import queue
import time
from typing import Any

import numpy as np


FAILURE_THRESHOLD = 5
FAILURE_MIN_DURATION_SECONDS = 0.5
RETRIEVE_TIMEOUT_MS = 100
STARTUP_SETTLE_SECONDS = 2.0


def _emit(events, kind: str, message: str) -> None:
    if events is None:
        print(f"[CAMERA-WARN] {message}", flush=True)
        return
    try:
        events.put_nowait((kind, str(message)))
    except (queue.Full, OSError):
        pass


def _set_feature(camera, name: str, value, *, required: bool) -> bool:
    try:
        node = getattr(camera, name)
        setter = getattr(node, "SetValue", None)
        if callable(setter):
            setter(value)
        else:
            node.Value = value
        return True
    except Exception as exc:
        # Some USB camera states expose an already-correct value as read-only
        # during a reconnect. In that case there is nothing to change. This
        # is especially common for AcquisitionMode after a transport fault.
        try:
            current = _get_feature(camera, name)
            if current is not None and str(current) == str(value):
                return True
        except Exception:
            pass
        message = f"{name}={value!r} konnte nicht gesetzt werden: {exc}"
        if required:
            raise RuntimeError(message) from exc
        _emit(None, "warning", message)
        return False


def _get_feature(camera, name: str):
    try:
        node = getattr(camera, name)
        getter = getattr(node, "GetValue", None)
        return getter() if callable(getter) else node.Value
    except Exception:
        return None


def _device_serial(device_info) -> str:
    try:
        return str(device_info.GetSerialNumber())
    except Exception:
        return ""


def _device_label(device_info) -> str:
    values = []
    for method_name in ("GetModelName", "GetSerialNumber", "GetDeviceClass"):
        try:
            value = getattr(device_info, method_name)()
        except Exception:
            continue
        if value:
            values.append(str(value))
    return " / ".join(values) or "unbekanntes Basler-Geraet"


def _select_device_info(factory, cfg):
    devices = list(factory.EnumerateDevices())
    if not devices:
        raise RuntimeError(
            "Keine Basler-Kamera gefunden (pypylon EnumerateDevices=0)."
        )

    wanted_serial = str(getattr(cfg, "serial_number", "") or "").strip()
    if not wanted_serial:
        return devices[0]

    for device_info in devices:
        if _device_serial(device_info) == wanted_serial:
            return device_info

    available = ", ".join(_device_serial(info) for info in devices)
    raise RuntimeError(
        f"Basler-Kamera mit Seriennummer {wanted_serial!r} nicht gefunden. "
        f"Verfuegbare Seriennummern: {available or 'unbekannt'}"
    )


def _configure_camera(camera, cfg) -> None:
    _set_feature(camera, "TriggerSelector", "FrameStart", required=False)
    _set_feature(camera, "TriggerMode", "Off", required=True)
    _set_feature(camera, "AcquisitionMode", "Continuous", required=True)

    _set_feature(camera, "ExposureAuto", "Off", required=True)
    _set_feature(camera, "GainAuto", "Off", required=True)
    _set_feature(
        camera,
        "PixelFormat",
        getattr(cfg, "pixel_format", "BayerRG8"),
        required=True,
    )
    _set_feature(camera, "ExposureTime", float(cfg.exposure_time), required=True)
    _set_feature(camera, "Gain", float(cfg.gain), required=True)

    _set_feature(camera, "OffsetX", 0, required=True)
    _set_feature(camera, "OffsetY", 0, required=True)
    _set_feature(camera, "Width", int(cfg.width), required=True)
    _set_feature(camera, "Height", int(cfg.height), required=True)
    _set_feature(camera, "OffsetX", int(cfg.offset_x), required=True)
    _set_feature(camera, "OffsetY", int(cfg.offset_y), required=True)

    max_buffers = max(1, int(getattr(cfg, "max_num_buffer", 256)))
    try:
        camera.MaxNumBuffer = max_buffers
        print(f"pylon MaxNumBuffer = {max_buffers}", flush=True)
    except Exception as exc:
        print(f"[CAMERA-WARN] MaxNumBuffer nicht setzbar: {exc}", flush=True)

    requested_fps = float(getattr(cfg, "force_fps", 0) or 0)
    if requested_fps > 0:
        _set_feature(camera, "AcquisitionFrameRateEnable", True, required=False)
        _set_feature(camera, "AcquisitionFrameRate", requested_fps, required=False)


def _open_camera(cfg, pylon):
    factory = pylon.TlFactory.GetInstance()
    device_info = _select_device_info(factory, cfg)
    camera = pylon.InstantCamera(factory.CreateDevice(device_info))
    # Do not let pylon's default AcquireContinuousConfiguration write
    # AcquisitionMode during OnOpened. After a USB transport fault the node
    # can already be Continuous but temporarily read-only; our configuration
    # below can then safely accept that existing value.
    camera.RegisterConfiguration(
        None,
        pylon.RegistrationMode_ReplaceAll,
        pylon.Cleanup_None,
    )
    camera.Open()
    _configure_camera(camera, cfg)
    strategy = (
        pylon.GrabStrategy_LatestImageOnly
        if getattr(cfg, "grab_strategy_latest_only", False)
        else pylon.GrabStrategy_OneByOne
    )
    camera.StartGrabbing(strategy)
    return camera, device_info


def _settle_stream(camera, pylon) -> None:
    deadline = time.perf_counter() + STARTUP_SETTLE_SECONDS
    while time.perf_counter() < deadline:
        result = None
        try:
            result = camera.RetrieveResult(
                RETRIEVE_TIMEOUT_MS,
                pylon.TimeoutHandling_Return,
            )
        except Exception:
            time.sleep(0.01)
        finally:
            if result is not None:
                try:
                    result.Release()
                except Exception:
                    pass


def _stop_and_close(camera) -> None:
    if camera is None:
        return
    try:
        if camera.IsGrabbing():
            camera.StopGrabbing()
    except Exception as exc:
        print(f"[CAMERA] StopGrabbing fehlgeschlagen: {exc}", flush=True)
    try:
        if camera.IsOpen():
            camera.Close()
    except Exception as exc:
        print(f"[CAMERA] Close fehlgeschlagen: {exc}", flush=True)
    try:
        attached = getattr(camera, "IsPylonDeviceAttached", None)
        if not callable(attached) or attached():
            destroy = getattr(camera, "DestroyDevice", None)
            if callable(destroy):
                destroy()
    except Exception as exc:
        print(f"[CAMERA] DestroyDevice fehlgeschlagen: {exc}", flush=True)


def _dtype_code(dtype) -> int:
    if dtype == np.dtype(np.uint8):
        return 1
    if dtype == np.dtype(np.uint16):
        return 2
    raise RuntimeError(f"Nicht unterstuetzter Kamera-Datentyp: {dtype}")


def _publish_frame(
    image: np.ndarray,
    shared_buffer,
    slot_bytes: int,
    slot_locks,
    slot_sequences,
    slot_timestamps,
    slot_ndims,
    slot_shapes0,
    slot_shapes1,
    slot_shapes2,
    slot_dtype_codes,
    latest_sequence,
    timestamp: float,
) -> None:
    image = np.ascontiguousarray(image)
    dtype_code = _dtype_code(image.dtype)
    if image.ndim not in (2, 3):
        raise RuntimeError(f"Unerwartete Kamera-Bilddimensionen: {image.shape}")
    if image.ndim == 3 and image.shape[2] > 4:
        raise RuntimeError(f"Zu viele Kamera-Bildkanaele: {image.shape}")
    if image.nbytes > slot_bytes:
        raise RuntimeError(
            f"Kamera-Frame ist zu gross fuer Ringpuffer: "
            f"{image.nbytes} > {slot_bytes} Bytes"
        )

    sequence = int(latest_sequence.value) + 1
    slot = sequence % len(slot_locks)
    offset = slot * slot_bytes
    with slot_locks[slot]:
        target = np.ndarray(
            (image.nbytes,),
            dtype=np.uint8,
            buffer=shared_buffer,
            offset=offset,
        )
        source = np.frombuffer(image.data, dtype=np.uint8, count=image.nbytes)
        target[:] = source
        slot_ndims[slot] = image.ndim
        slot_shapes0[slot] = image.shape[0]
        slot_shapes1[slot] = image.shape[1]
        slot_shapes2[slot] = image.shape[2] if image.ndim == 3 else 0
        slot_dtype_codes[slot] = dtype_code
        slot_timestamps[slot] = timestamp
        slot_sequences[slot] = sequence

    with latest_sequence.get_lock():
        latest_sequence.value = sequence


def _grab_error_code(grab_result) -> str:
    try:
        return str(grab_result.GetErrorCode())
    except Exception:
        return "unbekannt"


def _grab_error_text(grab_result) -> str:
    try:
        return str(grab_result.GetErrorDescription())
    except Exception:
        return "unbekannt"


def run_basler_capture_process(
    cfg,
    shared_buffer,
    slot_bytes: int,
    slot_locks,
    slot_sequences,
    slot_timestamps,
    slot_ndims,
    slot_shapes0,
    slot_shapes1,
    slot_shapes2,
    slot_dtype_codes,
    latest_sequence,
    stop_event,
    ready_event,
    startup_state,
    fatal_state,
    fps_value,
    events,
) -> None:
    """Open pylon and publish frames until stopped or fatally disconnected."""
    camera: Any = None
    try:
        from pypylon import pylon

        camera, device_info = _open_camera(cfg, pylon)
        _settle_stream(camera, pylon)
        print(
            f"[CAMERA] Basler geoeffnet: {_device_label(device_info)}",
            flush=True,
        )
        for name in (
            "DeviceModelName",
            "PixelFormat",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "ExposureAuto",
            "ExposureTime",
            "GainAuto",
            "Gain",
            "AcquisitionMode",
            "TriggerMode",
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate",
            "BslResultingAcquisitionFrameRate",
            "ResultingFrameRate",
        ):
            value = _get_feature(camera, name)
            if value is not None:
                print(f"pylon {name} = {value}", flush=True)

        startup_state.value = 1
        ready_event.set()

        consecutive_failures = 0
        first_failure_time = 0.0
        report_time = time.monotonic()
        report_frames = 0

        while not stop_event.is_set():
            grab_result = None
            ok = False
            try:
                if not camera.IsGrabbing():
                    raise RuntimeError("Kamera grabbt nicht (IsGrabbing=False)")

                grab_result = camera.RetrieveResult(
                    RETRIEVE_TIMEOUT_MS,
                    pylon.TimeoutHandling_Return,
                )
                if grab_result is None:
                    raise RuntimeError("RetrieveResult liefert None")
                if not grab_result.IsValid():
                    raise RuntimeError(
                        "Grab-Ergebnis ungueltig: "
                        f"code={_grab_error_code(grab_result)} "
                        f"text={_grab_error_text(grab_result)}"
                    )
                if not grab_result.GrabSucceeded():
                    raise RuntimeError(
                        "Grab fehlgeschlagen: "
                        f"code={_grab_error_code(grab_result)} "
                        f"text={_grab_error_text(grab_result)}"
                    )

                image = grab_result.GetArray()
                if image is None:
                    raise RuntimeError("Grab-Ergebnis enthaelt kein Bild")

                timestamp = time.perf_counter()
                _publish_frame(
                    image,
                    shared_buffer,
                    slot_bytes,
                    slot_locks,
                    slot_sequences,
                    slot_timestamps,
                    slot_ndims,
                    slot_shapes0,
                    slot_shapes1,
                    slot_shapes2,
                    slot_dtype_codes,
                    latest_sequence,
                    timestamp,
                )
                ok = True
                report_frames += 1
            except Exception as exc:
                message = f"Basler-Grab fehlgeschlagen: {exc}"
                _emit(events, "error", message)
                consecutive_failures += 1
                if consecutive_failures == 1:
                    first_failure_time = time.monotonic()
                if (
                    consecutive_failures >= FAILURE_THRESHOLD
                    and time.monotonic() - first_failure_time
                    >= FAILURE_MIN_DURATION_SECONDS
                ):
                    fatal_state.value = 1
                    print(f"[CAM-ERR] {message}", flush=True)
                    _emit(
                        events,
                        "fatal",
                        "Dauerhafter Basler-Read-Fehler; Capture-Prozess wird beendet",
                    )
                    # Leave the grab loop normally. This lets the finally
                    # block release the current GrabResult and lets the
                    # process-level cleanup try StopGrabbing/Close. The
                    # parent has a bounded join and will terminate/kill us if
                    # pylon blocks during that cleanup.
                    break
                time.sleep(0.005)
            finally:
                if grab_result is not None:
                    try:
                        grab_result.Release()
                    except Exception as exc:
                        _emit(events, "error", f"Grab-Ergebnis Release fehlgeschlagen: {exc}")

            if ok:
                consecutive_failures = 0
                first_failure_time = 0.0

            now = time.monotonic()
            if now - report_time >= 1.0:
                fps_value.value = report_frames / (now - report_time)
                report_frames = 0
                report_time = now
    except Exception as exc:
        startup_state.value = -1
        ready_event.set()
        _emit(events, "fatal", f"Basler-Prozess konnte nicht starten: {exc}")
    finally:
        # On a normal stop this is best-effort. The parent has a hard timeout
        # and terminates this process if pylon blocks here.
        if camera is not None:
            _stop_and_close(camera)
