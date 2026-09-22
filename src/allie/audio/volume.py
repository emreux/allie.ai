"""The system volume as a number (plan.md D24, spec 2026-09-21 section 4).

The media keys step the volume and toggle the mute; they cannot set it to
thirty. Windows can, through Core Audio: the default playback endpoint's
`IAudioEndpointVolume` reads and writes the master level as a scalar from
0 to 1, and that is all this module asks of it - the level, the mute, and
nothing per channel.

**COM, on a worker thread, per call.** `comtypes` initialises COM on the
thread that imports it, so it is imported inside `_endpoint` and not at the
top; the tool calls this under `asyncio.to_thread` (section 3.1 rule 4),
and COM is initialised and released around each call, as
`media/window.py::text_boxes` does for the same reason. The interface
declarations are the three the call needs, each with its methods **in
vtable order** - COM finds a method by its position, and a declaration
with one missing calls the wrong one - and only as far down the table as
the last method used (`GetMute`); the declarations follow the Windows SDK
headers `mmdeviceapi.h` and `endpointvolume.h`.

`Volume` is the protocol the tool is written against; `SystemVolume` is the
one that asks Windows, with the endpoint injectable for the tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

__all__ = ["CLSID_DEVICE_ENUMERATOR", "SystemVolume", "Volume", "to_percent", "to_scalar"]

# `mmdeviceapi.h`: the enumerator's class, the two interfaces, and the two
# enums this asks with - `eRender` (playback) and `eConsole` (the role the
# volume mixer shows as the device's own).
CLSID_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
IID_DEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
IID_DEVICE = "{D666063F-1587-4E43-81F1-B948E807363F}"
# `endpointvolume.h`
IID_ENDPOINT_VOLUME = "{5CDF2C82-841E-4546-9722-0CF74078229A}"
_RENDER = 0
_CONSOLE = 0


def to_percent(scalar: float) -> int:
    """0..1 as the whole percentage the user hears; anything outside is clamped."""
    return round(max(0.0, min(1.0, scalar)) * 100)


def to_scalar(percent: int) -> float:
    return max(0, min(100, percent)) / 100


class Volume(Protocol):
    """What the tool needs: the level, and setting it."""

    def level(self) -> int: ...

    def set_level(self, percent: int) -> None: ...


class SystemVolume:
    """Windows' default playback endpoint."""

    def __init__(self, endpoint: Callable[[], AbstractContextManager[Any]] | None = None) -> None:
        self._endpoint = endpoint if endpoint is not None else _endpoint

    def level(self) -> int:
        with self._endpoint() as volume:
            return to_percent(float(volume.GetMasterVolumeLevelScalar()))

    def set_level(self, percent: int) -> None:
        """Sets the level; a muted endpoint is unmuted, because a level was asked for."""
        with self._endpoint() as volume:
            volume.SetMasterVolumeLevelScalar(to_scalar(percent), None)
            if volume.GetMute():
                volume.SetMute(False, None)


_interfaces: tuple[Any, Any, Any] | None = None


def _declare() -> tuple[Any, Any, Any]:
    """The three interfaces, declared once, on first use."""
    global _interfaces
    if _interfaces is not None:
        return _interfaces

    from ctypes import HRESULT, POINTER, c_float
    from ctypes.wintypes import BOOL, DWORD, LPCWSTR, UINT

    from comtypes import COMMETHOD, GUID, IUnknown  # type: ignore[import-untyped]

    class IAudioEndpointVolume(IUnknown):  # type: ignore[misc]
        _iid_ = GUID(IID_ENDPOINT_VOLUME)
        _methods_ = (
            COMMETHOD(
                [], HRESULT, "RegisterControlChangeNotify", (["in"], POINTER(IUnknown), "pNotify")
            ),
            COMMETHOD(
                [],
                HRESULT,
                "UnregisterControlChangeNotify",
                (["in"], POINTER(IUnknown), "pNotify"),
            ),
            COMMETHOD([], HRESULT, "GetChannelCount", (["out"], POINTER(UINT), "pnChannelCount")),
            COMMETHOD(
                [],
                HRESULT,
                "SetMasterVolumeLevel",
                (["in"], c_float, "fLevelDB"),
                (["in"], POINTER(GUID), "pguidEventContext"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "SetMasterVolumeLevelScalar",
                (["in"], c_float, "fLevel"),
                (["in"], POINTER(GUID), "pguidEventContext"),
            ),
            COMMETHOD(
                [], HRESULT, "GetMasterVolumeLevel", (["out"], POINTER(c_float), "pfLevelDB")
            ),
            COMMETHOD(
                [], HRESULT, "GetMasterVolumeLevelScalar", (["out"], POINTER(c_float), "pfLevel")
            ),
            COMMETHOD(
                [],
                HRESULT,
                "SetChannelVolumeLevel",
                (["in"], UINT, "nChannel"),
                (["in"], c_float, "fLevelDB"),
                (["in"], POINTER(GUID), "pguidEventContext"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "SetChannelVolumeLevelScalar",
                (["in"], UINT, "nChannel"),
                (["in"], c_float, "fLevel"),
                (["in"], POINTER(GUID), "pguidEventContext"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetChannelVolumeLevel",
                (["in"], UINT, "nChannel"),
                (["out"], POINTER(c_float), "pfLevelDB"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetChannelVolumeLevelScalar",
                (["in"], UINT, "nChannel"),
                (["out"], POINTER(c_float), "pfLevel"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "SetMute",
                (["in"], BOOL, "bMute"),
                (["in"], POINTER(GUID), "pguidEventContext"),
            ),
            COMMETHOD([], HRESULT, "GetMute", (["out"], POINTER(BOOL), "pbMute")),
        )

    class IMMDevice(IUnknown):  # type: ignore[misc]
        _iid_ = GUID(IID_DEVICE)
        _methods_ = (
            COMMETHOD(
                [],
                HRESULT,
                "Activate",
                (["in"], POINTER(GUID), "iid"),
                (["in"], DWORD, "dwClsCtx"),
                (["in"], POINTER(DWORD), "pActivationParams"),
                (["out"], POINTER(POINTER(IUnknown)), "ppInterface"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "OpenPropertyStore",
                (["in"], DWORD, "stgmAccess"),
                (["out"], POINTER(POINTER(IUnknown)), "ppProperties"),
            ),
            COMMETHOD([], HRESULT, "GetId", (["out"], POINTER(LPCWSTR), "ppstrId")),
            COMMETHOD([], HRESULT, "GetState", (["out"], POINTER(DWORD), "pdwState")),
        )

    class IMMDeviceEnumerator(IUnknown):  # type: ignore[misc]
        _iid_ = GUID(IID_DEVICE_ENUMERATOR)
        _methods_ = (
            COMMETHOD(
                [],
                HRESULT,
                "EnumAudioEndpoints",
                (["in"], DWORD, "dataFlow"),
                (["in"], DWORD, "dwStateMask"),
                (["out"], POINTER(POINTER(IUnknown)), "ppDevices"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetDefaultAudioEndpoint",
                (["in"], DWORD, "dataFlow"),
                (["in"], DWORD, "role"),
                (["out"], POINTER(POINTER(IMMDevice)), "ppEndpoint"),
            ),
        )

    _interfaces = (IMMDeviceEnumerator, IMMDevice, IAudioEndpointVolume)
    return _interfaces


@contextmanager
def _endpoint() -> Iterator[Any]:
    """The default playback endpoint's volume interface, for one call, with
    COM initialised and released around it."""
    import comtypes
    from comtypes import CLSCTX_ALL, GUID, CoCreateInstance

    enumerator_interface, _, volume_interface = _declare()
    comtypes.CoInitialize()
    try:
        enumerator = CoCreateInstance(
            GUID(CLSID_DEVICE_ENUMERATOR), interface=enumerator_interface, clsctx=CLSCTX_ALL
        )
        device = enumerator.GetDefaultAudioEndpoint(_RENDER, _CONSOLE)
        raw = device.Activate(volume_interface._iid_, CLSCTX_ALL, None)
        yield raw.QueryInterface(volume_interface)
    finally:
        comtypes.CoUninitialize()
