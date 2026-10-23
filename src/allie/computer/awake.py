"""Awake while the phone can reach the computer (plan.md D61).

Measured 2026-10-23: this laptop has Modern Standby only, and on the
Balanced plan it falls asleep after 3 hours idle on mains and 1 on battery.
Asleep, Windows pauses desktop programs - Allie among them - and the phone
gets no answer. A power request keeps the system up while the phone remote
runs, the way a media player keeps a computer up, and is given back when it
stops; the screen may still turn off. `powercfg /requests` shows it under
SYSTEM with `REASON`.

What it cannot do, by Microsoft's rules: on battery the request is dropped
five minutes after the sleep timeout, and the power button, a closed lid
(when set to sleep) and Start → Sleep still put the machine to sleep. The
owner keeps it plugged in and open.

A request that Windows will not make is a warning, not a reason to keep the
phone from answering while the computer is awake anyway.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from types import TracebackType
from typing import Protocol

from loguru import logger

__all__ = ["REASON", "Kernel", "StayAwake", "WindowsKernel"]

# What the request is filed under, where `powercfg /requests` shows it.
REASON = "Allie: Telegram remote control"

# winbase.h / winnt.h.
POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
# `PowerRequestSystemRequired` of POWER_REQUEST_TYPE: the system stays up,
# the display may turn off.
POWER_REQUEST_SYSTEM_REQUIRED = 1
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class Kernel(Protocol):
    """The four calls of a power request: made, set, cleared, closed."""

    def create(self, reason: str) -> int | None: ...

    def set(self, handle: int) -> bool: ...

    def clear(self, handle: int) -> bool: ...

    def close(self, handle: int) -> None: ...


class StayAwake:
    """Holds a system-required power request while it is entered."""

    def __init__(self, kernel: Kernel | None = None) -> None:
        self._kernel = kernel if kernel is not None else WindowsKernel()
        self._handle: int | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def __enter__(self) -> StayAwake:
        handle = self._kernel.create(REASON)
        if handle is None:
            logger.warning("the computer cannot be kept awake: Windows made no power request")
            return self
        if not self._kernel.set(handle):
            logger.warning("the computer cannot be kept awake: Windows refused the power request")
            self._kernel.close(handle)
            return self
        self._handle = handle
        logger.info("the computer is kept awake for the phone")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        failure: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        self._kernel.clear(handle)
        self._kernel.close(handle)
        logger.info("the computer may sleep again")


class _Detailed(ctypes.Structure):
    _fields_ = (
        ("LocalizedReasonModule", wintypes.HMODULE),
        ("LocalizedReasonId", wintypes.ULONG),
        ("ReasonStringCount", wintypes.ULONG),
        ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR)),
    )


class _Reason(ctypes.Union):
    _fields_ = (("Detailed", _Detailed), ("SimpleReasonString", wintypes.LPWSTR))


class _ReasonContext(ctypes.Structure):
    """REASON_CONTEXT, with the simple string of the two reasons it can hold."""

    _fields_ = (("Version", wintypes.ULONG), ("Flags", wintypes.DWORD), ("Reason", _Reason))


class WindowsKernel:
    """`kernel32`'s power requests (Windows 7 and later)."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._create = kernel32.PowerCreateRequest
        self._create.argtypes = (ctypes.POINTER(_ReasonContext),)
        self._create.restype = wintypes.HANDLE
        self._set = kernel32.PowerSetRequest
        self._set.argtypes = (wintypes.HANDLE, ctypes.c_int)
        self._set.restype = wintypes.BOOL
        self._clear = kernel32.PowerClearRequest
        self._clear.argtypes = (wintypes.HANDLE, ctypes.c_int)
        self._clear.restype = wintypes.BOOL
        self._close = kernel32.CloseHandle
        self._close.argtypes = (wintypes.HANDLE,)
        self._close.restype = wintypes.BOOL

    def create(self, reason: str) -> int | None:
        context = _ReasonContext(
            Version=POWER_REQUEST_CONTEXT_VERSION, Flags=POWER_REQUEST_CONTEXT_SIMPLE_STRING
        )
        context.Reason.SimpleReasonString = reason
        handle: int | None = self._create(ctypes.byref(context))
        if handle is None or handle == INVALID_HANDLE_VALUE:
            return None
        return handle

    def set(self, handle: int) -> bool:
        return bool(self._set(handle, POWER_REQUEST_SYSTEM_REQUIRED))

    def clear(self, handle: int) -> bool:
        return bool(self._clear(handle, POWER_REQUEST_SYSTEM_REQUIRED))

    def close(self, handle: int) -> None:
        self._close(handle)
