"""The machine's switches (D40): screen brightness, dark mode, focus assist
("do not disturb" on Windows 10) and the Bluetooth and Wi-Fi radios.

Measured on the owner's machine 2026-09-27 (N0): brightness over WMI's COM
reads in 0.23 s and writes in 0.01 s (PowerShell took 3.5 s); focus assist
has no documented API on Windows 10 - its WNF state was written 0 -> 1 -> 0
and read back; the Radios API allowed access without admin and listed
Bluetooth and Wi-Fi. Everything but the radios blocks and is run on a worker
thread by the tool.
"""

from __future__ import annotations

import ctypes
import subprocess
import winreg
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol

# pywin32 ships no type information (`tools/web.py` says the same).
import pythoncom  # type: ignore[import-untyped]
import pywintypes  # type: ignore[import-untyped]
import win32com.client  # type: ignore[import-untyped]

__all__ = [
    "Brightness",
    "FocusAssist",
    "RadioKind",
    "RadioOutcome",
    "Radios",
    "RegistryTheme",
    "Switches",
    "Theme",
    "WinRtRadios",
    "WmiBrightness",
    "WnfFocusAssist",
]

RadioKind = Literal["bluetooth", "wifi"]
RadioOutcome = Literal["done", "already", "missing", "denied"]

PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
HWND_BROADCAST = 0xFFFF
WM_SETTINGCHANGE = 0x001A
SMTO_ABORTIFHUNG = 0x0002

# WNF_SHEL_QUIETHOURS_ACTIVE_PROFILE_CHANGED: 0 off, 1 priority only, 2 alarms only.
QUIET_HOURS = 0x0D83063EA3BF1C75
PRIORITY_ONLY = 1

_QUIET = subprocess.CREATE_NO_WINDOW


class Brightness(Protocol):
    def level(self) -> int | None: ...

    def set_level(self, percent: int) -> bool: ...


class Theme(Protocol):
    def dark(self) -> bool: ...

    def set_dark(self, on: bool) -> None: ...


class FocusAssist(Protocol):
    def on(self) -> bool | None: ...

    def set(self, on: bool) -> bool: ...


class Radios(Protocol):
    async def set(self, kind: RadioKind, on: bool) -> RadioOutcome: ...

    def disconnect_wifi(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class Switches:
    brightness: Brightness
    theme: Theme
    focus_assist: FocusAssist
    radios: Radios


class WmiBrightness:
    """The built-in screen's brightness through WMI; `None` / `False` on a
    machine whose screen has none (an external monitor)."""

    def level(self) -> int | None:
        return _in_wmi(_level, None)

    def set_level(self, percent: int) -> bool:
        return _in_wmi(partial(_set_level, percent=percent), False)


def _level(wmi: Any) -> int | None:
    found = list(wmi.InstancesOf("WmiMonitorBrightness"))
    return int(found[0].CurrentBrightness) if found else None


def _set_level(wmi: Any, *, percent: int) -> bool:
    methods = list(wmi.InstancesOf("WmiMonitorBrightnessMethods"))
    if not methods:
        return False
    screen = methods[0]
    params = screen.Methods_("WmiSetBrightness").InParameters.SpawnInstance_()
    params.Properties_.Item("Timeout").Value = 0
    params.Properties_.Item("Brightness").Value = percent
    screen.ExecMethod_("WmiSetBrightness", params)
    return True


def _in_wmi[T](work: Callable[[Any], T], failed: T) -> T:
    """`work` over WMI's namespace, with COM open for this thread - the tool
    runs it on whichever worker the loop picked; `failed` when COM refuses.

    Every COM object is let go before COM closes: `work`'s locals end with
    its frame, and a refusal's traceback with the `except` below. One let go
    after `CoUninitialize` makes pywin32 print "Win32 exception occurred
    releasing IUnknown" (seen at the desk, 2026-09-27).
    """
    pythoncom.CoInitialize()
    try:
        try:
            return work(win32com.client.GetObject("winmgmts:\\\\.\\root\\WMI"))
        except pywintypes.com_error:
            return failed
    finally:
        pythoncom.CoUninitialize()


class RegistryTheme:
    """Dark mode: the two `Personalize` values, then the message every
    window listens for to repaint."""

    def dark(self) -> bool:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return int(value) == 0

    def set_dark(self, on: bool) -> None:
        light = 0 if on else 1
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE, 0, winreg.KEY_SET_VALUE) as key:
            for name in ("AppsUseLightTheme", "SystemUsesLightTheme"):
                winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, light)
        result = ctypes.c_size_t()
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST,
            WM_SETTINGCHANGE,
            0,
            "ImmersiveColorSet",
            SMTO_ABORTIFHUNG,
            1000,
            ctypes.byref(result),
        )


class WnfFocusAssist:
    """Focus assist through its undocumented WNF state (Windows 10 has no
    documented call). `on` is `None` and `set` is `False` when the state
    cannot be read or written - the tool then opens its Settings page."""

    def on(self) -> bool | None:
        state = ctypes.c_uint64(QUIET_HOURS)
        stamp = wintypes.ULONG()
        value = wintypes.DWORD()
        size = wintypes.ULONG(ctypes.sizeof(value))
        status = ctypes.windll.ntdll.NtQueryWnfStateData(
            ctypes.byref(state),
            None,
            None,
            ctypes.byref(stamp),
            ctypes.byref(value),
            ctypes.byref(size),
        )
        return None if status != 0 else value.value != 0

    def set(self, on: bool) -> bool:
        state = ctypes.c_uint64(QUIET_HOURS)
        value = wintypes.DWORD(PRIORITY_ONLY if on else 0)
        status = ctypes.windll.ntdll.NtUpdateWnfStateData(
            ctypes.byref(state), ctypes.byref(value), ctypes.sizeof(value), None, None, 0, 0
        )
        return int(status) == 0


class WinRtRadios:
    """Bluetooth and Wi-Fi through `Windows.Devices.Radios`, awaited on the loop."""

    async def set(self, kind: RadioKind, on: bool) -> RadioOutcome:
        from winrt.windows.devices.radios import (
            Radio,
            RadioAccessStatus,
            RadioState,
        )
        from winrt.windows.devices.radios import RadioKind as WindowsKind

        if await Radio.request_access_async() != RadioAccessStatus.ALLOWED:
            return "denied"
        wanted = WindowsKind.BLUETOOTH if kind == "bluetooth" else WindowsKind.WI_FI
        radios = [radio for radio in await Radio.get_radios_async() if radio.kind == wanted]
        if not radios:
            return "missing"
        state = RadioState.ON if on else RadioState.OFF
        if all(radio.state == state for radio in radios):
            return "already"
        for radio in radios:
            if await radio.set_state_async(state) != RadioAccessStatus.ALLOWED:
                return "denied"
        return "done"

    def disconnect_wifi(self) -> bool:
        """Drops the Wi-Fi connection without switching the radio off."""
        done = subprocess.run(
            ["netsh", "wlan", "disconnect"],  # noqa: S607  # Windows' own, fixed arguments
            capture_output=True,
            creationflags=_QUIET,
        )
        return done.returncode == 0
