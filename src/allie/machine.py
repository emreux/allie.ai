"""What the computer says about itself, asked of Windows directly.

Battery, memory, the system drive, the processor and the network: five
readings, each one call into a DLL Windows already has loaded, and each
answered in well under a millisecond (measured 2026-09-15: battery 0.25 ms,
memory 0.03 ms, disk 0.1 ms; the wireless query 15 ms the first time and
0.5 ms after, the online check 22 ms the first time). Nothing here needs
`psutil` - the tool that reads these (`tools/status.py`) wants five numbers,
not a process table - and nothing here shells out to `netsh` or `wmic`, whose
output is printed in the language of the Windows it runs on and would have
to be parsed by its labels (design.md section 3.12).

**Everything is a struct.** Every reading is `ctypes` filling a structure
from `winuser.h`'s neighbours, and every "unknown" Windows can answer with -
a battery percentage of 255, a lifetime of `0xFFFFFFFF`, a wireless adapter
that is not connected - is turned into `None` here so that the tool never has
to know the sentinel. A machine without a battery is `battery() -> None`,
which is the honest answer on a desktop.

**The processor is two readings, not one.** `GetSystemTimes` answers with
counters since boot; how busy the machine is *now* is the difference between
two of them a moment apart, and the moment of waiting belongs to the caller,
who can `await` it (`tools/status.py`); this module only reads.

`Machine` is the protocol the tool is written against, so that its tests can
hand it a machine with whatever battery they like; `Win32Machine` is the one
that asks Windows.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

__all__ = [
    "Battery",
    "CpuTimes",
    "Disk",
    "Machine",
    "Memory",
    "Network",
    "Win32Machine",
    "cpu_percent",
]

# `SYSTEM_POWER_STATUS`: what each field means when Windows does not know.
_AC_UNKNOWN = 255
_NO_BATTERY = 128  # BatteryFlag bit
_PERCENT_UNKNOWN = 255
_SECONDS_UNKNOWN = 0xFFFFFFFF

# `WlanQueryInterface` opcode for the current connection, from wlanapi.h
# (`wlan_intf_opcode_current_connection`), and the client version Windows
# Vista and later negotiate.
_WLAN_CURRENT_CONNECTION = 7
_WLAN_CLIENT_VERSION = 2
# `WLAN_INTERFACE_STATE`: `wlan_interface_state_connected`.
_WLAN_CONNECTED = 1


@dataclass(frozen=True, slots=True)
class Battery:
    """The battery as Windows reports it. `percent` and `plugged` are `None`
    when Windows says it does not know; `minutes_left` is `None` when it does
    not know or when the machine is plugged in, where the question is moot."""

    percent: int | None
    plugged: bool | None
    minutes_left: int | None


@dataclass(frozen=True, slots=True)
class Memory:
    total_bytes: int
    available_bytes: int
    # Windows' own figure, which is what the Task Manager shows.
    percent_used: int


@dataclass(frozen=True, slots=True)
class Disk:
    drive: str  # "C:"
    total_bytes: int
    free_bytes: int  # free to this user, which quotas can make less than free


@dataclass(frozen=True, slots=True)
class CpuTimes:
    """Two counters since boot, in 100-nanosecond units. `total` is kernel
    plus user time, and the kernel figure already contains the idle time -
    which is why `idle` is not added to it."""

    idle: int
    total: int


@dataclass(frozen=True, slots=True)
class Network:
    online: bool
    # The network's name (SSID) when a wireless adapter is connected, else "".
    wifi: str
    # Windows' 0-100 signal quality for that connection, else `None`.
    signal_percent: int | None


class Machine(Protocol):
    """The five readings. `Win32Machine` asks Windows; a test answers itself."""

    def battery(self) -> Battery | None: ...

    def memory(self) -> Memory: ...

    def disk(self) -> Disk: ...

    def cpu_times(self) -> CpuTimes: ...

    def network(self) -> Network: ...


def cpu_percent(before: CpuTimes, after: CpuTimes) -> int | None:
    """How busy the processors were between two readings, 0-100, or `None`
    when no time passed between them."""
    elapsed = after.total - before.total
    if elapsed <= 0:
        return None
    idle = after.idle - before.idle
    return max(0, min(100, round(100 * (1 - idle / elapsed))))


# --------------------------------------------------------------------------
# The structures Windows fills in
# --------------------------------------------------------------------------


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = (
        ("ac_line_status", ctypes.c_ubyte),
        ("battery_flag", ctypes.c_ubyte),
        ("battery_life_percent", ctypes.c_ubyte),
        ("system_status_flag", ctypes.c_ubyte),
        ("battery_life_time", wintypes.DWORD),
        ("battery_full_life_time", wintypes.DWORD),
    )


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("length", wintypes.DWORD),
        ("memory_load", wintypes.DWORD),
        ("total_phys", ctypes.c_ulonglong),
        ("avail_phys", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("avail_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("avail_virtual", ctypes.c_ulonglong),
        ("avail_extended_virtual", ctypes.c_ulonglong),
    )


class _Guid(ctypes.Structure):
    _fields_ = (
        ("data1", wintypes.DWORD),
        ("data2", wintypes.WORD),
        ("data3", wintypes.WORD),
        ("data4", ctypes.c_ubyte * 8),
    )


class _WlanInterfaceInfo(ctypes.Structure):
    _fields_ = (
        ("guid", _Guid),
        ("description", wintypes.WCHAR * 256),
        ("state", ctypes.c_uint),
    )


class _WlanInterfaceInfoList(ctypes.Structure):
    _fields_ = (
        ("count", wintypes.DWORD),
        ("index", wintypes.DWORD),
        ("interfaces", _WlanInterfaceInfo * 1),
    )


class _Dot11Ssid(ctypes.Structure):
    _fields_ = (("length", ctypes.c_ulong), ("ssid", ctypes.c_ubyte * 32))


class _WlanAssociationAttributes(ctypes.Structure):
    _fields_ = (
        ("ssid", _Dot11Ssid),
        ("bss_type", ctypes.c_uint),
        ("bssid", ctypes.c_ubyte * 6),
        ("phy_type", ctypes.c_uint),
        ("phy_index", ctypes.c_ulong),
        ("signal_quality", ctypes.c_ulong),
        ("rx_rate", ctypes.c_ulong),
        ("tx_rate", ctypes.c_ulong),
    )


class _WlanConnectionAttributes(ctypes.Structure):
    # The security attributes follow in Windows' own layout; nothing here
    # reads past the association, so they are left off.
    _fields_ = (
        ("state", ctypes.c_uint),
        ("connection_mode", ctypes.c_uint),
        ("profile_name", wintypes.WCHAR * 256),
        ("association", _WlanAssociationAttributes),
    )


class Win32Machine:
    """The readings, asked of `kernel32`, `wininet` and `wlanapi`."""

    def battery(self) -> Battery | None:
        status = _SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            raise ctypes.WinError()
        if status.battery_flag & _NO_BATTERY:
            return None
        percent = status.battery_life_percent
        plugged = status.ac_line_status
        seconds = status.battery_life_time
        return Battery(
            percent=None if percent == _PERCENT_UNKNOWN else int(percent),
            plugged=None if plugged == _AC_UNKNOWN else plugged == 1,
            minutes_left=None if seconds == _SECONDS_UNKNOWN else int(seconds) // 60,
        )

    def memory(self) -> Memory:
        status = _MemoryStatusEx()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return Memory(
            total_bytes=int(status.total_phys),
            available_bytes=int(status.avail_phys),
            percent_used=int(status.memory_load),
        )

    def disk(self) -> Disk:
        # The drive Windows itself is on: the one that fills up and the one
        # the user means by "the disk".
        drive = os.environ.get("SYSTEMDRIVE", "C:")
        free = ctypes.c_ulonglong()
        total = ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            f"{drive}\\", ctypes.byref(free), ctypes.byref(total), None
        ):
            raise ctypes.WinError()
        return Disk(drive=drive, total_bytes=int(total.value), free_bytes=int(free.value))

    def cpu_times(self) -> CpuTimes:
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise ctypes.WinError()
        return CpuTimes(idle=_ticks(idle), total=_ticks(kernel) + _ticks(user))

    def network(self) -> Network:
        flags = wintypes.DWORD()
        online = bool(ctypes.windll.wininet.InternetGetConnectedState(ctypes.byref(flags), 0))
        wifi, signal = _wireless()
        return Network(online=online, wifi=wifi, signal_percent=signal)


def _ticks(time: wintypes.FILETIME) -> int:
    return (int(time.dwHighDateTime) << 32) | int(time.dwLowDateTime)


def _wireless() -> tuple[str, int | None]:
    """The connected wireless network's name and signal, or `("", None)`.

    A machine with no wireless adapter, a service that is not running, or a
    query that fails all end the same way: not on Wi-Fi as far as anyone can
    tell, and a line in the log. The online flag above stands on its own.
    """
    try:
        wlan = ctypes.windll.wlanapi
    except OSError as failure:
        logger.debug("no wireless API on this machine: {}", failure)
        return "", None

    handle = wintypes.HANDLE()
    version = wintypes.DWORD()
    if wlan.WlanOpenHandle(_WLAN_CLIENT_VERSION, None, ctypes.byref(version), ctypes.byref(handle)):
        logger.debug("the wireless service would not open a handle")
        return "", None
    try:
        listing = ctypes.POINTER(_WlanInterfaceInfoList)()
        if wlan.WlanEnumInterfaces(handle, None, ctypes.byref(listing)):
            return "", None
        try:
            count = int(listing.contents.count)
            adapters = ctypes.cast(
                listing.contents.interfaces, ctypes.POINTER(_WlanInterfaceInfo * count)
            ).contents
            for adapter in adapters:
                if adapter.state != _WLAN_CONNECTED:
                    continue
                found = _connection(wlan, handle, adapter)
                if found is not None:
                    return found
            return "", None
        finally:
            wlan.WlanFreeMemory(listing)
    finally:
        wlan.WlanCloseHandle(handle, None)


def _connection(
    wlan: ctypes.WinDLL, handle: wintypes.HANDLE, adapter: _WlanInterfaceInfo
) -> tuple[str, int | None] | None:
    size = wintypes.DWORD()
    data = ctypes.c_void_p()
    if wlan.WlanQueryInterface(
        handle,
        ctypes.byref(adapter.guid),
        _WLAN_CURRENT_CONNECTION,
        None,
        ctypes.byref(size),
        ctypes.byref(data),
        None,
    ):
        return None
    try:
        attributes = ctypes.cast(data, ctypes.POINTER(_WlanConnectionAttributes)).contents
        ssid = attributes.association.ssid
        name = bytes(ssid.ssid[: int(ssid.length)]).decode("utf-8", errors="replace")
        return name, int(attributes.association.signal_quality)
    finally:
        wlan.WlanFreeMemory(data)
