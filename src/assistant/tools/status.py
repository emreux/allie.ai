"""`system_status`: the computer's own state, in one tool (design.md section 3.6,
15 Sep 2026).

One tool with a `part` parameter rather than five tools, for the reason the
tool-count note of 2026-09-11 gives: every tool's schema rides on every
request, and "how much battery is left" and "is the disk full" are the same
kind of question with a different noun. `all` is what a model that was asked
"how is the computer doing" chooses.

The readings come from `machine.py`, which asks Windows and answers in
numbers; this file turns the numbers into one English line each, addressed
to the model, which says them in the user's language (section 3.12). The
processor is the one reading that takes time - two counters half a second
apart - and the half second is `await`ed here, not slept (section 3.1 rule
4). Everything else is a fraction of a millisecond and still runs off the
loop, because a first call into `wininet` was measured at 22 ms and nothing
awaited in `app.py` gets to spend that.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from assistant.machine import Battery, CpuTimes, Machine, cpu_percent
from assistant.tools.registry import Tool, tool

__all__ = ["CPU_SAMPLE_SECONDS", "PARTS", "Part", "system_status_for"]

Part = Literal["battery", "cpu", "memory", "disk", "network", "all"]
PARTS: tuple[str, ...] = ("battery", "cpu", "memory", "disk", "network")

# Between the two processor readings. Long enough for the counters to have
# moved on a quiet machine, short enough not to be heard as a pause.
CPU_SAMPLE_SECONDS = 0.5

GIB = 2**30


def system_status_for(machine: Machine) -> Tool:
    """`system_status`, bound to the machine it reads."""

    @tool(risk="safe")
    async def system_status(
        part: Annotated[
            Part,
            "Which reading: battery, cpu, memory, disk, network - or all of them.",
        ] = "all",
    ) -> str:
        """Reports the computer's own state: the battery's charge and whether
        it is plugged in, how busy the processor is, how much memory is in
        use, the free space on the system drive, and the network - online or
        not, and the name of the Wi-Fi. Use it for any question about the
        machine itself: "how much battery is left", "is the internet
        working", "is the disk full", "how is the computer doing". Not for the
        time or the date; get_current_time answers those."""
        wanted = PARTS if part == "all" else (part,)
        lines = [await _read(machine, name) for name in wanted]
        return "\n".join(lines)

    return system_status


async def _read(machine: Machine, part: str) -> str:
    if part == "cpu":
        before: CpuTimes = await asyncio.to_thread(machine.cpu_times)
        await asyncio.sleep(CPU_SAMPLE_SECONDS)
        after: CpuTimes = await asyncio.to_thread(machine.cpu_times)
        busy = cpu_percent(before, after)
        if busy is None:
            return "Processor: no reading - the counters did not move."
        return f"Processor: {busy}% busy over the last half second."
    if part == "battery":
        return _battery(await asyncio.to_thread(machine.battery))
    if part == "memory":
        memory = await asyncio.to_thread(machine.memory)
        used = memory.total_bytes - memory.available_bytes
        return (
            f"Memory: {_gib(used)} of {_gib(memory.total_bytes)} GB in use "
            f"({memory.percent_used}%)."
        )
    if part == "disk":
        disk = await asyncio.to_thread(machine.disk)
        percent = round(100 * disk.free_bytes / disk.total_bytes) if disk.total_bytes else 0
        return (
            f"Disk {disk.drive} {_gib(disk.free_bytes)} of {_gib(disk.total_bytes)} GB free "
            f"({percent}%)."
        )
    network = await asyncio.to_thread(machine.network)
    if not network.online:
        return "Network: offline - no connection to the internet."
    if not network.wifi:
        return "Network: online, over a wired connection or no Wi-Fi Windows can name."
    signal = f" (signal {network.signal_percent}%)" if network.signal_percent is not None else ""
    return f"Network: online, on the Wi-Fi network {network.wifi!r}{signal}."


def _battery(battery: Battery | None) -> str:
    if battery is None:
        return "Battery: none - this is a desktop, or Windows reports no battery."
    charge = "unknown charge" if battery.percent is None else f"{battery.percent}%"
    if battery.plugged is None:
        power = "power source unknown"
    elif battery.plugged:
        power = "plugged in"
    else:
        power = "on battery"
    left = ""
    if battery.minutes_left is not None and not battery.plugged:
        hours, minutes = divmod(battery.minutes_left, 60)
        left = f", about {hours} h {minutes} min left" if hours else f", about {minutes} min left"
    return f"Battery: {charge}, {power}{left}."


def _gib(size: int) -> str:
    return f"{size / GIB:.1f}"
