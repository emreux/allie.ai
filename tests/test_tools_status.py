"""`system_status` (15 Sep 2026): five readings of the machine, one tool.

The tool is tested against a machine that answers whatever the test says,
so that a desktop without a battery and a laptop at 3% can both be tried on
this laptop. `Win32Machine` itself is tried once for real at the end: the
suite runs on Windows, and the claim that the readings come back at all is
worth one test.
"""

from __future__ import annotations

import threading

import pytest

from assistant.machine import Battery, CpuTimes, Disk, Memory, Network, Win32Machine, cpu_percent
from assistant.tools import status
from assistant.tools.registry import Tool
from assistant.tools.status import PARTS, system_status_for

GIB = 2**30

# This laptop, more or less, on 2026-09-15.
PLUGGED_IN = Battery(percent=96, plugged=True, minutes_left=None)
NEARLY_FULL = Memory(total_bytes=16 * GIB, available_bytes=1 * GIB, percent_used=94)
SYSTEM_DRIVE = Disk(drive="C:", total_bytes=238 * GIB, free_bytes=48 * GIB)
HOME_WIFI = Network(online=True, wifi="Home", signal_percent=84)
# 100 ticks elapsed, 75 of them idle.
QUARTER_BUSY = (CpuTimes(idle=100, total=200), CpuTimes(idle=175, total=300))


class FakeMachine:
    """Answers with what it was given, and remembers the thread it was asked on."""

    def __init__(
        self,
        *,
        battery: Battery | None = PLUGGED_IN,
        memory: Memory = NEARLY_FULL,
        disk: Disk = SYSTEM_DRIVE,
        network: Network = HOME_WIFI,
        cpu: tuple[CpuTimes, CpuTimes] = QUARTER_BUSY,
    ) -> None:
        self._battery = battery
        self._memory = memory
        self._disk = disk
        self._network = network
        self._cpu = list(cpu)
        self.threads: list[threading.Thread] = []

    def battery(self) -> Battery | None:
        self.threads.append(threading.current_thread())
        return self._battery

    def memory(self) -> Memory:
        self.threads.append(threading.current_thread())
        return self._memory

    def disk(self) -> Disk:
        self.threads.append(threading.current_thread())
        return self._disk

    def cpu_times(self) -> CpuTimes:
        self.threads.append(threading.current_thread())
        return self._cpu.pop(0)

    def network(self) -> Network:
        self.threads.append(threading.current_thread())
        return self._network


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    # The half second between the two processor readings is real time; a
    # test about the sentence has no use for it.
    monkeypatch.setattr(status, "CPU_SAMPLE_SECONDS", 0.0)


def tool_over(machine: FakeMachine) -> Tool:
    return system_status_for(machine)


def test_it_is_a_safe_tool_whose_parts_are_an_enum_with_all_as_the_default() -> None:
    tool = tool_over(FakeMachine())

    assert tool.risk == "safe"
    assert tool.spec.name == "system_status"
    assert tool.spec.parameters["required"] == []
    assert tool.spec.parameters["properties"]["part"]["enum"] == [*PARTS, "all"]


async def test_all_is_the_five_readings_in_a_fixed_order() -> None:
    said = await tool_over(FakeMachine()).run()

    assert said.splitlines() == [
        "Battery: 96%, plugged in.",
        "Processor: 25% busy over the last half second.",
        "Memory: 15.0 of 16.0 GB in use (94%).",
        "Disk C: 48.0 of 238.0 GB free (20%).",
        "Network: online, on the Wi-Fi network 'Home' (signal 84%).",
    ]


async def test_one_part_is_one_line() -> None:
    assert await tool_over(FakeMachine()).run(part="disk") == "Disk C: 48.0 of 238.0 GB free (20%)."


# --------------------------------------------------------------------------
# The battery
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("battery", "said"),
    [
        (None, "Battery: none - this is a desktop, or Windows reports no battery."),
        (
            Battery(percent=40, plugged=False, minutes_left=130),
            "Battery: 40%, on battery, about 2 h 10 min left.",
        ),
        (
            Battery(percent=3, plugged=False, minutes_left=12),
            "Battery: 3%, on battery, about 12 min left.",
        ),
        (Battery(percent=40, plugged=False, minutes_left=None), "Battery: 40%, on battery."),
        # Plugged in, Windows still reports a lifetime on some machines; it
        # is the time left *if unplugged*, which nobody asked.
        (Battery(percent=100, plugged=True, minutes_left=300), "Battery: 100%, plugged in."),
        (
            Battery(percent=None, plugged=None, minutes_left=None),
            "Battery: unknown charge, power source unknown.",
        ),
    ],
)
async def test_the_battery_line_says_what_windows_knows(battery: Battery | None, said: str) -> None:
    assert await tool_over(FakeMachine(battery=battery)).run(part="battery") == said


# --------------------------------------------------------------------------
# The processor
# --------------------------------------------------------------------------


def test_the_processor_percentage_is_the_share_of_time_not_idle() -> None:
    before = CpuTimes(idle=1_000, total=2_000)
    after = CpuTimes(idle=1_250, total=3_000)  # 250 idle of 1000 elapsed

    assert cpu_percent(before, after) == 75


def test_a_processor_that_did_not_move_is_no_reading() -> None:
    same = CpuTimes(idle=5, total=5)

    assert cpu_percent(same, same) is None


def test_the_percentage_stays_between_zero_and_a_hundred() -> None:
    # Counters read on different processors can add up to more idle than
    # elapsed time; the answer is still a percentage.
    assert cpu_percent(CpuTimes(idle=0, total=0), CpuTimes(idle=200, total=100)) == 0


async def test_the_processor_is_read_twice_and_the_sentence_is_the_difference() -> None:
    machine = FakeMachine(cpu=(CpuTimes(idle=0, total=0), CpuTimes(idle=10, total=100)))

    assert (
        await tool_over(machine).run(part="cpu") == "Processor: 90% busy over the last half second."
    )
    assert machine._cpu == []


async def test_counters_that_did_not_move_are_said_so() -> None:
    machine = FakeMachine(cpu=(CpuTimes(idle=5, total=5), CpuTimes(idle=5, total=5)))

    assert (
        await tool_over(machine).run(part="cpu")
        == "Processor: no reading - the counters did not move."
    )


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("network", "said"),
    [
        (
            Network(online=False, wifi="", signal_percent=None),
            "Network: offline - no connection to the internet.",
        ),
        (
            Network(online=True, wifi="", signal_percent=None),
            "Network: online, over a wired connection or no Wi-Fi Windows can name.",
        ),
        (
            Network(online=True, wifi="Ev Ağı", signal_percent=None),
            "Network: online, on the Wi-Fi network 'Ev Ağı'.",
        ),
        # Offline on a Wi-Fi that leads nowhere: the flag wins, the name is
        # not said - "connected to Home" would be the wrong news.
        (
            Network(online=False, wifi="Home", signal_percent=50),
            "Network: offline - no connection to the internet.",
        ),
    ],
)
async def test_the_network_line(network: Network, said: str) -> None:
    assert await tool_over(FakeMachine(network=network)).run(part="network") == said


# --------------------------------------------------------------------------
# Where it runs
# --------------------------------------------------------------------------


async def test_the_machine_is_never_asked_on_the_event_loop() -> None:
    """Section 3.1 rule 4: a first call into `wininet` was measured at 22 ms."""
    machine = FakeMachine()

    await tool_over(machine).run()

    assert machine.threads and all(t is not threading.main_thread() for t in machine.threads)


# --------------------------------------------------------------------------
# The real thing, once
# --------------------------------------------------------------------------


def test_windows_answers_every_reading() -> None:
    """Not what the numbers are - this laptop's battery is its own business -
    but that each call comes back in the shape the tool reads."""
    machine = Win32Machine()

    battery = machine.battery()
    memory = machine.memory()
    disk = machine.disk()
    times = machine.cpu_times()
    network = machine.network()

    assert battery is None or (battery.percent is None or 0 <= battery.percent <= 100)
    assert memory.total_bytes > memory.available_bytes > 0
    assert 0 <= memory.percent_used <= 100
    assert disk.drive.endswith(":") and disk.total_bytes >= disk.free_bytes > 0
    assert times.total >= times.idle > 0
    assert isinstance(network.online, bool)
    assert network.signal_percent is None or (network.wifi and 0 <= network.signal_percent <= 100)
