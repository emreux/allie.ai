"""`power` and `computer` (D40): the question each power action asks, what
runs when, and every switch's answers. Over fakes: nothing here locks,
sleeps, shuts down or switches a radio."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from allie.agent.policy import dispatch
from allie.computer.switches import RadioOutcome, Switches
from allie.computer.windows import RunningApps
from allie.live.base import ToolCall
from allie.tools.computer import (
    CLOSE_APP,
    NOT_OPEN,
    POWER_ACTIONS,
    TEXT,
    close_app_for,
    computer_for,
    power_for,
)
from allie.tools.registry import Tool, ToolRegistry
from allie.tools.system import AppCatalog, AppEntry

from .test_computer_windows import CHROME, CHROME_2, NOTEPAD, FakeDesktop


class FakeMachine:
    def __init__(self) -> None:
        self.done: list[str] = []
        self.waiting = True

    def lock(self) -> None:
        self.done.append("lock")

    def sleep(self) -> None:
        self.done.append("sleep")

    def shutdown(self, *, restart: bool, seconds: int) -> None:
        self.done.append(f"{'restart' if restart else 'shutdown'} in {seconds}")

    def cancel_shutdown(self) -> bool:
        self.done.append("cancel")
        return self.waiting


class FakeRadios:
    def __init__(self, outcome: RadioOutcome = "done", *, disconnects: bool = True) -> None:
        self.outcome: RadioOutcome = outcome
        self.disconnects = disconnects
        self.set_to: list[tuple[str, bool]] = []
        self.disconnected = 0

    async def set(self, kind: str, on: bool) -> RadioOutcome:
        self.set_to.append((kind, on))
        return self.outcome

    def disconnect_wifi(self) -> bool:
        self.disconnected += 1
        return self.disconnects


class FakeBrightness:
    def __init__(self, level: int | None = 50) -> None:
        self.now = level

    def level(self) -> int | None:
        return self.now

    def set_level(self, percent: int) -> bool:
        if self.now is None:
            return False
        self.now = percent
        return True


class FakeTheme:
    def __init__(self) -> None:
        self.is_dark = False

    def dark(self) -> bool:
        return self.is_dark

    def set_dark(self, on: bool) -> None:
        self.is_dark = on


class FakeFocus:
    def __init__(self, *, works: bool = True) -> None:
        self.works = works
        self.state = False

    def on(self) -> bool | None:
        return self.state if self.works else None

    def set(self, on: bool) -> bool:
        if self.works:
            self.state = on
        return self.works


class Later:
    def __init__(self) -> None:
        self.scheduled: list[tuple[float, Callable[[], Awaitable[object]]]] = []

    def __call__(self, seconds: float, work: Callable[[], Awaitable[object]]) -> None:
        self.scheduled.append((seconds, work))


class Confirm:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def __call__(self, question: str) -> bool:
        self.asked.append(question)
        return True


@pytest.fixture
def machine() -> FakeMachine:
    return FakeMachine()


async def gate(tool: Tool, **arguments: object) -> tuple[str, list[str]]:
    confirm = Confirm()
    said = await dispatch(
        ToolCall(id="c1", name=tool.spec.name, arguments=dict(arguments)),
        turn_id="t1",
        registry=ToolRegistry([tool]),
        confirm=confirm,
    )
    return said, confirm.asked


# -- power ---------------------------------------------------------------------


def test_power_asks_first_and_each_action_has_a_question_of_its_own(
    machine: FakeMachine,
) -> None:
    tool = power_for(machine, FakeRadios())

    assert tool.risk == "confirm"
    assert tool.confirm_prompt == "{action}"
    assert tool.spec.parameters["properties"]["action"]["enum"] == list(POWER_ACTIONS)
    assert tool.said_as == {
        "action": {action: TEXT[f"power_{action}_confirm"] for action in POWER_ACTIONS}
    }


async def test_the_question_is_the_packs_sentence_for_the_action(machine: FakeMachine) -> None:
    tool = power_for(machine, FakeRadios(), questions={"lock": "Bilgisayarı kilitleyeyim mi?"})

    said, asked = await gate(tool, action="lock")

    assert asked == ["Bilgisayarı kilitleyeyim mi?"]
    assert machine.done == ["lock"]
    assert said == "The computer is locked."


async def test_sleep_waits_for_the_answer_to_be_said(machine: FakeMachine) -> None:
    later = Later()

    said = await power_for(machine, FakeRadios(), later=later).run(action="sleep")

    assert machine.done == []
    assert [seconds for seconds, _ in later.scheduled] == [2.0]
    await later.scheduled[0][1]()
    assert machine.done == ["sleep"]
    assert "2 seconds" in said


async def test_shutdown_and_restart_wait_thirty_seconds_and_say_how_to_stop(
    machine: FakeMachine,
) -> None:
    tool = power_for(machine, FakeRadios())

    down = await tool.run(action="shutdown")
    again = await tool.run(action="restart")

    assert machine.done == ["shutdown in 30", "restart in 30"]
    assert "cancel_shutdown" in down and "30 seconds" in down
    assert "restart" in again.casefold()


async def test_wifi_goes_off_after_the_answer_and_the_answer_says_so(machine: FakeMachine) -> None:
    radios = FakeRadios()
    later = Later()

    said = await power_for(machine, radios, later=later).run(action="wifi_off")

    assert radios.set_to == []
    assert [seconds for seconds, _ in later.scheduled] == [3.0]
    await later.scheduled[0][1]()
    assert radios.set_to == [("wifi", False)]
    assert "cannot hear" in said


async def test_a_radio_windows_keeps_is_disconnected_instead(machine: FakeMachine) -> None:
    radios = FakeRadios("denied")
    later = Later()

    await power_for(machine, radios, later=later).run(action="wifi_off")
    await later.scheduled[0][1]()

    assert radios.disconnected == 1


# -- computer -------------------------------------------------------------------


def switches(
    *,
    brightness: FakeBrightness | None = None,
    focus: FakeFocus | None = None,
    radios: FakeRadios | None = None,
) -> Switches:
    return Switches(
        brightness=brightness or FakeBrightness(),
        theme=FakeTheme(),
        focus_assist=focus or FakeFocus(),
        radios=radios or FakeRadios(),
    )


class Pages:
    def __init__(self) -> None:
        self.opened: list[str] = []

    async def __call__(self, target: str) -> None:
        self.opened.append(target)


def test_computer_is_safe(machine: FakeMachine) -> None:
    tool = computer_for(switches(), machine)

    assert tool.risk == "safe"
    assert tool.spec.parameters["required"] == ["action"]


async def test_brightness_is_set_clamped_and_said_with_the_level_before(
    machine: FakeMachine,
) -> None:
    light = FakeBrightness(50)
    tool = computer_for(switches(brightness=light), machine)

    assert await tool.run(action="brightness", value="140") == (
        "Screen brightness set to 100 % (was 50 %)."
    )
    assert light.now == 100


async def test_a_screen_without_a_brightness_control_is_said(machine: FakeMachine) -> None:
    said = await computer_for(switches(brightness=FakeBrightness(None)), machine).run(
        action="brightness", value="40"
    )

    assert "cannot be set from here" in said


async def test_brightness_needs_a_number(machine: FakeMachine) -> None:
    said = await computer_for(switches(), machine).run(action="brightness", value="bright")

    assert said.startswith("Give brightness as a number")


async def test_bluetooth_is_switched_through_the_radio(machine: FakeMachine) -> None:
    radios = FakeRadios()

    said = await computer_for(switches(radios=radios), machine).run(action="bluetooth", value="off")

    assert radios.set_to == [("bluetooth", False)]
    assert said == "Bluetooth is off."


async def test_a_bluetooth_windows_keeps_opens_its_page(machine: FakeMachine) -> None:
    pages = Pages()

    said = await computer_for(switches(radios=FakeRadios("missing")), machine, open_page=pages).run(
        action="bluetooth", value="on"
    )

    assert pages.opened == ["ms-settings:bluetooth"]
    assert "opened the Bluetooth page" in said


async def test_dark_mode_is_switched_and_already_is_said(machine: FakeMachine) -> None:
    tool = computer_for(switches(), machine)

    assert await tool.run(action="dark_mode", value="on") == "Dark mode is on."
    assert await tool.run(action="dark_mode", value="on") == "Dark mode was already on."


async def test_do_not_disturb_that_will_not_switch_opens_its_page(machine: FakeMachine) -> None:
    pages = Pages()

    said = await computer_for(switches(focus=FakeFocus(works=False)), machine, open_page=pages).run(
        action="do_not_disturb", value="on"
    )

    assert pages.opened == ["ms-settings:quiethours"]
    assert "Settings" in said


async def test_do_not_disturb_on(machine: FakeMachine) -> None:
    said = await computer_for(switches(), machine).run(action="do_not_disturb", value="on")

    assert said.startswith("Do not disturb is on")


async def test_a_switch_needs_on_or_off(machine: FakeMachine) -> None:
    said = await computer_for(switches(), machine).run(action="dark_mode", value="maybe")

    assert said == "Say 'on' or 'off' for dark_mode."


async def test_a_waiting_shutdown_is_stopped_without_a_question(machine: FakeMachine) -> None:
    tool = computer_for(switches(), machine)

    assert await tool.run(action="cancel_shutdown") == "The shutdown or restart is stopped."
    machine.waiting = False
    assert await tool.run(action="cancel_shutdown") == "No shutdown or restart was waiting."


# -- windows (D41) ---------------------------------------------------------------

CATALOG = AppCatalog([AppEntry(name="Google Chrome", launch="chrome.lnk")])


def test_close_app_asks_first_with_the_apps_real_name() -> None:
    tool = close_app_for(RunningApps(FakeDesktop(CHROME), own_pid=999), CATALOG)

    assert tool.risk == "confirm"
    assert tool.prepare is not None
    assert tool.confirm_prompt == TEXT["close_app_confirm"] == "Shall I close {name}?"


async def test_the_question_names_the_app_as_the_catalogue_does() -> None:
    desktop = FakeDesktop(CHROME, NOTEPAD)
    tool = close_app_for(RunningApps(desktop, own_pid=999), CATALOG, wait=0)

    said, asked = await gate(tool, name="chrome")

    assert asked == ["Shall I close Google Chrome?"]
    assert desktop.closed == [11]
    assert said == CLOSE_APP.format(name="Google Chrome", count=1)


async def test_an_app_that_is_not_open_is_said_and_nothing_is_asked() -> None:
    tool = close_app_for(RunningApps(FakeDesktop(NOTEPAD), own_pid=999), CATALOG, wait=0)

    said, asked = await gate(tool, name="Spotify")

    assert asked == []
    assert said == NOT_OPEN.format(name="Spotify")


async def test_a_window_that_stays_is_the_user_being_asked_to_save() -> None:
    class Stubborn(FakeDesktop):
        def alive(self, handle: int) -> bool:
            return handle == 12

    tool = close_app_for(RunningApps(Stubborn(CHROME, CHROME_2), own_pid=999), CATALOG, wait=0)

    said, _ = await gate(tool, name="chrome")

    assert "asking something" in said


async def test_focus_brings_the_front_window_of_the_app_forward(machine: FakeMachine) -> None:
    desktop = FakeDesktop(NOTEPAD, CHROME)
    tool = computer_for(switches(), machine, running=RunningApps(desktop, own_pid=999))

    said = await tool.run(action="focus", value="chrome")

    assert desktop.fronted == [11]
    assert said == "Brought Yeni sekme - Google Chrome to the front."


async def test_focus_on_an_app_that_is_not_open_says_so(machine: FakeMachine) -> None:
    tool = computer_for(switches(), machine, running=RunningApps(FakeDesktop(NOTEPAD), own_pid=999))

    said = await tool.run(action="focus", value="Spotify")

    assert "open_app" in said


async def test_minimize_all(machine: FakeMachine) -> None:
    desktop = FakeDesktop()
    tool = computer_for(switches(), machine, running=RunningApps(desktop, own_pid=999))

    assert await tool.run(action="minimize_all") == "Every window is minimised."
    assert desktop.minimised == 1
