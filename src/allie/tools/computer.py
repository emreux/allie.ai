"""`power`, `computer` and `close_app`: the machine itself (plan.md D40,
D41).

Separate tools because risk is per tool: `power` locks, sleeps, shuts down,
restarts and turns Wi-Fi off, and asks first; `computer` flips the switches
nobody minds being flipped - brightness, Bluetooth, dark mode, do not
disturb - stops a waiting shutdown, brings an open app to the front and
minimises every window, and does not ask; `close_app` closes an app the way
its close box does, after a question that names it as the catalogue does.

**The question is the pack's sentence for the action** (`Tool.said_as`,
D40): `power`'s `confirm_prompt` is `{action}`, and the gate reads `lock` as
"Bilgisayarı kilitleyeyim mi?" while the tool still gets `lock`.

**What would cut the answer off waits for it.** Sleep waits two seconds and
Wi-Fi three, so that the model can say what it is doing before the machine
stops listening; shutdown and restart wait thirty, for "stop it".
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Annotated, Literal, get_args

from loguru import logger

from allie import shell
from allie.computer.power import SHUTDOWN_SECONDS, Power
from allie.computer.switches import Radios, Switches
from allie.computer.windows import AppWindow, RunningApps
from allie.tools.registry import Tool, tool
from allie.tools.system import AppCatalog

__all__ = [
    "CLOSE_APP",
    "CLOSE_WAIT_SECONDS",
    "NOT_OPEN",
    "POWER_ACTIONS",
    "SLEEP_SECONDS",
    "TEXT",
    "WIFI_SECONDS",
    "ComputerAction",
    "Later",
    "PowerAction",
    "close_app_for",
    "computer_for",
    "power_for",
]

# The last link of the chain of section 3.12 for the questions `power`
# asks, one per action, and the one `close_app` asks (D36: each ends in "?"
# and holds no no word).
TEXT: dict[str, str] = {
    "close_app_confirm": "Shall I close {name}?",
    "power_lock_confirm": "Shall I lock the computer?",
    "power_sleep_confirm": "Shall I put the computer to sleep?",
    "power_shutdown_confirm": "It shuts down after thirty seconds; shall I shut the computer down?",
    "power_restart_confirm": "It restarts after thirty seconds; shall I restart the computer?",
    "power_wifi_off_confirm": (
        "Once Wi-Fi is off I cannot hear you, and turning it back on is up to you; "
        "shall I turn it off?"
    ),
}

PowerAction = Literal["lock", "sleep", "shutdown", "restart", "wifi_off"]
POWER_ACTIONS: tuple[str, ...] = get_args(PowerAction)
ComputerAction = Literal[
    "brightness",
    "bluetooth",
    "dark_mode",
    "do_not_disturb",
    "cancel_shutdown",
    "focus",
    "minimize_all",
]

SLEEP_SECONDS = 2.0
WIFI_SECONDS = 3.0
# How long a closed app is given before its windows are counted again: one
# still there is asking something - whether to save, most likely.
CLOSE_WAIT_SECONDS = 2.0

# Schedules async work after a delay without holding the tool's answer.
Later = Callable[[float, Callable[[], Awaitable[object]]], None]

# The answers, addressed to the model.
LOCKED = "The computer is locked."
SLEEPING = "The computer goes to sleep in {seconds:.0f} seconds; say so now, in a few words."
SHUTTING_DOWN = (
    "The computer shuts down in {seconds} seconds. If the user says to stop it, call computer "
    "with action cancel_shutdown."
)
RESTARTING = (
    "The computer restarts in {seconds} seconds. If the user says to stop it, call computer "
    "with action cancel_shutdown."
)
WIFI_GOING = (
    "Wi-Fi goes off in {seconds:.0f} seconds. Tell the user now: after that you cannot hear "
    "them until they turn Wi-Fi back on themselves."
)
UNKNOWN_POWER = "No power action {action!r}; the actions are: {actions}."
UNKNOWN_ACTION = "No action {action!r}; the actions are: {actions}."
BAD_SWITCH = "Say 'on' or 'off' for {action}."
BAD_LEVEL = "Give brightness as a number from 0 to 100, not {value!r}."
NO_BRIGHTNESS = (
    "This screen's brightness cannot be set from here - an external monitor has its own "
    "buttons. Tell the user."
)
BRIGHTNESS_SET = "Screen brightness set to {level} % (was {before} %)."
SWITCHED = "{name} is {state}."
ALREADY = "{name} was already {state}."
RADIO_PAGE = (
    "Windows did not let this program switch {name}; opened the {name} page of Settings for "
    "the user to switch it."
)
FOCUS_ON = "Do not disturb is on: only priority notifications show."
FOCUS_OFF = "Do not disturb is off."
FOCUS_PAGE = (
    "Do not disturb could not be switched from here; opened its page in Settings (focus "
    "assist) for the user."
)
CANCELLED = "The shutdown or restart is stopped."
NOTHING_WAITING = "No shutdown or restart was waiting."
CLOSE_APP = "Closed {name} ({count} windows)."
CLOSE_ASKING = (
    "{name} was asked to close and is asking something first - probably whether to save. "
    "Tell the user to look at the screen."
)
NOT_OPEN = "No window of {name!r} is open. Nothing was closed."
FOCUSED = "Brought {title} to the front."
FOCUS_BLOCKED = (
    "{title} is open, but Windows kept it behind the window in front; it is flashing on the "
    "taskbar. Tell the user."
)
FOCUS_NOT_OPEN = "No window of {name!r} is open. To open it, call open_app."
MINIMISED = "Every window is minimised."
NO_WINDOWS = "Windows cannot be managed from here."

_TASKS: set[asyncio.Task[None]] = set()


def _later(seconds: float, work: Callable[[], Awaitable[object]]) -> None:
    """Runs `work` after `seconds` on the running loop, kept referenced so
    that it is not collected half way; a failure is logged, never raised."""

    async def run() -> None:
        await asyncio.sleep(seconds)
        try:
            await work()
        except Exception as failure:
            logger.warning("a delayed power action failed: {}", type(failure).__name__)

    task = asyncio.get_running_loop().create_task(run())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def power_for(
    machine: Power,
    radios: Radios,
    *,
    questions: Mapping[str, str] | None = None,
    later: Later = _later,
    shutdown_seconds: int = SHUTDOWN_SECONDS,
    sleep_seconds: float = SLEEP_SECONDS,
    wifi_seconds: float = WIFI_SECONDS,
) -> Tool:
    """`power`, bound to the machine, the radios and the pack's questions
    (`questions` by action; an action it leaves out is asked in `TEXT`'s
    English)."""

    async def wifi_off() -> None:
        outcome = await radios.set("wifi", False)
        if outcome in ("missing", "denied"):
            dropped = await asyncio.to_thread(radios.disconnect_wifi)
            logger.info("Wi-Fi radio {}, disconnected instead: {}", outcome, dropped)

    @tool(risk="confirm", confirm_prompt="{action}")
    async def power(
        action: Annotated[
            PowerAction,
            "lock, sleep, shutdown, restart, or wifi_off to turn the Wi-Fi radio off.",
        ],
    ) -> str:
        """Locks the computer, puts it to sleep, shuts it down or restarts it,
        or turns Wi-Fi off - once the user has confirmed out loud; the
        question is asked for you, so do not ask it yourself. Shutting down
        and restarting wait thirty seconds, and computer's cancel_shutdown
        stops them. With Wi-Fi off you cannot hear the user until they turn
        it back on themselves."""
        # A plain string: the schema offers five actions, and a model can
        # still send a sixth (mypy would call the last line unreachable).
        chosen: str = action
        if chosen == "lock":
            await asyncio.to_thread(machine.lock)
            return LOCKED
        if chosen == "sleep":
            later(sleep_seconds, partial(asyncio.to_thread, machine.sleep))
            return SLEEPING.format(seconds=sleep_seconds)
        if chosen in ("shutdown", "restart"):
            restart = chosen == "restart"
            await asyncio.to_thread(
                partial(machine.shutdown, restart=restart, seconds=shutdown_seconds)
            )
            return (RESTARTING if restart else SHUTTING_DOWN).format(seconds=shutdown_seconds)
        if chosen == "wifi_off":
            later(wifi_seconds, wifi_off)
            return WIFI_GOING.format(seconds=wifi_seconds)
        return UNKNOWN_POWER.format(action=chosen, actions=", ".join(POWER_ACTIONS))

    given = questions or {}
    said = {
        action: given.get(action) or TEXT[f"power_{action}_confirm"] for action in POWER_ACTIONS
    }
    return replace(power, said_as={"action": said})


def computer_for(
    switches: Switches,
    machine: Power,
    *,
    open_page: Callable[[str], Awaitable[None]] = shell.open_target,
    running: RunningApps | None = None,
) -> Tool:
    """`computer`, bound to the machine's switches and its open windows."""
    actions = get_args(ComputerAction)

    @tool(risk="safe")
    async def computer(
        action: Annotated[ComputerAction, "Which switch."],
        value: Annotated[
            str,
            "brightness: a number from 0 to 100; bluetooth, dark_mode, do_not_disturb: 'on' or "
            "'off'; focus: the application's name; empty for cancel_shutdown and minimize_all.",
        ] = "",
    ) -> str:
        """Flips the computer's switches: the screen's brightness, Bluetooth,
        dark mode, do not disturb (focus assist); brings an open app's window
        to the front (focus, with the app's name); minimises every window
        (minimize_all) - and cancel_shutdown stops a shutdown or restart that
        power set going. Nothing here asks the user first. Wi-Fi is power's,
        because turning it off cuts you off."""
        if action == "brightness":
            return await _brightness(switches, value)
        if action == "cancel_shutdown":
            stopped = await asyncio.to_thread(machine.cancel_shutdown)
            return CANCELLED if stopped else NOTHING_WAITING
        if action == "focus":
            return await _focus_window(running, value)
        if action == "minimize_all":
            if running is None:
                return NO_WINDOWS
            await asyncio.to_thread(running.minimize_all)
            return MINIMISED
        if action not in actions:
            return UNKNOWN_ACTION.format(action=action, actions=", ".join(actions))
        on = _switch(value)
        if on is None:
            return BAD_SWITCH.format(action=action)
        if action == "bluetooth":
            outcome = await switches.radios.set("bluetooth", on)
            return await _radio("Bluetooth", on, outcome, "ms-settings:bluetooth", open_page)
        if action == "dark_mode":
            return await _dark_mode(switches, on)
        return await _focus(switches, on, open_page)

    return computer


def close_app_for(
    running: RunningApps,
    catalog: AppCatalog,
    *,
    confirm_prompt: str = TEXT["close_app_confirm"],
    wait: float = CLOSE_WAIT_SECONDS,
) -> Tool:
    """`close_app`, bound to the open windows, the app catalogue and the
    question it asks (D41). `prepare` finds the app before the question, so
    that the question names it as the catalogue does and an app that is not
    open is said instead of asked about."""

    def windows_of(name: str) -> list[AppWindow]:
        return running.find(name, catalog.find(name))

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def close_app(
        name: Annotated[str, "The application, as the user said it."],
    ) -> str:
        """Closes an open application the way its close button does - every
        window of it; an app with unsaved work asks first, and nothing is
        forced. The user is asked to confirm first; do not ask them
        yourself."""
        windows = await asyncio.to_thread(windows_of, name)
        if not windows:
            return NOT_OPEN.format(name=name)
        await asyncio.to_thread(running.close, windows)
        await asyncio.sleep(wait)
        if await asyncio.to_thread(running.alive, windows):
            return CLOSE_ASKING.format(name=name)
        return CLOSE_APP.format(name=name, count=len(windows))

    async def prepare(name: str) -> dict[str, str] | str:
        windows = await asyncio.to_thread(windows_of, name)
        if not windows:
            return NOT_OPEN.format(name=name)
        entry = catalog.find(name)
        return {"name": entry.spoken if entry is not None else name.strip()}

    return replace(close_app, prepare=prepare)


async def _focus_window(running: RunningApps | None, name: str) -> str:
    """The front window of the app `name` means, brought forward. By the
    spoken name only - `computer` has no catalogue; `open_app`, which has
    one, brings an open app forward too."""
    if running is None:
        return NO_WINDOWS
    windows = await asyncio.to_thread(running.find, name)
    if not windows:
        return FOCUS_NOT_OPEN.format(name=name)
    if await asyncio.to_thread(running.front, windows[0]):
        return FOCUSED.format(title=windows[0].title)
    return FOCUS_BLOCKED.format(title=windows[0].title)


async def _brightness(switches: Switches, value: str) -> str:
    try:
        wanted = max(0, min(100, int(float(value.strip().rstrip("%")))))
    except ValueError:
        return BAD_LEVEL.format(value=value)
    before = await asyncio.to_thread(switches.brightness.level)
    if before is None or not await asyncio.to_thread(switches.brightness.set_level, wanted):
        return NO_BRIGHTNESS
    return BRIGHTNESS_SET.format(level=wanted, before=before)


async def _radio(
    name: str,
    on: bool,
    outcome: str,
    page: str,
    open_page: Callable[[str], Awaitable[None]],
) -> str:
    state = "on" if on else "off"
    if outcome == "done":
        return SWITCHED.format(name=name, state=state)
    if outcome == "already":
        return ALREADY.format(name=name, state=state)
    await open_page(page)
    return RADIO_PAGE.format(name=name)


async def _dark_mode(switches: Switches, on: bool) -> str:
    state = "on" if on else "off"
    if await asyncio.to_thread(switches.theme.dark) == on:
        return ALREADY.format(name="Dark mode", state=state)
    await asyncio.to_thread(switches.theme.set_dark, on)
    return SWITCHED.format(name="Dark mode", state=state)


async def _focus(switches: Switches, on: bool, open_page: Callable[[str], Awaitable[None]]) -> str:
    if await asyncio.to_thread(switches.focus_assist.set, on):
        return FOCUS_ON if on else FOCUS_OFF
    await open_page("ms-settings:quiethours")
    return FOCUS_PAGE


def _switch(value: str) -> bool | None:
    said = value.strip().casefold()
    if said in ("on", "true", "1", "yes"):
        return True
    if said in ("off", "false", "0", "no"):
        return False
    return None
