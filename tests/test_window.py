"""The window (`ui/window.py`, plan.md D20): the product's face, tested without a screen.

What is claimed: every `Screen` call is a queue put that returns at once -
the loop never waits on the window (spec section 2); the view shows the
state, the mode, the session and the phase in the pack's words; finished
turns become rows and the oldest go after two hundred; the quiet-microphone
sentence is said once, like the line; a level feeds the orb; the wizard's
page holds what was said and the open question, and an answer from the Tk
thread resolves the future on the loop; buttons reach the loop through
`call_soon_threadsafe`; closing hides with a tray and quits without. The
real Tk panel is built once, in `test_the_real_panel_comes_up_and_goes_down`
(skipped where there is no display).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from assistant import locales
from assistant.app import State, Turn
from assistant.audio.capture import QUIET_DBFS
from assistant.setup_wizard import TEXT as WIZARD_TEXT
from assistant.setup_wizard import Option
from assistant.ui import orb, status
from assistant.ui import window as window_ui
from assistant.ui.window import (
    TEXT,
    TRANSCRIPT_ROWS,
    Switch,
    View,
    Window,
    WindowError,
    WindowPrompter,
)

TR = locales.load("tr")


def said(key: str, table: dict[str, str] = TEXT) -> str:
    return TR.say(key, table[key])


def label_of(state: State) -> str:
    return said(status.label_key(state), status.TEXT)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class FakePanel:
    """The Tk half without Tk: drains the queue into the view the way the
    tick does, on the window's own thread, until it is told to quit."""

    def __init__(self, view: View, window: Window) -> None:
        self.view = view
        self.window = window
        self.thread: int | None = None
        self.quit = False

    def run(self) -> None:
        self.thread = threading.get_ident()
        while not self.quit:
            for message in self.window.drain():
                if message == ("quit",):
                    self.quit = True
                    return
                self.view.apply(message)
            time.sleep(0.005)


class StalledPanel(FakePanel):
    """A Tk half that never gets to drain - the window being dragged."""

    def run(self) -> None:
        self.thread = threading.get_ident()
        while not self.quit:
            time.sleep(0.005)
            if any(message == ("quit",) for message in self.window.drain()):
                return


class Built:
    """A window over a fake panel, with what its buttons did written down."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        tray: bool = False,
        panel: type[FakePanel] = FakePanel,
    ) -> None:
        self.toggles = 0
        self.quits = 0
        self.settings = 0
        self.threads: list[int] = []
        self.clock = Clock()
        panels: list[FakePanel] = []

        def build(view: View, window: Window) -> FakePanel:
            panels.append(panel(view, window))
            return panels[-1]

        def note(counter: str) -> None:
            setattr(self, counter, getattr(self, counter) + 1)
            self.threads.append(threading.get_ident())

        self.window = Window(
            TR,
            loop=loop,
            on_toggle=lambda: note("toggles"),
            on_quit=lambda: note("quits"),
            on_settings=lambda: note("settings"),
            tray=tray,
            panel=build,
            clock=self.clock,
        )
        self.window.start()
        [self.panel] = panels

    @property
    def view(self) -> View:
        return self.window.view

    def settle(self) -> None:
        """Waits until the fake panel has drained what was posted."""
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            time.sleep(0.01)
            if self.window.pending() == 0:
                return
        raise AssertionError("the panel did not drain the queue")


@pytest.fixture
def built() -> Iterator[Callable[..., Built]]:
    loop = asyncio.new_event_loop()
    windows: list[Built] = []

    def make(**parts: Any) -> Built:
        windows.append(Built(loop, **parts))
        return windows[-1]

    try:
        yield make
    finally:
        for one in windows:
            one.window.stop()
        loop.close()


def turn(heard: str = "saat kaç", said_: str = "On beş kırk iki.") -> Turn:
    return Turn(heard=heard, said=said_)


# --------------------------------------------------------------------------
# The loop never waits on the window
# --------------------------------------------------------------------------


def test_every_screen_call_is_a_queue_put_that_returns_at_once(built: Any) -> None:
    """With the Tk half stalled - a drag - the loop's calls still cost a
    queue put and nothing else (spec section 2, rule 2)."""
    one = built(panel=StalledPanel)
    screen = one.window

    before = time.perf_counter()
    screen.starting()
    screen.checking_model()
    screen.notice("a line")
    screen.state(State.SPEAKING)
    screen.session(True)
    screen.hands_free(False)
    screen.turn(turn())
    screen.level(-20.0)
    screen.microphone_level(-45.0)
    screen.loading()
    screen.show()
    screen.wizard(True)
    screen.say("hello")
    elapsed = time.perf_counter() - before

    assert elapsed < 0.05
    assert one.window.pending() == 13


def test_start_waits_for_the_panel_and_stop_joins_its_thread(built: Any) -> None:
    one = built()

    assert one.panel.thread is not None
    assert one.panel.thread != threading.get_ident()

    one.window.stop()
    assert one.panel.quit is True


def test_a_panel_that_cannot_be_built_is_a_window_error() -> None:
    def broken(view: View, window: Window) -> FakePanel:
        raise RuntimeError("no display")

    loop = asyncio.new_event_loop()
    try:
        broken_window = Window(
            TR,
            loop=loop,
            on_toggle=lambda: None,
            on_quit=lambda: None,
            on_settings=lambda: None,
            tray=False,
            panel=broken,
        )
        with pytest.raises(WindowError, match="no display"):
            broken_window.start()
    finally:
        loop.close()


# --------------------------------------------------------------------------
# What the view shows
# --------------------------------------------------------------------------


def test_it_starts_loading_and_shows_the_state_once_there_is_one(built: Any) -> None:
    one = built()
    assert one.view.label() == said("window_loading")

    one.window.starting()
    one.settle()
    assert one.view.label() == said("loading_speech", status.TEXT)

    one.window.state(State.IDLE)
    one.settle()
    assert one.view.label() == label_of(State.IDLE)


@pytest.mark.parametrize("state", list(State))
def test_every_state_has_a_label_and_a_colour(built: Any, state: State) -> None:
    one = built()
    one.window.state(state)
    one.settle()

    assert one.view.label() == label_of(state)
    assert one.view.frame(0.0).colour == orb.COLOURS[state]


def test_an_idle_microphone_that_is_off_says_not_listening_and_offers_to_start(
    built: Any,
) -> None:
    one = built()
    one.window.state(State.IDLE)
    one.window.hands_free(False)
    one.settle()

    assert one.view.label() == said("tray_not_listening", window_ui.TRAY_TEXT)
    assert one.view.switch_label() == said("tray_start_listening", window_ui.TRAY_TEXT)

    one.window.hands_free(True)
    one.settle()
    assert one.view.label() == label_of(State.IDLE)
    assert one.view.switch_label() == said("tray_stop_listening", window_ui.TRAY_TEXT)


def test_the_meter_says_whether_a_session_is_open_and_the_minutes(built: Any) -> None:
    one = built()
    closed = said("session_closed", status.TEXT)
    opened = said("session_open", status.TEXT)
    minutes = said("session_minutes", status.TEXT)

    assert one.view.meter() == f"{closed} · {minutes.format(minutes=0)}"

    one.window.session(True)
    one.settle()
    one.clock.now += 150
    assert one.view.meter() == f"{opened} · {minutes.format(minutes=2)}"

    one.window.session(False)
    one.settle()
    assert one.view.meter() == f"{closed} · {minutes.format(minutes=2)}"


def test_the_quit_and_settings_buttons_wear_the_packs_words(built: Any) -> None:
    one = built()

    assert one.view.quit_label() == said("tray_quit", window_ui.TRAY_TEXT)
    assert one.view.said["window_settings"] == said("window_settings")


# --------------------------------------------------------------------------
# The transcript
# --------------------------------------------------------------------------


def test_a_finished_turn_is_two_rows_you_and_the_assistant(built: Any) -> None:
    one = built()
    one.window.turn(turn())
    one.settle()

    assert one.view.rows == [("you", "saat kaç"), ("it", "On beş kırk iki.")]
    assert one.view.row_label("you") == said("you_said", status.TEXT)
    assert one.view.row_label("it") == said("it_said", status.TEXT)
    assert one.view.row_label("notice") == ""


def test_a_turn_with_nothing_heard_is_no_row_and_a_missed_one_shows_the_number(
    built: Any,
) -> None:
    one = built()
    one.window.turn(Turn())
    one.window.turn(Turn(missed=True, confidence=0.4))
    one.settle()

    assert one.view.rows[0] == ("you", said("not_caught", status.TEXT).format(confidence="0.40"))
    assert len(one.view.rows) == 2


def test_the_oldest_rows_go_after_two_hundred(built: Any) -> None:
    one = built()
    for index in range(TRANSCRIPT_ROWS):
        one.window.turn(turn(heard=f"q{index}", said_=f"a{index}"))
    one.settle()

    assert len(one.view.rows) == TRANSCRIPT_ROWS
    assert one.view.rows[0] == ("you", f"q{TRANSCRIPT_ROWS // 2}")
    assert one.view.version == TRANSCRIPT_ROWS * 2


def test_a_notice_is_a_row_of_its_own(built: Any) -> None:
    one = built()
    one.window.notice("the model does not call tools")
    one.settle()

    assert one.view.rows == [("notice", "the model does not call tools")]


def test_a_quiet_microphone_is_said_once_like_the_line(built: Any) -> None:
    one = built()
    one.window.microphone_level(None)
    one.window.microphone_level(-45.0)
    one.window.microphone_level(-47.0)
    one.window.microphone_level(-30.0)
    one.window.microphone_level(-45.0)
    one.settle()

    sentence = said("microphone_quiet", status.TEXT)
    assert one.view.rows == [
        ("notice", sentence.format(level=-45, quiet=int(QUIET_DBFS))),
        ("notice", sentence.format(level=-45, quiet=int(QUIET_DBFS))),
    ]


# --------------------------------------------------------------------------
# The orb
# --------------------------------------------------------------------------


def test_a_level_feeds_the_orb_and_a_change_of_state_lets_it_go(built: Any) -> None:
    one = built()
    one.window.state(State.USER_SPEAKING)
    one.window.level(-10.0)
    one.settle()
    assert one.view.orb.level.target == 1.0

    one.window.state(State.IDLE)
    one.settle()
    assert one.view.orb.level.target == 0.0


def test_the_frame_advances_with_the_time_it_is_given(built: Any) -> None:
    one = built()
    one.window.state(State.IDLE)
    one.settle()

    first = one.view.frame(0.0).angles
    second = one.view.frame(1.0).angles

    assert first != second


# --------------------------------------------------------------------------
# The wizard's page
# --------------------------------------------------------------------------


async def test_the_page_holds_what_was_said_and_the_open_question() -> None:
    one = Built(asyncio.get_running_loop())
    try:
        one.window.wizard(True)
        one.window.say("welcome")
        answer = one.window.ask("choose", "which?", [Option("a", "A"), Option("b", "B")])
        one.settle()

        page = one.view.wizard
        assert page is not None
        assert page.lines == ["welcome"]
        assert (page.kind, page.question) == ("choose", "which?")
        assert [option.value for option in page.options] == ["a", "b"]
        assert page.answer is answer and page.open

        one.window.wizard(False)
        one.settle()
        assert one.view.wizard is None
    finally:
        one.window.stop()


async def test_an_answer_from_the_tk_thread_resolves_the_future_on_the_loop() -> None:
    one = Built(asyncio.get_running_loop())
    try:
        one.window.wizard(True)
        answer = one.window.ask("ask", "address?", ())
        one.settle()

        def continue_pressed() -> None:
            page = one.view.wizard
            assert page is not None
            taken = one.view.take_answer()
            assert taken is answer and not page.open
            one.window.answer(taken, "http://example.test/v1")

        await asyncio.to_thread(continue_pressed)

        assert await asyncio.wait_for(answer, 2) == "http://example.test/v1"
    finally:
        one.window.stop()


async def test_cancel_answers_none_and_a_quitting_window_answers_none_at_once() -> None:
    one = Built(asyncio.get_running_loop())
    try:
        one.window.wizard(True)
        first = one.window.ask("secret", "key?", ())
        one.settle()
        await asyncio.to_thread(lambda: one.window.answer(one.view.take_answer(), None))
        assert await asyncio.wait_for(first, 2) is None

        await asyncio.to_thread(one.window.quit)
        second = one.window.ask("ask", "voice?", ())
        assert second.done() and second.result() is None
    finally:
        one.window.stop()


async def test_take_answer_gives_nothing_when_no_question_is_open() -> None:
    one = Built(asyncio.get_running_loop())
    try:
        assert one.view.take_answer() is None
        one.window.wizard(True)
        one.settle()
        assert one.view.take_answer() is None
    finally:
        one.window.stop()


async def test_the_prompter_asks_the_wizards_questions_through_the_window() -> None:
    """The same keys the terminal prompter uses, worded by the same table;
    each question is a page, each answer the page's."""
    one = Built(asyncio.get_running_loop())
    try:
        prompter = WindowPrompter(one.window, text=WIZARD_TEXT)
        one.window.wizard(True)
        prompter.say("key_url", url="https://keys.example.test")

        async def answer_each() -> None:
            for value in ("gemini", "sk-secret", "http://x/v1"):
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    page = one.view.wizard
                    if page is not None and page.open:
                        break
                else:
                    raise AssertionError("no question opened")
                one.window.answer(one.view.take_answer(), value)

        answering = asyncio.create_task(answer_each())
        chosen = await prompter.choose("provider", [Option("gemini", "Gemini Live")])
        secret = await prompter.secret("api_key")
        address = await prompter.ask("base_url")
        await answering
        one.settle()

        assert (chosen, secret, address) == ("gemini", "sk-secret", "http://x/v1")
        page = one.view.wizard
        assert page is not None
        assert page.lines == [WIZARD_TEXT["key_url"].format(url="https://keys.example.test")]
        assert page.question == WIZARD_TEXT["base_url"]
    finally:
        one.window.stop()


# --------------------------------------------------------------------------
# The buttons, and closing
# --------------------------------------------------------------------------


async def test_the_buttons_reach_the_loop_from_the_tk_thread() -> None:
    one = Built(asyncio.get_running_loop())
    try:
        tk_thread: list[int] = []

        def clicked(action: Any) -> None:
            tk_thread.append(threading.get_ident())
            action()

        await asyncio.to_thread(clicked, one.window.toggle)
        await asyncio.to_thread(clicked, one.window.settings)
        await asyncio.to_thread(clicked, one.window.quit)
        await asyncio.sleep(0)

        assert (one.toggles, one.settings, one.quits) == (1, 1, 1)
        assert one.threads == [threading.get_ident()] * 3
        assert threading.get_ident() not in tk_thread
    finally:
        one.window.stop()


def test_closing_hides_with_a_tray_and_quits_without(built: Any) -> None:
    assert built(tray=True).window.hides_on_close() is True
    assert built(tray=False).window.hides_on_close() is False


def test_the_switch_does_nothing_until_it_has_a_target_and_then_calls_it() -> None:
    switch = Switch()
    switch()

    calls: list[str] = []
    switch.target = lambda: calls.append("toggled")
    switch()

    assert calls == ["toggled"]


def test_an_unknown_message_is_a_bug_not_a_silent_drop() -> None:
    view = View(TR)

    with pytest.raises(ValueError, match="unknown window message"):
        view.apply(("dance",))


# --------------------------------------------------------------------------
# The real thing, once
# --------------------------------------------------------------------------


def test_the_real_panel_comes_up_and_goes_down() -> None:
    """Built for real on its own thread, told a state, a turn and a level,
    given a few ticks, and taken down. Skipped where Tk has no display."""
    tkinter = pytest.importorskip("tkinter")
    loop = asyncio.new_event_loop()
    real: Window | None = None
    try:
        real = Window(
            TR,
            loop=loop,
            on_toggle=lambda: None,
            on_quit=lambda: None,
            on_settings=lambda: None,
            tray=False,
        )
        try:
            real.start()
        except WindowError as refused:
            if isinstance(refused.__cause__, tkinter.TclError):
                pytest.skip(f"no display: {refused.__cause__}")
            raise
        real.state(State.SPEAKING)
        real.turn(turn())
        real.level(-20.0)
        real.wizard(True)
        real.say("welcome")
        question = real.ask("choose", "which?", [Option("a", "A")])
        time.sleep(0.3)
        real.wizard(False)
        time.sleep(0.2)
        assert real.pending() == 0
        assert real.view.state is State.SPEAKING
        assert not question.done()
    finally:
        if real is not None:
            real.stop()
        loop.close()
