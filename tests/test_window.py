"""The window (`ui/window.py`, plan.md D20): the product's face, tested without a screen.

What is claimed: every `Screen` call is a queue put that returns at once -
the loop never waits on the window (spec section 2); the view shows the
state, the mode, the session and the phase in the pack's words; finished
turns become rows and the oldest go after two hundred; the conversation is
folded at the start, unfolds and folds at the header, and unfolds for a
notice (D33); the quiet-microphone sentence is said once, like the line; a
level feeds the orb; the wizard's
page holds what was said and the open question, and an answer from the Tk
thread resolves the future on the loop; buttons reach the loop through
`call_soon_threadsafe`; minimising hides into the tray when there is one,
and the close box quits (D34). The real Tk panel is built in
`test_the_real_panel_comes_up_and_goes_down`, and once more on Windows to
be minimised, brought back and closed (skipped where there is no display).
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from allie import config, locales
from allie.app import State, Turn
from allie.audio.capture import QUIET_DBFS
from allie.setup_wizard import TEXT as WIZARD_TEXT
from allie.setup_wizard import Option
from allie.ui import orb, status
from allie.ui import window as window_ui
from allie.ui.window import (
    BACKGROUND,
    TEXT,
    TICK_MS,
    TRANSCRIPT_ROWS,
    Switch,
    View,
    Window,
    WindowError,
    WindowPrompter,
    plate,
    split_row,
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
    assert one.view.label() == said("starting_up", status.TEXT)

    one.window.state(State.IDLE)
    one.settle()
    assert one.view.label() == label_of(State.IDLE)


def test_a_start_that_failed_says_so_where_the_state_would_be(built: Any) -> None:
    """2026-09-23: over a microphone Teams was holding, the line went on
    saying "ready". A failed start replaces whatever it said."""
    one = built()
    one.window.state(State.IDLE)
    one.window.starting()
    one.window.failed()
    one.settle()

    assert one.view.label() == said("window_failed")
    assert one.view.label() != label_of(State.IDLE)


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


def test_the_orbs_corners_say_whether_a_session_is_open_and_the_minutes(built: Any) -> None:
    """Two readouts rather than one line joined by a dot (D33)."""
    one = built()
    closed = said("session_closed", status.TEXT)
    opened = said("session_open", status.TEXT)
    minutes = said("session_minutes", status.TEXT)

    assert one.view.readouts() == (closed, minutes.format(minutes=0))

    one.window.session(True)
    one.settle()
    one.clock.now += 150
    assert one.view.readouts() == (opened, minutes.format(minutes=2))

    one.window.session(False)
    one.settle()
    assert one.view.readouts() == (closed, minutes.format(minutes=2))


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


def test_a_turn_with_nothing_heard_is_no_row(built: Any) -> None:
    """The status line's rule: a key tapped by accident or a question withdrawn
    mid-turn is not a turn the user had."""
    one = built()
    one.window.turn(Turn())
    one.window.turn(Turn(heard="saat kaç", said="Üç buçuk."))
    one.settle()

    assert one.view.rows == [("you", "saat kaç"), ("it", "Üç buçuk.")]


def test_a_turn_that_looked_something_up_has_the_searches_between(built: Any) -> None:
    """D29: the row Google's terms ask for, in the chat, not a notice."""
    one = built()
    one.window.turn(
        Turn(heard="BIST kaç", said="13.337 puan.", searched=("BIST 100", "BIST kapanış"))
    )
    one.settle()

    assert one.view.rows == [
        ("you", "BIST kaç"),
        ("searched", "BIST 100 · BIST kapanış"),
        ("it", "13.337 puan."),
    ]


def test_the_oldest_rows_go_after_two_hundred(built: Any) -> None:
    one = built()
    for index in range(TRANSCRIPT_ROWS):
        one.window.turn(turn(heard=f"q{index}", said_=f"a{index}"))
    one.settle()

    assert len(one.view.rows) == TRANSCRIPT_ROWS
    assert one.view.rows[0] == ("you", f"q{TRANSCRIPT_ROWS // 2}")
    assert one.view.version == TRANSCRIPT_ROWS * 2


def test_a_notice_is_a_row_of_its_own_and_unfolds_the_conversation(built: Any) -> None:
    """A notice is there to be read - "could not start, the reason is
    below" - so it does not stay in a folded accordion (D33)."""
    one = built()
    one.window.notice("the model does not call tools")
    one.settle()

    assert one.view.rows == [("notice", "the model does not call tools")]
    assert one.view.log_open is True


# --------------------------------------------------------------------------
# The accordion (D33)
# --------------------------------------------------------------------------


def test_the_conversation_starts_folded_and_the_header_unfolds_and_folds_it() -> None:
    """At every start the window is the orb and nothing more."""
    view = View(TR)
    assert view.log_open is False

    view.toggle_log()
    assert view.log_open is True
    view.toggle_log()
    assert view.log_open is False


def test_a_turn_is_written_into_the_conversation_without_unfolding_it(built: Any) -> None:
    one = built()
    one.window.turn(turn())
    one.settle()

    assert one.view.rows
    assert one.view.log_open is False


def test_the_window_wants_its_full_height_for_the_conversation_or_the_wizard() -> None:
    """Folded, the window is as tall as the orb and the buttons; the
    conversation and the wizard's page each need the full height."""
    view = View(TR)
    assert view.wants_room() is False

    view.toggle_log()
    assert view.wants_room() is True
    view.toggle_log()

    view.apply(("wizard", True))
    assert view.wants_room() is True
    view.apply(("wizard", False))
    assert view.wants_room() is False


def test_the_header_wears_the_packs_word() -> None:
    assert View(TR).said["window_conversation"] == said("window_conversation")
    assert said("window_conversation") == "Konuşma"


# --------------------------------------------------------------------------
# The settings list's rows
# --------------------------------------------------------------------------


def test_a_settings_row_is_split_into_what_it_is_and_its_value() -> None:
    assert split_row("Asistan: Jarvis") == ("Asistan", "Jarvis")
    assert split_row("Evet ya da hayırını kim duysun: Google'ın tanıyıcısı") == (
        "Evet ya da hayırını kim duysun",
        "Google'ın tanıyıcısı",
    )


def test_only_the_first_colon_splits_a_row_and_a_row_without_one_stays_whole() -> None:
    """A microphone's name can carry a colon of its own; a pack that words
    a row without one gets a single line, not a broken one."""
    assert split_row("Mikrofon: Mic: USB, WASAPI") == ("Mikrofon", "Mic: USB, WASAPI")
    assert split_row("gemini-3.8-live") == ("", "gemini-3.8-live")
    assert split_row("ratio 16:9") == ("", "ratio 16:9")


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


async def test_the_settings_list_is_a_menu_page_with_change_and_close() -> None:
    """D32: the list of settings is the same page - a `menu` question, no
    filter box - and its buttons say Change and Close; every other question
    keeps Continue and Cancel."""
    one = Built(asyncio.get_running_loop())
    try:
        prompter = WindowPrompter(one.window, text=WIZARD_TEXT)
        one.window.wizard(True)
        rows = [Option("assistant", "Assistant: Vesper"), Option("model", "Model: fast")]

        async def pick_then_close() -> None:
            for value in ("model", None):
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    page = one.view.wizard
                    if page is not None and page.open:
                        break
                else:
                    raise AssertionError("no list opened")
                kinds.append(page.kind)
                buttons.append(one.view.page_buttons())
                one.window.answer(one.view.take_answer(), value)

        kinds: list[str] = []
        buttons: list[tuple[str, str]] = []
        answering = asyncio.create_task(pick_then_close())
        chosen = await asyncio.wait_for(prompter.menu("settings", rows), 5)
        closed = await asyncio.wait_for(prompter.menu("settings", rows), 5)
        await answering

        assert (chosen, closed) == ("model", None)
        assert kinds == ["menu", "menu"]
        assert buttons == [(said("wizard_change"), said("wizard_close"))] * 2
        one.window.ask("choose", "which?", [Option("a", "A")])
        one.settle()
        assert one.view.page_buttons() == (said("wizard_continue"), said("wizard_cancel"))
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


def test_minimising_hides_into_the_tray_only_when_there_is_one(built: Any) -> None:
    """D34: with no tray to bring it back from, a hidden window would be
    lost - it stays on the taskbar."""
    assert built(tray=True).window.hides_on_minimise() is True
    assert built(tray=False).window.hides_on_minimise() is False


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
# The look: what can be checked without a screen
# --------------------------------------------------------------------------


def test_the_window_wears_the_products_name_and_not_its_folders() -> None:
    """The title bar says Allie (U3); the folder, the Credential Manager
    entry and the environment variable keep the name the machine already
    has, or the settings and the key on it would be lost."""
    assert config.APP_TITLE == "Allie"
    assert config.APP_NAME == "allie"
    assert config.KEYRING_SERVICE == "allie"


def test_the_tick_is_sixty_frames_a_second() -> None:
    """Fifteen milliseconds, not sixteen: Windows rounds a wait up to the
    next 15.6 ms and sixteen would come back at thirty-one (U1)."""
    assert TICK_MS == 15
    assert 1000 / TICK_MS >= 60


def test_the_arc_the_eye_follows_is_moved_every_frame() -> None:
    """The heavy inner arc turned 0.6 of a degree a frame at thirty frames
    and cleared the half degree it takes to be redrawn. At sixty it turns
    0.3 - under that old threshold, which would have drawn it every other
    frame and handed back the thirty frames U1 was about. The threshold is
    still there, and still under half a pixel at the orb's size: an arc it
    holds back is one that did not visibly move."""
    a_frame = abs(orb.RINGS[0].speed) * orb.SPEED[State.IDLE] * (TICK_MS / 1000)

    assert a_frame == pytest.approx(0.27, abs=0.05)
    assert a_frame > window_ui.MIN_TURN_DEGREES
    assert math.radians(window_ui.MIN_TURN_DEGREES) * (window_ui.ORB_HEIGHT / 2) < 0.5


def test_a_button_plate_is_a_rounded_shape_on_the_windows_own_dark() -> None:
    """Drawn by Pillow because Tk has neither a rounded corner nor an
    anti-aliased one (U4): the middle is the fill, the corner is the
    background it was painted on, and the corner is not a hard step."""
    fill = (31, 106, 196)
    image = plate(80, 32, 9, fill, None, BACKGROUND)

    assert image.size == (80, 32)
    assert image.mode == "RGB"
    assert image.getpixel((40, 16)) == fill
    assert image.getpixel((0, 0)) == BACKGROUND

    corner = image.getpixel((2, 2))
    assert isinstance(corner, tuple)
    # A stepped corner would be one colour or the other; a smooth one is
    # part way between them on every channel.
    assert all(
        dark < shade < bright for dark, shade, bright in zip(BACKGROUND, corner, fill, strict=True)
    )


def test_a_plate_of_no_size_is_a_bug() -> None:
    with pytest.raises(ValueError, match="cannot be drawn"):
        plate(0, 20, 6, (10, 10, 10), None, BACKGROUND)


def test_every_button_answers_the_pointer_and_says_when_it_is_out_of_use() -> None:
    """A style's plates at rest, under the pointer and held down are three
    different colours, or the button would not answer at all."""
    for style in (window_ui.SOLID, window_ui.PLATE, window_ui.GHOST, window_ui.DANGER):
        assert len({style.fill, style.hover, style.press}) == 3


def test_one_button_is_filled_and_the_quiet_ones_are_words_on_the_ground() -> None:
    """One filled button a page (D33): Continue and Change are solid, the
    switch a plate, the rest ghosts until the pointer is on them."""
    assert window_ui.GHOST.fill == BACKGROUND
    assert window_ui.DANGER.fill == BACKGROUND
    assert window_ui.PLATE.fill != BACKGROUND
    assert sum(window_ui.SOLID.fill) > 3 * 200


def test_the_ground_has_no_colour_of_its_own() -> None:
    """The orb's colour is the only colour (D33): on the old blue-black
    only the idle blue looked at home. Neutral means the three channels
    within a hair of each other."""
    assert max(BACKGROUND) - min(BACKGROUND) <= 6
    assert max(BACKGROUND) < 20


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
        real.notice("a notice unfolds the conversation")
        time.sleep(0.4)
        assert real.view.log_open is True
        real.wizard(True)
        real.say("welcome")
        question = real.ask("choose", "which?", [Option("a", "A"), Option("b", "B")])
        time.sleep(0.3)
        real.say("saved")
        menu = real.ask("menu", "settings", [Option("x", "What: value"), Option("y", "whole")])
        time.sleep(0.3)
        secret = real.ask("secret", "key?", ())
        time.sleep(0.3)
        real.wizard(False)
        # A tick that fell over on any page would leave this undrained.
        real.state(State.IDLE)
        time.sleep(0.4)
        assert real.pending() == 0
        assert real.view.state is State.IDLE
        assert not question.done() and not menu.done() and not secret.done()
    finally:
        if real is not None:
            real.stop()
        loop.close()


def _top_level_of_this_process() -> int:
    """The real window's handle, found the way Windows lists windows: the
    one top-level Tk window this process owns."""
    import ctypes
    import os
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    each = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [each, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    found: list[int] = []

    def look(hwnd: int, _: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, name, 64)
        if owner.value == os.getpid() and name.value == "TkTopLevel":
            found.append(hwnd)
        return True

    user32.EnumWindows(each(look), 0)
    assert len(found) <= 1, "more than one Tk window up"
    return found[0] if found else 0


def test_the_real_window_minimises_into_the_tray_comes_back_and_closes() -> None:
    """D34 on a real window, driven the way Windows drives it: the minimise
    box hides it (no taskbar button - it is not visible at all), the tray's
    line brings it back in front and not minimised, and the close box quits
    rather than hides. Skipped off Windows and where Tk has no display."""
    tkinter = pytest.importorskip("tkinter")
    if sys.platform != "win32":
        pytest.skip("Windows' own window calls")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindowAsync.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    sw_minimize, wm_close = 6, 0x0010

    loop = asyncio.new_event_loop()
    quits: list[str] = []
    real: Window | None = None

    def until(done: Callable[[], bool]) -> bool:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            # The loop is run in steps, so that what the window posted to it
            # with `call_soon_threadsafe` gets done.
            loop.run_until_complete(asyncio.sleep(0.02))
            if done():
                return True
        return False

    try:
        real = Window(
            TR,
            loop=loop,
            on_toggle=lambda: None,
            on_quit=lambda: quits.append("quit"),
            on_settings=lambda: None,
            tray=True,
        )
        try:
            real.start()
        except WindowError as refused:
            if isinstance(refused.__cause__, tkinter.TclError):
                pytest.skip(f"no display: {refused.__cause__}")
            raise
        assert until(lambda: bool(user32.IsWindowVisible(_top_level_of_this_process())))
        hwnd = _top_level_of_this_process()

        user32.ShowWindowAsync(hwnd, sw_minimize)
        assert until(lambda: not user32.IsWindowVisible(hwnd))

        real.show()
        assert until(lambda: bool(user32.IsWindowVisible(hwnd)) and not user32.IsIconic(hwnd))

        user32.PostMessageW(hwnd, wm_close, 0, 0)
        assert until(lambda: quits == ["quit"])
        assert user32.IsWindowVisible(hwnd)
    finally:
        if real is not None:
            real.stop()
        loop.close()
