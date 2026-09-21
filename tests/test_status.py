"""The terminal status line - the whole interface phase 1 has (item 1.11).

There is no window and no tray icon yet, so this one line is where the user
finds out whether the assistant heard them, is thinking, or is talking. Three
claims are worth testing.

**It says what is happening, in the user's language.** The words come from the
locale pack with the English constants of `TEXT` behind them, exactly like
every other sentence in the product (section 3.12).

**A turn leaves something behind.** The line itself is overwritten as the state
changes; what was heard, what was answered and what it cost scroll past above
it, which is the only record the user gets in phase 1.

**A terminal that is not a terminal still works.** `live-assistant run > run.log`
redirects the output to a file, and a status line that only knew how to draw
itself on a console would take the whole assistant down with it.

**The session and its minutes are on the line** (plan.md 4.2, L1.7). A live
model bills by the minute while a session is open, silent or not, so the
line says whether one is open and how many minutes this run has spent -
counted here, from the clock, since the usage rows do not carry minutes yet.
And a microphone the server hears too quietly (D18) is said once, on a line
that stays.
"""

from __future__ import annotations

import re
import threading
from io import StringIO

import pytest
from rich.console import Console

from assistant.app import State, Turn
from assistant.audio.capture import DEFAULT_TOGGLE_HOTKEY, QUIET_DBFS
from assistant.live.base import Usage
from assistant.locales import Locale
from assistant.ui.status import TEXT, SessionMinutes, StatusLine, label_key, spell

TURKISH = Locale(
    code="tr",
    name="Türkçe",
    stt_language="tr",
    voices={},
    ui={
        "state_user_speaking": "seni duyuyor",
        "state_idle": "hazır",
        "paused": "Dinlemiyorum. {toggle} açar.",
        "you_said": "sen",
        "session_open": "oturum açık",
    },
)


class Clock:
    """A clock the test moves by hand, in seconds."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


CONTROL = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class Screen:
    """A console that writes into a string instead of onto a terminal."""

    def __init__(self, *, terminal: bool = True) -> None:
        self.written = StringIO()
        # Wide enough that nothing wraps: a line broken in the middle of a
        # sentence would fail an assertion about the sentence.
        self.console = Console(file=self.written, force_terminal=terminal, width=200, no_color=True)

    def __str__(self) -> str:
        """What a person would read, with the cursor moves taken out."""
        return CONTROL.sub("", self.written.getvalue())


@pytest.fixture
def screen() -> Screen:
    return Screen()


def line(screen: Screen, locale: Locale | None = None, clock: Clock | None = None) -> StatusLine:
    return StatusLine(
        locale or Locale("en", "English", "en", {}, {}),
        console=screen.console,
        clock=clock if clock is not None else Clock(),
    )


def shown(screen: Screen) -> str:
    """The line as it is now: what follows the last carriage return."""
    return str(screen).rstrip().rpartition(chr(13))[2]


# --------------------------------------------------------------------------
# The line itself
# --------------------------------------------------------------------------


def test_every_state_the_machine_can_be_in_has_something_to_show() -> None:
    """A state added later without a label would leave the user looking at a
    line that says nothing while the assistant does something."""
    for state in State:
        assert label_key(state) in TEXT, state
    assert TEXT["state_sleeping"] == "asleep"


def test_the_line_says_what_the_assistant_is_doing(screen: Screen) -> None:
    """Read inside the block rather than after it: a line that only appears
    once the program has ended is not a status line."""
    with line(screen) as status:
        status.state(State.USER_SPEAKING)

        assert TEXT["state_user_speaking"] in str(screen)


def test_the_line_is_on_screen_before_anything_has_happened(screen: Screen) -> None:
    """Starting up draws it straight away. A terminal that stays blank until
    the first key is pressed reads as a program that failed to start."""
    with line(screen):
        assert TEXT["state_idle"] in str(screen)


def test_the_line_keeps_the_key_in_view(screen: Screen) -> None:
    """The one thing a new user needs to know is which key switches the
    microphone, and there is nowhere else to put it."""
    with line(screen) as status:
        status.state(State.IDLE)

    assert "Ctrl+Alt+H" in str(screen)


def test_before_the_microphone_reports_itself_the_line_says_paused(screen: Screen) -> None:
    """Honest until told otherwise: the capture reports its mode at start."""
    with line(screen) as status:
        status.state(State.IDLE)

    assert "Not listening" in str(screen)


def test_the_words_are_the_pack_s_and_not_the_code_s(screen: Screen) -> None:
    with line(screen, TURKISH) as status:
        status.state(State.USER_SPEAKING)

    assert "seni duyuyor" in str(screen)
    assert TEXT["state_user_speaking"] not in str(screen)


def test_a_sentence_the_pack_leaves_out_is_still_said(screen: Screen) -> None:
    """The end of the chain of section 3.12. The Turkish pack above translates
    four keys; the rest have to come out in English rather than not at all."""
    with line(screen, TURKISH) as status:
        status.state(State.SPEAKING)

    assert TEXT["state_speaking"] in str(screen)


def test_the_speech_model_is_loading_before_it_can_say_anything_else(screen: Screen) -> None:
    """Whisper takes seconds to load. A blank terminal during them reads as a
    program that failed to start."""
    with line(screen) as status:
        status.starting()

    assert TEXT["loading_speech"] in str(screen)


def test_the_model_is_being_checked_before_anything_else_loads(screen: Screen) -> None:
    """The probe of 2.6 takes a second or two on the network; the user is
    told what the wait is."""
    with line(screen) as status:
        status.checking_model()

    assert TEXT["checking_model"] in str(screen)


def test_a_notice_stays_on_screen_when_the_state_moves_on(screen: Screen) -> None:
    """The line is overwritten; a notice is not. It is for the one thing the
    user should read once - that the model failed the probe."""
    with line(screen) as status:
        status.notice("The model does not call tools.")
        status.state(State.USER_SPEAKING)
        status.state(State.IDLE)

    assert "The model does not call tools." in str(screen)


def test_the_hotkey_is_spelled_the_way_a_keyboard_is() -> None:
    """`pynput` writes it for a parser; the user reads it off their keyboard."""
    assert spell(DEFAULT_TOGGLE_HOTKEY) == "Ctrl+Alt+H"
    assert spell("<ctrl>+<shift>+k") == "Ctrl+Shift+K"


def test_the_line_says_when_the_microphone_is_live_without_a_key(screen: Screen) -> None:
    """The only thing on screen that answers it. Left unsaid, the mode is one
    the user forgets is on in a room with other people in it."""
    with line(screen) as status:
        status.state(State.IDLE)
        status.hands_free(True)

    assert TEXT["hands_free"].format(toggle="Ctrl+Alt+H") in str(screen)


def test_switching_it_off_puts_the_key_back_on_the_line(screen: Screen) -> None:
    with line(screen) as status:
        status.hands_free(True)
        status.hands_free(False)

    assert str(screen).rstrip().endswith("Ctrl+C stops.")


def test_the_mode_changes_without_the_state_changing(screen: Screen) -> None:
    """The two are independent: the assistant is thinking about the same thing
    whether or not the microphone stayed open behind it."""
    with line(screen) as status:
        status.state(State.USER_SPEAKING)
        status.hands_free(True)

    assert TEXT["state_user_speaking"] in shown(screen)
    assert "Ctrl+Alt+H stops listening" in shown(screen)


def test_a_pack_that_says_nothing_about_the_mode_still_says_something(
    screen: Screen,
) -> None:
    """`TURKISH` above translates four keys and this is not one of them."""
    with line(screen, TURKISH) as status:
        status.hands_free(True)

    assert "Ctrl+Alt+H" in str(screen)


# --------------------------------------------------------------------------
# The session and its minutes (plan.md 4.2, L1.7)
# --------------------------------------------------------------------------


def test_the_line_says_no_session_is_open_before_one_is(screen: Screen) -> None:
    """Honest from the start: the door is open and the meter is not running."""
    with line(screen):
        assert TEXT["session_closed"] in shown(screen)
        assert TEXT["session_minutes"].format(minutes=0) in shown(screen)


def test_the_line_says_when_a_session_opens_and_when_it_closes(screen: Screen) -> None:
    with line(screen) as status:
        status.session(True)
        assert TEXT["session_open"] in shown(screen)
        assert TEXT["session_closed"] not in shown(screen)

        status.session(False)
        assert TEXT["session_closed"] in shown(screen)


def test_the_minutes_are_the_time_sessions_were_open_this_run(screen: Screen) -> None:
    """What the live model bills by (D5): open time, silent or not, whole
    minutes, added up across the sessions of one run."""
    clock = Clock()
    with line(screen, clock=clock) as status:
        status.session(True)
        clock.now += 150
        status.session(False)
        status.session(True)
        clock.now += 60
        status.session(False)

        assert TEXT["session_minutes"].format(minutes=3) in shown(screen)
        assert status.minutes == 3.5


def test_the_minutes_of_the_session_under_way_count_too(screen: Screen) -> None:
    """The line is redrawn at every change of state, and the number on it
    then includes the session that is still open."""
    clock = Clock()
    with line(screen, clock=clock) as status:
        status.session(True)
        clock.now += 125
        status.state(State.SPEAKING)

        assert TEXT["session_minutes"].format(minutes=2) in shown(screen)


def test_the_session_survives_a_change_of_state_and_of_mode(screen: Screen) -> None:
    with line(screen) as status:
        status.session(True)
        status.state(State.USER_SPEAKING)
        status.hands_free(True)

        assert TEXT["session_open"] in shown(screen)
        assert TEXT["state_user_speaking"] in shown(screen)


def test_the_session_is_said_in_the_pack_s_words(screen: Screen) -> None:
    with line(screen, TURKISH) as status:
        status.session(True)

    assert "oturum açık" in str(screen)
    assert TEXT["session_open"] not in str(screen)


def test_opening_twice_or_closing_twice_counts_once() -> None:
    """The state machine may say the same thing twice on a reconnect; the
    meter must not lose the minutes in between or double them."""
    clock = Clock()
    meter = SessionMinutes(clock=clock)

    meter.told(True)
    meter.told(True)
    clock.now += 60
    meter.told(False)
    meter.told(False)

    assert meter.minutes == 1.0
    assert meter.open is False


# --------------------------------------------------------------------------
# A microphone the server hears too quietly (D18)
# --------------------------------------------------------------------------


def test_a_quiet_microphone_is_said_once_on_a_line_that_stays(screen: Screen) -> None:
    """Measured 2026-09-18: the owner's array sends -45 dBFS, and a quiet
    microphone is the first thing to check when the model hears another
    language. Said once, not after every session."""
    with line(screen) as status:
        status.microphone_level(-45.2)
        status.microphone_level(-46.0)
        status.state(State.IDLE)

    said = TEXT["microphone_quiet"].format(level=-45, quiet=int(QUIET_DBFS))
    assert str(screen).count(said) == 1


def test_a_microphone_that_is_loud_enough_is_not_mentioned(screen: Screen) -> None:
    with line(screen) as status:
        status.microphone_level(-30.0)
        status.microphone_level(None)

    assert TEXT["microphone_quiet"].split("{")[0] not in str(screen)


def test_a_microphone_that_went_quiet_again_is_said_again(screen: Screen) -> None:
    """Fixed and then broken again - a new headset, a Windows update -
    deserves a new line."""
    with line(screen) as status:
        status.microphone_level(-45.0)
        status.microphone_level(-30.0)
        status.microphone_level(-45.0)

    assert str(screen).count(TEXT["microphone_quiet"].split("{")[0]) == 2


def test_the_quiet_microphone_is_said_in_the_pack_s_words(screen: Screen) -> None:
    pack = Locale("tr", "Türkçe", "tr", {}, {"microphone_quiet": "mikrofon kısık ({level} dBFS)"})

    with line(screen, pack) as status:
        status.microphone_level(-45.0)

    assert "mikrofon kısık (-45 dBFS)" in str(screen)


# --------------------------------------------------------------------------
# What a turn leaves behind
# --------------------------------------------------------------------------


def test_a_finished_turn_shows_what_was_heard_and_what_was_answered(screen: Screen) -> None:
    with line(screen) as status:
        status.turn(Turn(heard="saat kaç", said="Üç buçuk."))

    assert "saat kaç" in str(screen)
    assert "Üç buçuk." in str(screen)


def test_a_turn_reports_what_it_spent(screen: Screen) -> None:
    """Item 1.11 puts the tokens in the log; showing them is what makes a model
    that costs ten times as much noticeable on the day it is chosen."""
    with line(screen) as status:
        status.turn(Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10)))

    assert TEXT["turn_cost"].format(input=300, output=10) in str(screen)


def test_a_turn_that_was_missed_shows_the_number_instead_of_the_words(screen: Screen) -> None:
    """There is no transcript worth printing - that is what missed means - and
    the number is what tells the user whether speaking up would have helped."""
    with line(screen) as status:
        status.turn(Turn(said="I did not catch that.", missed=True, confidence=0.55))

    shown = str(screen)
    assert "0.55" in shown
    assert "I did not catch that." in shown


def test_a_missed_turn_with_no_number_is_still_shown(screen: Screen) -> None:
    with line(screen) as status:
        status.turn(Turn(said="I did not catch that.", missed=True, confidence=None))

    assert "I did not catch that." in str(screen)


def test_a_missed_turn_says_so_in_the_user_s_language(screen: Screen) -> None:
    pack = Locale("tr", "Türkçe", "tr", {}, {"not_caught": "(anlaşılmadı - güven {confidence})"})

    with line(screen, pack) as status:
        status.turn(Turn(said="Seni anlayamadım.", missed=True, confidence=0.55))

    assert "anlaşılmadı" in str(screen)


def test_a_turn_that_heard_nothing_is_not_written_down(screen: Screen) -> None:
    """A key tapped by accident, or a recording of silence. Neither is a turn
    the user had, and a screen full of empty ones hides the real ones."""
    with line(screen) as status:
        before = str(screen)
        status.turn(Turn())
        after = str(screen)

    assert after == before


def test_a_turn_that_failed_does_not_claim_to_have_been_free(screen: Screen) -> None:
    """A failed turn reports no tokens (`app.py`). `0 in, 0 out` beside an
    error message reads as a price rather than as the absence of one."""
    with line(screen) as status:
        status.turn(Turn(heard="saat kaç", said="Sağlayıcıya bağlanamadım."))

    assert TEXT["turn_cost"].format(input=0, output=0) not in str(screen)


def test_who_said_which_half_is_written_in_the_user_s_language(screen: Screen) -> None:
    with line(screen, TURKISH) as status:
        status.turn(Turn(heard="saat kaç", said="Üç buçuk."))

    assert "sen" in str(screen)


def test_the_line_does_not_start_a_thread_to_redraw_itself(screen: Screen) -> None:
    """It is redrawn when the state changes and at no other time. An animation
    thread would spend the fifty milliseconds rule 4 of section 3.1 gives the
    whole event loop on a line that has not moved."""
    alone = threading.active_count()

    with line(screen) as status:
        status.state(State.USER_SPEAKING)

        assert threading.active_count() == alone


# --------------------------------------------------------------------------
# Terminals that are not terminals
# --------------------------------------------------------------------------


def test_output_that_is_a_file_rather_than_a_console_still_gets_the_turns() -> None:
    """`live-assistant run > run.log`. Nothing here may depend on a cursor that can
    be moved back to the start of a line."""
    written = Screen(terminal=False)

    with line(written) as status:
        status.state(State.USER_SPEAKING)
        status.turn(Turn(heard="saat kaç", said="Üç buçuk."))

    assert "saat kaç" in str(written)
    assert "Üç buçuk." in str(written)


# --------------------------------------------------------------------------
# The Screen protocol (plan.md D20): the line and the window are one thing to `run`
# --------------------------------------------------------------------------


def test_the_line_is_a_screen_and_a_level_costs_it_nothing(screen: Screen) -> None:
    """`run` knows the line and the window as one `Screen` (D20); the
    line takes the sound's level and does nothing with it."""
    from assistant.ui.status import Screen as ScreenProtocol

    with line(screen) as shown_line:
        face: ScreenProtocol = shown_line
        before = str(screen)
        face.level(-20.0)
        face.level(-90.0)

        assert isinstance(shown_line, ScreenProtocol)
        assert str(screen) == before
