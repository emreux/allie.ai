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
"""

from __future__ import annotations

import re
import threading
from io import StringIO

import pytest
from rich.console import Console

from assistant.app import State, Turn
from assistant.audio.capture import DEFAULT_TOGGLE_HOTKEY
from assistant.live.base import Usage
from assistant.locales import Locale
from assistant.ui.status import TEXT, StatusLine, label_key, spell

TURKISH = Locale(
    code="tr",
    name="Türkçe",
    stt_language="tr",
    voices={},
    ui={
        "state_thinking": "düşünüyor",
        "state_idle": "hazır",
        "paused": "Dinlemiyorum. {toggle} açar.",
        "you_said": "sen",
    },
)

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


def line(screen: Screen, locale: Locale | None = None) -> StatusLine:
    return StatusLine(locale or Locale("en", "English", "en", {}, {}), console=screen.console)


# --------------------------------------------------------------------------
# The line itself
# --------------------------------------------------------------------------


def test_every_state_the_machine_can_be_in_has_something_to_show() -> None:
    """A state added later without a label would leave the user looking at a
    line that says nothing while the assistant does something."""
    for state in State:
        assert label_key(state) in TEXT, state


def test_the_line_says_what_the_assistant_is_doing(screen: Screen) -> None:
    """Read inside the block rather than after it: a line that only appears
    once the program has ended is not a status line."""
    with line(screen) as status:
        status.state(State.THINKING)

        assert TEXT["state_thinking"] in str(screen)


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
        status.state(State.THINKING)

    assert "düşünüyor" in str(screen)
    assert TEXT["state_thinking"] not in str(screen)


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
        status.state(State.LISTENING)
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
        status.state(State.THINKING)
        status.hands_free(True)

    shown = str(screen).rstrip().rpartition(chr(13))[2]
    assert TEXT["state_thinking"] in shown
    assert "Ctrl+Alt+H stops listening" in shown


def test_a_pack_that_says_nothing_about_the_mode_still_says_something(
    screen: Screen,
) -> None:
    """`TURKISH` above translates four keys and this is not one of them."""
    with line(screen, TURKISH) as status:
        status.hands_free(True)

    assert "Ctrl+Alt+H" in str(screen)


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

    assert "0" not in str(screen)


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
        status.state(State.THINKING)

        assert threading.active_count() == alone


# --------------------------------------------------------------------------
# Terminals that are not terminals
# --------------------------------------------------------------------------


def test_output_that_is_a_file_rather_than_a_console_still_gets_the_turns() -> None:
    """`live-assistant run > run.log`. Nothing here may depend on a cursor that can
    be moved back to the start of a line."""
    written = Screen(terminal=False)

    with line(written) as status:
        status.state(State.THINKING)
        status.turn(Turn(heard="saat kaç", said="Üç buçuk."))

    assert "saat kaç" in str(written)
    assert "Üç buçuk." in str(written)
