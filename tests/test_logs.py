"""The application log: what a turn cost, and nothing that could identify it.

Item 1.11 asks for one thing - the token count of every turn written down, so
that the cost report of section 6 has something to be built from in phase 2.4.
What the log does *not* contain is the more interesting half.

**The terminal belongs to the status line.** `loguru` installs a handler that
writes to stderr the moment it is imported. Left in place it would draw log
lines through the middle of the one line phase 1 has to show the user.

**Nothing said out loud is written to disk.** The numbers are; the words are
not. A voice assistant that keeps a plaintext transcript of every sentence
spoken near a microphone is a liability nobody asked for, and section 3.7's
retention rules do not arrive until phase 4.

**No API key can reach the file.** Phase 1 logs no text that could carry one,
and the sink is configured so that a traceback cannot smuggle one out in the
value of a local variable.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from loguru import logger

from assistant import logs
from assistant.app import Turn
from assistant.config import config_dir
from assistant.live.base import Usage
from assistant.logs import keep_secret, log_path, log_turn, setup_logging

KEY = "AIzaSyD-notarealkeyatall-000000000000000"


@pytest.fixture(autouse=True)
def close_the_log() -> Iterator[None]:
    """No test leaves a handler behind holding a file open, and no test
    leaves a secret behind for the next one to mask."""
    kept = set(logs._SECRETS)
    logs._SECRETS.clear()
    yield
    logger.remove()
    logs._SECRETS.clear()
    logs._SECRETS.update(kept)


@pytest.fixture
def log(tmp_path: Path) -> Path:
    """A log file of this test's own, instead of the machine's."""
    return setup_logging(path=tmp_path / "logs" / "assistant.log")


def read(path: Path) -> str:
    """What ended up on disk. Removing the handler first closes the file."""
    logger.remove()
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Where it goes
# --------------------------------------------------------------------------


def test_the_log_goes_to_a_file(log: Path) -> None:
    logger.info("something happened")

    assert "something happened" in read(log)


def test_nothing_writes_on_the_terminal_any_more(tmp_path: Path) -> None:
    """`loguru` starts with a handler on stderr. The status line is the only
    thing allowed to draw there, so starting up has to take that handler away."""
    terminal = StringIO()
    logger.add(terminal)

    setup_logging(path=tmp_path / "assistant.log")
    logger.info("something happened")

    assert terminal.getvalue() == ""


def test_the_directory_is_made_on_the_first_run(tmp_path: Path) -> None:
    """Nothing creates `%LOCALAPPDATA%\\live-assistant\\logs` before this does."""
    path = setup_logging(path=tmp_path / "never" / "existed" / "assistant.log")
    logger.info("something happened")

    assert read(path).strip()


def test_starting_twice_does_not_write_everything_twice(tmp_path: Path) -> None:
    """Two handlers on one file is one duplicated line per entry, which reads
    as the assistant having done everything twice."""
    path = tmp_path / "assistant.log"
    setup_logging(path=path)
    setup_logging(path=path)

    logger.info("something happened")

    assert read(path).count("something happened") == 1


def test_the_log_is_written_in_the_language_it_is_read_in(log: Path) -> None:
    """Windows opens files in a legacy code page unless told otherwise, and
    half the Turkish alphabet has no place in it."""
    logger.info("Türkçe çıktı: ağır ışık")

    assert "Türkçe çıktı: ağır ışık" in read(log)


def test_the_machine_s_own_log_lives_where_the_data_does() -> None:
    """Not beside the settings: a log file has no business following the user
    to another machine through a roaming profile (section 3.3)."""
    assert log_path().suffix == ".log"
    assert log_path().parent.is_absolute()
    assert config_dir() not in log_path().parents


# --------------------------------------------------------------------------
# What a turn leaves in it
# --------------------------------------------------------------------------


def test_a_turn_writes_down_what_it_spent(log: Path) -> None:
    log_turn(Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10, 256)))

    # The whole line, not the numbers on their own: a timestamp has three
    # digits of milliseconds in it and would answer for any of them.
    assert "300 in, 10 out, 256 cached" in read(log)


def test_a_turn_writes_down_how_many_tools_ran_and_what_it_cost(log: Path) -> None:
    """The same number the turn's row in `usage_log` carries (2.4), so that
    the log and `live-assistant cost` never disagree about a turn."""
    log_turn(
        Turn(heard="saat kaç", said="Üç.", usage=Usage(300, 10), tool_calls=2, cost_usd=0.0004)
    )

    assert "300 in, 10 out, 0 cached, 2 tools, $0.0004" in read(log)


def test_a_turn_writes_down_how_long_its_first_sound_took(log: Path) -> None:
    """The number 2.8 is about, watched here across days rather than
    heard once; a turn that made no sound has none to write."""
    log_turn(Turn(heard="saat kaç", said="Üç.", usage=Usage(300, 10), first_sound_ms=1840.2))
    log_turn(Turn(heard="saat kaç", said="", usage=Usage(300, 10)))

    first, second = read(log).strip().splitlines()
    assert first.endswith("first sound 1840 ms")
    assert "first sound" not in second


def test_a_turn_with_no_known_price_says_so_rather_than_claim_it_was_free(log: Path) -> None:
    log_turn(Turn(heard="saat kaç", said="Üç.", usage=Usage(300, 10)))

    line = read(log)
    assert "price unknown" in line
    assert "$0" not in line


def test_a_turn_is_filed_under_the_name_it_has_in_the_audit_table(log: Path) -> None:
    """`tool_audit.turn_id` and this line are the two halves of one story;
    the id is what lets them be read together."""
    log_turn(Turn(heard="saat kaç", said="Üç.", usage=Usage(300, 10), turn_id="9f2a"))

    assert "turn 9f2a: 300 in, 10 out" in read(log)


def test_a_failed_turn_is_filed_under_its_name_too(log: Path) -> None:
    log_turn(Turn(heard="saat kaç", said="...", failure="took_too_long", turn_id="9f2a"))

    assert "turn 9f2a failed: took_too_long" in read(log)


def test_what_was_said_out_loud_is_not_written_to_disk(log: Path) -> None:
    log_turn(Turn(heard="kapıyı kilitledim mi", said="Bilmiyorum.", usage=Usage(300, 10)))

    written = read(log)

    assert "kapıyı kilitledim mi" not in written
    assert "Bilmiyorum." not in written


def test_a_turn_that_never_happened_is_not_a_line_in_the_log(log: Path) -> None:
    """A tapped key or a recording of silence. Logging those buries the turns
    that cost something under the ones that cost nothing."""
    log_turn(Turn())

    assert read(log).strip() == ""


def test_a_turn_that_was_missed_is_a_line_with_a_number_in_it(log: Path) -> None:
    """A user reporting that the assistant "does nothing" and a log full of
    missed turns at 0.5 have already answered each other."""
    log_turn(Turn(said="Seni anlayamadim.", missed=True, confidence=0.55))

    assert "0.55" in read(log)


def test_a_missed_turn_does_not_write_down_what_was_misheard(log: Path) -> None:
    """A transcript nothing stood behind is no more worth keeping than one
    that did."""
    log_turn(Turn(said="Seni anlayamadim.", missed=True, confidence=0.55))

    assert "Seni anlayamadim." not in read(log)


def test_a_missed_turn_with_no_number_still_gets_its_line(log: Path) -> None:
    """Not every recogniser reports a confidence, and the line is the point."""
    log_turn(Turn(said="...", missed=True, confidence=None))

    assert read(log).strip() != ""


def test_a_turn_that_failed_is_still_a_turn(log: Path) -> None:
    """It spent no tokens anybody can account for, and it is exactly the turn
    worth finding in the log afterwards."""
    log_turn(Turn(heard="saat kaç", said="Sağlayıcıya bağlanamadım.", failure="unreachable"))

    assert read(log).strip()


def test_a_turn_that_failed_says_why_and_claims_no_tokens(log: Path) -> None:
    """`turn: 0 in, 0 out` reads as a free success - the log of 2026-09-04 had
    one, and nothing said which of the three failures it was. The line names
    the kind, and only the kind: the provider's own words are where a key
    could travel, and they stay out of the file."""
    log_turn(Turn(heard="saat kaç", said="Sağlayıcıya bağlanamadım.", failure="unreachable"))

    line = read(log)
    assert "failed: unreachable" in line
    assert " in, " not in line, "a failed turn must not be written as a token count"
    assert "Sağlayıcıya" not in line


def test_a_turn_the_fast_path_answered_is_written_by_its_intent(log: Path) -> None:
    """No request was made, so no token count - and no zeros either, which
    would read as a request that happened to be free (2.5)."""
    log_turn(
        Turn(heard="saat kaç", said="Saat 14 3.", intent="get_time", tool_calls=1, turn_id="9f2a")
    )

    line = read(log)
    assert "turn 9f2a: intent get_time, 1 tools" in line
    assert " in, " not in line
    assert "$" not in line


def test_a_command_answered_by_silence_is_still_a_line(log: Path) -> None:
    """It is a turn the user had, and the count of them is how "dur" being
    heard as "durum" would be noticed."""
    log_turn(Turn(heard="dur", intent="stop"))

    assert "intent stop, 0 tools" in read(log)


# --------------------------------------------------------------------------
# What cannot reach it
# --------------------------------------------------------------------------


def test_the_value_of_a_variable_is_never_written_down(log: Path) -> None:
    """`loguru` decorates tracebacks with the values of the locals in every
    frame. One of those frames holds the API key (section 10)."""

    def build_a_provider(api_key: str) -> None:
        raise RuntimeError("the provider could not be built")

    try:
        build_a_provider(KEY)
    except RuntimeError:
        logger.exception("that did not work")

    assert KEY not in read(log)


def test_a_secret_the_program_was_told_about_is_masked_wherever_it_appears(log: Path) -> None:
    """Since 2026-09-21 the log carries the provider's own words when a
    session fails - the close code and the sentence that name the cause.
    A transport's exception could quote the address it was opening, and
    the key rides on that address. So the key is registered where it is
    loaded, and the sink masks it in every line from then on."""
    keep_secret(KEY)

    logger.warning("gemini could not be reached: wss://example/ws?key={} refused", KEY)
    logger.info("two keys in one line: {} and {}", KEY, KEY)

    written = read(log)
    assert KEY not in written
    assert "wss://example/ws?key=*** refused" in written
    assert "two keys in one line: *** and ***" in written


def test_an_empty_secret_masks_nothing(log: Path) -> None:
    """A provider without a key registers nothing; every character of
    every line would otherwise be a match."""
    keep_secret("")

    logger.info("plain words")

    assert "plain words" in read(log)


def test_a_short_secret_does_not_break_the_mask_of_a_long_one(log: Path) -> None:
    """The setup command registers whatever was typed, "key" included; the
    real key must still be masked whole, not nibbled around it."""
    keep_secret("key")
    keep_secret(KEY)

    logger.info("the key {}", KEY)

    written = read(log)
    assert KEY not in written
    assert "notareal" not in written


def test_a_secret_registered_before_the_log_is_set_up_is_still_masked(tmp_path: Path) -> None:
    keep_secret(KEY)
    path = setup_logging(path=tmp_path / "logs" / "assistant.log")

    logger.info("the key {} again", KEY)

    assert KEY not in read(path)
