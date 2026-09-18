"""The application log (item 1.11).

One line per turn, holding what it cost: the tokens, how many tools ran, and
the price in dollars - the same number the turn's row in `usage_log` carries
(2.4), so that the log and `live-assistant cost` never disagree about a turn. A
turn the fast path answered without the model (2.5) is written by its intent
instead: no request was made, so there is no count to write. Since 2.8 the
line also holds how long the first sound took - the number that step is
about, kept here so that it can be watched across days rather than heard
once.

Three decisions are worth stating, because each of them is about what is *not*
written.

**Nothing goes to the terminal.** `loguru` installs a handler on stderr when it
is imported, and phase 1 has exactly one line of terminal to say what the
assistant is doing (`ui/status.py`). Setting up takes that handler away rather
than adding to it, which also means starting twice does not log everything
twice.

**The numbers are written down; the words are not.** A voice assistant that
keeps a plaintext transcript of everything said near a microphone is a
liability nobody asked for, and the retention rules that would cover one
(section 3.7) are phase 4. The screen shows what was said, and forgets it.

**No API key can reach the file.** Nothing here logs text that could carry one,
and `diagnose` is off so that a traceback cannot smuggle one out in the value
of a local variable - which is exactly where it would be, one frame below the
adapter (section 10). The masking filter section 5 describes belongs with the
first thing that logs a provider's own words back to us.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from assistant.app import Turn
from assistant.config import log_dir

__all__ = ["LOG_FILE", "RETAINED_FILES", "ROTATE_AT", "log_path", "log_turn", "setup_logging"]

LOG_FILE = "assistant.log"

# A turn is one short line, so this is tens of thousands of them - long enough
# to still hold last month when somebody asks where the money went.
ROTATE_AT = "2 MB"
RETAINED_FILES = 5


def log_path() -> Path:
    """`%LOCALAPPDATA%\\live-assistant\\logs\\assistant.log`.

    Beside the database rather than beside the settings: a log belongs to the
    machine that wrote it and has no business following the user to another
    one through a roaming profile (section 3.3).
    """
    return log_dir() / LOG_FILE


def setup_logging(*, path: Path | None = None, level: str = "INFO") -> Path:
    """Points the log at a file and takes every other handler away."""
    target = path if path is not None else log_path()

    # Removes `loguru`'s own stderr handler, and any handler a previous call
    # added. Adding rather than replacing is how a log ends up with every line
    # in it twice.
    logger.remove()
    logger.add(
        # `loguru` makes the directory on the way, which is the whole of what
        # is needed on a machine that has never run this before.
        target,
        level=level,
        rotation=ROTATE_AT,
        retention=RETAINED_FILES,
        # Windows opens a file in the machine's legacy code page unless it is
        # told otherwise, and half the Turkish alphabet has no place in one.
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
    )
    return target


def log_turn(finished: Turn) -> None:
    """Writes down what one turn spent.

    A turn that heard nothing never happened - a tapped key, or a recording of
    silence - and logging those buries the turns that cost something under the
    ones that cost nothing.

    A turn that *failed* is kept, by the kind of failure and nothing else. It
    reports no tokens, and written as `turn: 0 in, 0 out` it read as a free
    success - the log of 2026-09-04 had one and nothing said which of the
    three sentences the user had heard. The provider's own words stay out: they
    are where a key could travel, and the masking filter of section 5 is not
    written yet.

    So is a turn that was *missed*: speech the recogniser could not turn into
    words. It cost nothing, which is exactly why the line matters - a user
    reporting that the assistant "does nothing" and a log full of these have
    already answered each other. The decoder's doubt goes with it when there
    was one. The words are still not written down: a transcript nothing stood
    behind is no more worth keeping than one that was.

    A turn the fast path answered (2.5) is written by its intent and the
    number of tools the gate ran for it, and by no token count: none was
    spent, and zeros would read as a request that happened to be free.
    """
    if finished.missed:
        confidence = "-" if finished.confidence is None else f"{finished.confidence:.2f}"
        logger.info("missed: nothing worth answering, confidence {value}", value=confidence)
        return

    # By the name the turn goes by in `tool_audit` (section 3.9), so that a
    # row there and a line here can be read together. A turn from before
    # there were names is still a turn.
    where = f"turn {finished.turn_id}" if finished.turn_id else "turn"

    if finished.failure is not None:
        logger.warning("{where} failed: {kind}", where=where, kind=finished.failure)
        return

    if not finished.heard:
        return

    if finished.intent is not None:
        logger.info(
            "{where}: intent {intent}, {tools} tools",
            where=where,
            intent=finished.intent,
            tools=finished.tool_calls,
        )
        return

    usage = finished.usage
    logger.info(
        "{where}: {input} in, {output} out, {cached} cached, {tools} tools, {cost}{first}",
        where=where,
        input=usage.input_tokens,
        output=usage.output_tokens,
        cached=usage.cached_tokens,
        tools=finished.tool_calls,
        cost=_dollars(finished.cost_usd),
        first=_first_sound(finished.first_sound_ms),
    )


def _first_sound(milliseconds: float | None) -> str:
    """`, first sound 1840 ms`, or nothing for a turn that made no sound."""
    return "" if milliseconds is None else f", first sound {milliseconds:.0f} ms"


def _dollars(cost: float | None) -> str:
    """`$0.0004`, or the admission that the price is not known.

    Not `$0.0000`: a model with no entry in `pricing.toml` did not run for
    free, and a zero in the log would say it did (section 6).
    """
    return "price unknown" if cost is None else f"${cost:.4f}"
