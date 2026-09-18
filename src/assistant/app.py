"""What the old state machine leaves behind for the live one (plan.md 4.2-4.4).

The live state machine - `LiveAssistant`, the session that opens on speech
and closes on silence, the model's voice played as it arrives - is task
L1.4 and is not here yet. What is here is the part of the old `app.py` that
the live loop keeps word for word, and that the rest of the tree already
reads: the states the status line and the tray show, what a turn came to
for the log, the yes-and-no window's judgement of a transcript and of an
answer, the sentences said when there is no session to say them, and the
name of the failure when no voice is installed.

**A recording is judged by whether it held speech, never by how sure the
decoder was of its words.** Whisper answers a recording of silence with
confident looking words, and answers a correct single word with unsure ones:
measured on the owner's machine (2026-09-05), a subtitle credit over silence
scored 0.54 and "Merhaba." alone 0.48, so no confidence floor can separate
them. The engine's own estimate of whether anything was said does separate
them (0.86 against 0.06), and `hear` below rests on that and on nothing
else. The words themselves are never looked at: a list of known
hallucinations would be a language constant in code, which section 3.12
does not allow. In the live product this judgement serves one place: the
yes-and-no window of the permission gate (plan.md D3), which is the one
thing the local recogniser still hears.

**A "no" anywhere in the answer wins over a "yes"** (`read_answer`), because
the window only ever guards something that should not happen by mistake.
The words that count as yes and no come from the locale pack; the English
ones below are the end of the chain.

**No user-facing sentence is written here.** The pack answers first and the
English constants below are the end of the chain, exactly as in the wizard
(section 3.12).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from assistant.live.base import Usage
from assistant.store.normalize import normalize_search
from assistant.stt.base import NO_SPEECH_CEILING, Transcript

__all__ = [
    "CONFIRM_WINDOW_SECONDS",
    "MIN_UTTERANCE_SECONDS",
    "NO_WORDS",
    "TEXT",
    "YES_WORDS",
    "Heard",
    "NoVoiceError",
    "State",
    "Turn",
    "hear",
    "read_answer",
]


class NoVoiceError(RuntimeError):
    """No speech voice is installed, for the locale or otherwise.

    The user can install one; the program cannot. Named so that `live-assistant
    run` can say so in a sentence instead of a traceback.
    """


class State(StrEnum):
    """Where the assistant is. A string so the status line and the logs can
    print it without a table of names.

    The set is the old pipeline's until L1.4 brings the live one (plan.md
    section 4.2): `TRANSCRIBING` and `THINKING` go, `USER_SPEAKING` and
    `RECONNECTING` come, and `OFF` becomes a state rather than a mode.
    """

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CONFIRMING = "confirming"
    SPEAKING = "speaking"
    ANNOUNCING = "announcing"


# Section 3.1 rule 6. How long the microphone stays open for a yes or a no
# after the question has been read; what comes after it is a no.
CONFIRM_WINDOW_SECONDS = 6.0

# Shorter than this and it was a noise rather than a word. Below a syllable,
# so nothing anybody meant to say is thrown away.
MIN_UTTERANCE_SECONDS = 0.35

# The last link of the chain of section 3.12 for the two words the window
# listens for, as `TEXT` is for the sentences: the pack's `[speech]` table
# answers first, and a pack that has none gets these.
YES_WORDS = ("yes", "ok", "okay", "confirm")
NO_WORDS = ("no", "cancel", "stop")

# The last link of the chain of section 3.12: what is said when no locale pack
# offers a translation. Keys are unique across the whole project - the pack has
# one table of sentences, and `test_locales.py` checks that no two modules
# claim the same key.
TEXT: dict[str, str] = {
    # The three failures said out loud when there is no session to say them
    # (plan.md 4.4 rule 7), by the local voice.
    "unreachable": "I could not reach the provider. Will you try again?",
    "key_invalid": "Your API key is not being accepted any more. You need to renew it.",
    "took_too_long": "That took too long. Will you try again?",
    # Read after the gate's question, so the user knows what kind of answer is
    # being listened for - and again, alone, when the answer had neither.
    "confirm_hint": "Say yes or no.",
    "confirm_again": "I did not catch that. Yes, or no?",
    # Said before a session opens once a spending limit is passed - every
    # time, until the day or the month turns; and instead of a session with
    # `hard_stop` on.
    "daily_over": "You have gone over today's spending limit.",
    "monthly_over": "You have gone over this month's spending limit.",
    "spend_stopped": "The spending limit has been passed, so I am not asking the model.",
}


@dataclass(frozen=True, slots=True)
class Heard:
    """What the recogniser made of a recording, and whether it is worth a reply.

    Three outcomes rather than two, because "nothing happened" and "you said
    something and I could not read it" are different things to the person in
    the chair. Silence over a quiet room deserves silence; speech that came
    back as no words deserves to be asked about again, or the assistant
    looks broken at exactly the moment it is working as designed.
    """

    text: str = ""
    missed: bool = False
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """What one turn came to, for whoever is showing or logging it.

    Nothing else keeps any of this: the transcript is gone once the model has
    it and the answer once it has been spoken. The tokens go to the log every
    turn, which is what the cost report of section 6 is built from - so they
    have to leave the turn, and a turn that only reported success would hide
    the ones that cost tokens and still failed.

    The shape is the old pipeline's until L1.4 (plan.md 4.4 rule 6): a live
    turn is one utterance and everything the model did until it said it was
    done, and what it cost is audio minutes as well as tokens.

    A turn that *failed* carries the key of the sentence that was said instead
    of an answer - `unreachable`, `key_invalid`, `took_too_long` - and nothing
    else about the failure. The log needs the kind (a line of zero tokens
    reads as a free success); the provider's own words are where a key could
    travel and stay in the exception.

    `turn_id` is the name the turn goes by in `tool_audit` (section 3.9),
    so that a row there and a line in the log can be read together. A turn
    that made no call has none.

    `cost_usd` is what the turn cost at its model's price - `None` when the
    price is not known, or the turn never reached the model - and
    `tool_calls` how many calls the gate ran. Both go to the log.
    """

    heard: str = ""
    said: str = ""
    usage: Usage = field(default_factory=Usage)
    missed: bool = False
    confidence: float | None = None
    failure: str | None = None
    turn_id: str = ""
    cost_usd: float | None = None
    tool_calls: int = 0
    intent: str | None = None
    first_sound_ms: float | None = None


def hear(transcript: Transcript) -> Heard:
    """What to make of a transcript: nothing, unreadable speech, or words.

    The decision rests on whether there was speech, never on how sure the
    decoder was of its words. Measured on the target machine (2026-09-05): a
    correct single "Merhaba." scores 0.48 confidence and silence scores up to
    0.54, so no confidence floor can tell them apart - but the engine's own
    no-speech estimate can (0.06 against 0.86). Confidence is still carried,
    for the log and the screen, because a run of low numbers is how a bad
    microphone is diagnosed afterwards.

    An engine with no opinion is believed about its words, and its silence is
    read as speech it could not make out: it cannot tell the two apart, and
    neither can this. Treating "no opinion" as "nothing was said" would make
    the assistant mute the day it moves to a cloud recogniser.
    """
    no_speech = transcript.no_speech_probability
    if no_speech is not None and no_speech >= NO_SPEECH_CEILING:
        # The engine says nothing was said. Words it produced anyway are what
        # a recogniser makes of silence, and they are not answered.
        return Heard()

    text = transcript.text.strip()
    if text:
        return Heard(text=text, confidence=transcript.confidence)

    # There was speech, or an engine with no opinion, and no words came of it.
    return Heard(missed=True, confidence=transcript.confidence)


# A word, for the purpose of hearing "yes" in an answer: letters and digits in
# any script. Punctuation and the spaces between are where words end.
_WORD = re.compile(r"\w+")


def read_answer(text: str, *, yes: Iterable[str], no: Iterable[str]) -> bool | None:
    """Whether `text` says yes, says no, or says neither.

    Whole words, folded the way search is (`store/normalize.py`): "Evet." and
    "EVET" are the same word, and "evetlemedim" is not it. A `no` word
    anywhere wins over a `yes` word - "yes, but no" is a no - because the
    window only ever guards something that should not happen by mistake.
    `None` means neither was heard, and is the caller's cue to ask once more.
    """
    said = _spaced(text)
    if any(phrase in said for phrase in _phrases(no)):
        return False
    if any(phrase in said for phrase in _phrases(yes)):
        return True
    return None


def _phrases(words: Iterable[str]) -> list[str]:
    """Each entry as it would appear inside `_spaced` text; blanks dropped."""
    spaced = (_spaced(word) for word in words)
    return [phrase for phrase in spaced if phrase.strip()]


def _spaced(text: str) -> str:
    """The words of `text`, folded, one space between and one either side -
    so that a phrase of one or more words can be found only at word edges."""
    return f" {' '.join(_WORD.findall(normalize_search(text)))} "
