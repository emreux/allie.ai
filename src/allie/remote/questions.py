"""The gate's question on the phone: two buttons (plan.md D59).

By voice, a tool that asks first is answered aloud and judged by the referee
(D54). On the phone the question - the gate's own, with the real values and
the person's full name in it - goes as a message with two buttons, and a tap
is not ambiguous: no referee, no word list. `ButtonQuestion` is the `confirm`
the phone's agent hands the gate (`agent/core.py::Confirm`).

**No answer is not a no** (D36). A question nobody answered within `seconds`,
or one the owner wrote past, is `None`: the message is rewritten to say that
nothing was done, and the gate tells the model the user did not answer.

**Typing the button's own word is pressing it.** People type "evet" at a
question as often as they tap; the two labels, folded (`normalize_search`),
are the only words taken. Anything else typed closes the question unanswered
and is handed back to the channel, to be read as the next message.

**A press belongs to one question.** The buttons carry a code of their own
question's; a press on another message, in another chat or with another code
is told the question is closed and moves nothing. The channel lets only the
owner's presses through (`remote/channel.py`); this is the second lock.

One question at a time: the agent runs a turn's tools one by one.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Protocol

from loguru import logger

from allie.remote.bot import BotError, Press
from allie.store.normalize import normalize_search

__all__ = ["TEXT", "ButtonQuestion", "Buttons"]

# The words under a question, and what it says once it is over - addressed to
# the owner, so through the pack.
TEXT: dict[str, str] = {
    "remote_yes": "Yes",
    "remote_no": "No",
    "remote_answered_yes": "→ Yes",
    "remote_answered_no": "→ No",
    "remote_unanswered": "→ No answer came; nothing was done.",
    "remote_question_closed": "This question is closed.",
}

# What a typed answer may carry around the word and still be the word.
_AROUND = " .!?,;:"


class Buttons(Protocol):
    """The slice of the bot a question needs (`remote/bot.py::BotApi`)."""

    async def send(
        self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()
    ) -> int: ...

    async def edit(self, chat: int, message_id: int, text: str) -> None: ...

    async def answer_button(self, query_id: str, text: str = "") -> None: ...


@dataclass(slots=True)
class _Open:
    """The question waiting for its answer: its code, its message, the answer."""

    code: str
    message_id: int
    answer: asyncio.Future[bool | None]


class ButtonQuestion:
    """Asks the owner on the phone and answers yes (`True`), no (`False`) or
    unheard (`None`)."""

    def __init__(
        self, bot: Buttons, chat: int, *, wording: Mapping[str, str], seconds: float
    ) -> None:
        self._bot = bot
        self._chat = chat
        self._wording = wording
        self._seconds = seconds
        self._numbers = count(1)
        self._open: _Open | None = None

    @property
    def open(self) -> bool:
        """Whether a question is waiting for its answer."""
        return self._open is not None

    async def __call__(self, question: str) -> bool | None:
        # A counter for order and four hex digits so that a code from an
        # earlier run of the program is not this run's by chance.
        code = f"{next(self._numbers):x}{secrets.token_hex(2)}"
        buttons = [
            (self._wording["remote_yes"], f"{code}:yes"),
            (self._wording["remote_no"], f"{code}:no"),
        ]
        try:
            message_id = await self._bot.send(self._chat, question, buttons=buttons)
        except BotError as failure:
            logger.warning("remote: a question could not be sent ({kind})", kind=failure.kind)
            return None
        answer: asyncio.Future[bool | None] = asyncio.get_running_loop().create_future()
        self._open = _Open(code, message_id, answer)
        heard: bool | None
        try:
            async with asyncio.timeout(self._seconds):
                heard = await answer
        except TimeoutError:
            heard = None
        finally:
            self._open = None
        await self._settle(question, message_id, heard)
        return heard

    async def press(self, pressed: Press) -> None:
        """A tap on a button: the open question's answer, or told it is closed."""
        waiting = self._open
        code, _, choice = pressed.data.partition(":")
        if (
            waiting is None
            or waiting.answer.done()
            or code != waiting.code
            or choice not in ("yes", "no")
            or pressed.chat_id != self._chat
            or pressed.message_id != waiting.message_id
        ):
            await self._acknowledge(pressed.query_id, self._wording["remote_question_closed"])
            return
        waiting.answer.set_result(choice == "yes")
        await self._acknowledge(pressed.query_id)

    def offer_text(self, text: str) -> bool:
        """A message written while a question is open: a button's own word is
        the answer (`True`, taken); anything else closes the question
        unanswered and is left to the caller (`False`). With nothing open,
        nothing is taken."""
        waiting = self._open
        if waiting is None or waiting.answer.done():
            return False
        said = _folded(text)
        if said == _folded(self._wording["remote_yes"]):
            waiting.answer.set_result(True)
            return True
        if said == _folded(self._wording["remote_no"]):
            waiting.answer.set_result(False)
            return True
        waiting.answer.set_result(None)
        return False

    async def _acknowledge(self, query_id: str, text: str = "") -> None:
        """Tells Telegram the tap arrived, or the button keeps spinning."""
        try:
            await self._bot.answer_button(query_id, text)
        except BotError as failure:
            logger.warning("remote: a tap could not be acknowledged ({kind})", kind=failure.kind)

    async def _settle(self, question: str, message_id: int, heard: bool | None) -> None:
        """The question rewritten with its outcome under it; its buttons go."""
        if heard is True:
            line = self._wording["remote_answered_yes"]
        elif heard is False:
            line = self._wording["remote_answered_no"]
        else:
            line = self._wording["remote_unanswered"]
        try:
            await self._bot.edit(self._chat, message_id, f"{question}\n\n{line}")
        except BotError as failure:
            logger.warning("remote: a question could not be closed ({kind})", kind=failure.kind)


def _folded(text: str) -> str:
    """A typed word or a label, as they are compared."""
    return normalize_search(text).strip(_AROUND)
