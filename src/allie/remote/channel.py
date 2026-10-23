"""The phone remote's channel (plan.md D57, D58): the owner's messages in,
one at a time, and the answers out.

Two tasks share the work. The **poll** asks Telegram for what was written to
the bot, over and over, and never waits for a turn: while the agent is busy
with one message - or waiting at a question - the poll goes on, because the
answer to that question arrives through it. The **worker** takes the owner's
messages from a queue and answers them one by one, in the order they came.

**The owner only.** An update counts when it comes from `owner_id` in the
owner's own chat; anything else is dropped without an answer, and the log
gets the sender's id once per run - never the text.

**What happens around a question.** A press goes to the question
(`remote/questions.py`). A message written while a question is open is
offered to it first: a button's own word answers it; anything else closes
it unanswered and waits its turn as the next message.

**What is not acted on.** A message older than `stale_minutes` by Telegram's
own date arrived while nobody could answer it - the computer was off, asleep
or offline - and a song started hours after it was asked for is a surprise,
not a service: such messages are counted, said in one sentence, and left. A
message with no words in it (a photo, a voice note) is told that only
writing is read.

**One conversation, kept a while.** The agent remembers the conversation so
that "onu da gönder" follows; after `fresh_minutes` without a message the
next one starts a new conversation.

**The answer goes back as one message.** Files `send_document` queued during
the turn (`Outgoing`) are the answer, and the model's words - if it wrote any
- are the caption of the last one; words too long for a caption come after
it. Without files the words go alone, and an answer with no words says that
it is done.

**A failure is one sentence,** chosen by what failed - Google's limit, the
network, the key - and logged by its kind or class, never by its words. A
failed turn sends none of the files it queued.

**Telegram's own trouble is waited out:** no network, `1, 2, 4 ... 60` seconds
(one WARNING per outage, an INFO when it is over); something else reading
the bot, `CONFLICT_SECONDS`; Telegram's "slow down", as long as it says. A
refused token ends the channel: `on_refused` is told once, and the voice
side runs on.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from loguru import logger

from allie.agent.core import Confirm
from allie.agent.text import Answer
from allie.live.base import AuthenticationError, ProviderError
from allie.remote.bot import CAPTION_LIMIT, POLL_SECONDS, Batch, BotError, Incoming, Press

__all__ = [
    "BACKOFF_SECONDS",
    "CONFLICT_SECONDS",
    "TEXT",
    "TYPING_EVERY_SECONDS",
    "Conversation",
    "Outgoing",
    "Question",
    "RemoteBot",
    "RemoteChannel",
]

# What the owner reads when something other than an answer comes back -
# addressed to the owner, so through the pack.
TEXT: dict[str, str] = {
    "remote_stale": (
        "{count} message(s) came while I could not answer; I did none of them. "
        "Write again whatever you still want."
    ),
    "remote_done": "Done.",
    "remote_text_only": "For now I read written messages only.",
    "remote_rate_limit": (
        "Google's limit is used up for now (15 requests a minute, 500 a day); "
        "try again a little later."
    ),
    "remote_unreachable": "I could not reach Google; try again in a moment.",
    "remote_key_refused": "Google refused the key; renew it in Settings on the computer.",
    "remote_failed": "Something went wrong; try again.",
    "remote_file_failed": "I could not send {file}.",
}

# "typing..." lasts about five seconds in Telegram; renewed a little sooner.
TYPING_EVERY_SECONDS = 4.0

# How long a lost network is waited out, attempt by attempt; the last stays.
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)

# Something else is reading this bot's messages (409): another copy of the
# program, or the settings row a moment ago (spec section 6).
CONFLICT_SECONDS = 30.0

# A model's failure, by kind, to the sentence the owner reads.
_SAID_FOR = {
    "rate_limit": "remote_rate_limit",
    "unreachable": "remote_unreachable",
    "timeout": "remote_unreachable",
}


class RemoteBot(Protocol):
    """The slice of `remote/bot.py::BotApi` the channel calls itself."""

    async def updates(self, offset: int | None, *, wait: int = POLL_SECONDS) -> Batch: ...

    async def send(
        self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()
    ) -> int: ...

    async def send_document(self, chat: int, path: Path, *, caption: str = "") -> int: ...

    async def typing(self, chat: int) -> None: ...

    async def aclose(self) -> None: ...


class Question(Protocol):
    """The phone's `confirm` (`remote/questions.py::ButtonQuestion`)."""

    @property
    def open(self) -> bool: ...

    async def __call__(self, question: str) -> bool | None: ...

    async def press(self, pressed: Press) -> None: ...

    def offer_text(self, text: str) -> bool: ...


class Conversation(Protocol):
    """One conversation with the text model (`agent/text.py::Agent`)."""

    async def reply(self, said: str, *, turn_id: str, confirm: Confirm) -> Answer: ...


class Outgoing:
    """The files a turn sends back: queued by `send_document` while the model
    works, handed over once when the turn's words are known."""

    def __init__(self) -> None:
        self._files: list[Path] = []

    def add(self, path: Path) -> None:
        self._files.append(path)

    def drain(self) -> list[Path]:
        files, self._files = self._files, []
        return files


class RemoteChannel:
    """The poll and the worker of the module's docstring, run by `run`."""

    def __init__(
        self,
        bot: RemoteBot,
        question: Question,
        *,
        owner_id: int,
        agent_for: Callable[[], Conversation],
        outgoing: Outgoing,
        wording: Mapping[str, str],
        stale_minutes: float,
        fresh_minutes: float,
        on_refused: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        typing_every: float = TYPING_EVERY_SECONDS,
        wait: int = POLL_SECONDS,
    ) -> None:
        self._bot = bot
        self._question = question
        self._owner = owner_id
        self._agent_for = agent_for
        self._outgoing = outgoing
        self._wording = wording
        self._stale_seconds = stale_minutes * 60
        self._fresh_seconds = fresh_minutes * 60
        self._on_refused = on_refused
        self._clock = clock
        self._sleep = sleep
        self._typing_every = typing_every
        self._wait = wait
        self._queue: asyncio.Queue[Incoming] = asyncio.Queue()
        self._strangers: set[int] = set()
        self._agent: Conversation | None = None
        self._last_message = 0.0

    async def run(self) -> None:
        """Polls and answers until the token is refused or the task is
        cancelled; the bot is closed either way."""
        worker = asyncio.create_task(self._work())
        try:
            await self._poll()
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            await self._bot.aclose()

    async def _poll(self) -> None:
        offset: int | None = None
        trouble: str | None = None
        attempt = 0
        while True:
            try:
                batch = await self._bot.updates(offset, wait=self._wait)
            except BotError as failure:
                if failure.kind == "token":
                    logger.error("remote: Telegram refused the bot's token; the phone remote stops")
                    self._on_refused()
                    return
                if trouble is None:
                    logger.warning(
                        "remote: Telegram cannot be read ({kind}); trying again",
                        kind=failure.kind,
                    )
                trouble = failure.kind
                await self._sleep(self._delay(failure, attempt))
                attempt += 1
                continue
            if trouble is not None:
                logger.info("remote: Telegram can be read again")
                trouble = None
                attempt = 0
            if batch.next_offset is not None:
                offset = batch.next_offset
            await self._take(batch.updates)

    def _delay(self, failure: BotError, attempt: int) -> float:
        if failure.kind == "conflict":
            return CONFLICT_SECONDS
        if failure.kind == "rate_limit" and failure.retry_after > 0:
            return failure.retry_after
        return BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]

    async def _take(self, updates: Sequence[Incoming | Press]) -> None:
        """One poll's updates, sorted: presses to the question, the owner's
        messages to the question or the queue, the rest dropped."""
        stale = 0
        stale_chat = self._owner
        for update in updates:
            if not self._is_owners(update):
                continue
            if isinstance(update, Press):
                await self._question.press(update)
                continue
            if self._clock() - update.date > self._stale_seconds:
                stale += 1
                stale_chat = update.chat_id
                continue
            if not update.text.strip():
                await self._say(update.chat_id, self._wording["remote_text_only"])
                continue
            if self._question.offer_text(update.text):
                continue
            await self._queue.put(update)
        if stale:
            logger.info("remote: {count} stale message(s) left undone", count=stale)
            await self._say(stale_chat, self._wording["remote_stale"].format(count=stale))

    def _is_owners(self, update: Incoming | Press) -> bool:
        private = update.private if isinstance(update, Incoming) else True
        if private and update.sender_id == self._owner and update.chat_id == self._owner:
            return True
        if update.sender_id not in self._strangers:
            self._strangers.add(update.sender_id)
            logger.info("remote: dropped what {sender} sent", sender=update.sender_id)
        return False

    async def _work(self) -> None:
        while True:
            message = await self._queue.get()
            await self._answer(message)

    async def _answer(self, message: Incoming) -> None:
        now = self._clock()
        if self._agent is None or now - self._last_message > self._fresh_seconds:
            self._agent = self._agent_for()
        self._last_message = now
        turn = f"telegram-{message.update_id}"
        typing = asyncio.create_task(self._typing(message.chat_id))
        words: str
        files: list[Path] = []
        try:
            answer = await self._agent.reply(message.text, turn_id=turn, confirm=self._question)
        except AuthenticationError:
            logger.warning("remote: {turn} failed: the model refused the key", turn=turn)
            words = self._wording["remote_key_refused"]
            self._outgoing.drain()
        except ProviderError as failure:
            logger.warning("remote: {turn} failed ({kind})", turn=turn, kind=failure.kind)
            words = self._wording[_SAID_FOR.get(failure.kind, "remote_failed")]
            self._outgoing.drain()
        except Exception as failure:
            # Whatever broke, the phone hears one sentence and the next
            # message is answered: the worker must not die with a turn.
            logger.error("remote: {turn} failed ({name})", turn=turn, name=type(failure).__name__)
            words = self._wording["remote_failed"]
            self._outgoing.drain()
        else:
            words = answer.text.strip()
            files = self._outgoing.drain()
            logger.info(
                "remote: {turn} answered: {tools} tool call(s), {files} file(s), {chars} chars",
                turn=turn,
                tools=answer.tool_calls,
                files=len(files),
                chars=len(words),
            )
        finally:
            typing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing
        await self._deliver(message.chat_id, words, files)

    async def _deliver(self, chat: int, words: str, files: list[Path]) -> None:
        """The turn's answer as one message: the files with the words under the
        last, or the words alone."""
        if not files:
            await self._say(chat, words or self._wording["remote_done"])
            return
        caption = words if len(words) <= CAPTION_LIMIT else ""
        captioned = False
        for number, path in enumerate(files, start=1):
            under = caption if number == len(files) else ""
            try:
                await self._bot.send_document(chat, path, caption=under)
            except BotError as failure:
                logger.warning("remote: a file could not be sent ({kind})", kind=failure.kind)
                await self._say(chat, self._wording["remote_file_failed"].format(file=path.name))
                continue
            captioned = captioned or bool(under)
        if words and not captioned:
            await self._say(chat, words)

    async def _typing(self, chat: int) -> None:
        """ "typing..." while the agent works, and not while a question waits."""
        while True:
            if not self._question.open:
                with contextlib.suppress(BotError):
                    await self._bot.typing(chat)
            await asyncio.sleep(self._typing_every)

    async def _say(self, chat: int, text: str) -> None:
        try:
            await self._bot.send(chat, text)
        except BotError as failure:
            logger.warning("remote: a message could not be sent ({kind})", kind=failure.kind)
