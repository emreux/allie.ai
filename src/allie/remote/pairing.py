"""Pairing the phone with the bot (plan.md D57, spec of 2026-10-23 section 6).

A bot can never write first, so the phone writes once: a six-digit code
shown on the computer, sent to the bot from the phone. Whoever sent it in a
private chat is the owner - the one `owner_id` the channel answers. The
code is what proves whose phone it is, and it works without the Telegram
login that `send_message` uses.

`pair` reads what is waiting for the bot at once (no long poll: the owner
pressed Enter after sending), and confirms everything it read - the code,
Telegram's own `/start`, anything else - so that the channel, when it
starts, never takes any of it for a request. Found, the phone is told.

A 409 is the assistant's own poll, stopped a moment ago for the settings
list and still open on Telegram's side: asked once more after
`PAIR_RETRY_SECONDS`. Anything else Telegram refuses is the caller's.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from allie.remote.bot import Batch, BotError, Incoming

__all__ = ["CODE_DIGITS", "PAIR_RETRY_SECONDS", "Pairing", "new_code", "pair"]

CODE_DIGITS = 6
PAIR_RETRY_SECONDS = 2.0


class Pairing(Protocol):
    """The slice of `remote/bot.py::BotApi` pairing calls."""

    async def updates(self, offset: int | None, *, wait: int = ...) -> Batch: ...

    async def send(
        self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()
    ) -> int: ...


def new_code() -> str:
    """Six digits, from the operating system's own randomness."""
    return f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"


async def pair(
    bot: Pairing,
    code: str,
    *,
    paired: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int | None:
    """The id of whoever sent `code` to the bot in a private chat - told
    `paired` - or `None`. A refused token or no network raises `BotError`."""
    batch = await _waiting(bot, sleep)
    if batch.next_offset is not None:
        await bot.updates(batch.next_offset, wait=0)
    owner = next(
        (
            update.sender_id
            for update in batch.updates
            if isinstance(update, Incoming)
            and update.private
            and "".join(update.text.split()) == code
        ),
        None,
    )
    if owner is not None:
        await bot.send(owner, paired)
    return owner


async def _waiting(bot: Pairing, sleep: Callable[[float], Awaitable[None]]) -> Batch:
    try:
        return await bot.updates(None, wait=0)
    except BotError as failure:
        if failure.kind != "conflict":
            raise
    await sleep(PAIR_RETRY_SECONDS)
    return await bot.updates(None, wait=0)
