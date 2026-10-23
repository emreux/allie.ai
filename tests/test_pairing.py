"""Pairing the phone with the bot (plan.md D57, spec section 6): the code
shown on the computer, sent from the phone, makes its sender the owner.

Over a fake bot whose waiting messages a test writes; nothing reaches
Telegram."""

from __future__ import annotations

import re

import pytest

from allie.remote.bot import Batch, BotError, Incoming, Press, Update
from allie.remote.pairing import PAIR_RETRY_SECONDS, new_code, pair

CODE = "482913"
OWNER = 987654321
PAIRED = "Eşleştik."


class FakeBot:
    """What is waiting for the bot, poll by poll; what was confirmed and
    what was written to the phone."""

    def __init__(self, *polls: Batch | BotError) -> None:
        self.polls = list(polls)
        self.asked: list[tuple[int | None, int]] = []
        self.sent: list[tuple[int, str]] = []

    async def updates(self, offset: int | None, *, wait: int = 50) -> Batch:
        self.asked.append((offset, wait))
        polled = self.polls.pop(0) if self.polls else Batch((), None)
        if isinstance(polled, BotError):
            raise polled
        return polled

    async def send(self, chat: int, text: str, *, buttons: object = ()) -> int:
        self.sent.append((chat, text))
        return 1


def written(update_id: int, text: str, *, sender: int = OWNER, private: bool = True) -> Incoming:
    return Incoming(
        update_id=update_id,
        chat_id=sender if private else -100123,
        private=private,
        sender_id=sender,
        text=text,
        date=0,
    )


def waiting(*updates: Update) -> Batch:
    return Batch(tuple(updates), max(update.update_id for update in updates) + 1)


class Naps:
    def __init__(self) -> None:
        self.taken: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.taken.append(seconds)


def test_a_code_is_six_digits_and_not_the_same_twice() -> None:
    codes = {new_code() for _ in range(20)}

    assert all(re.fullmatch(r"\d{6}", code) for code in codes)
    assert len(codes) > 1


async def test_whoever_sent_the_code_is_the_owner_and_is_told() -> None:
    """Telegram's own /start comes first; what was read is confirmed, so
    the channel never takes the code for a request."""
    bot = FakeBot(waiting(written(7, "/start"), written(8, CODE)))

    owner = await pair(bot, CODE, paired=PAIRED)

    assert owner == OWNER
    assert bot.asked == [(None, 0), (9, 0)]
    assert bot.sent == [(OWNER, PAIRED)]


async def test_the_code_typed_with_a_space_is_still_the_code() -> None:
    bot = FakeBot(waiting(written(3, "482 913")))

    assert await pair(bot, CODE, paired=PAIRED) == OWNER


async def test_the_code_in_a_group_pairs_nobody() -> None:
    """Only a private chat is the owner's; what was read is confirmed all
    the same, so that the channel does not answer it later."""
    bot = FakeBot(waiting(written(4, CODE, private=False)))

    assert await pair(bot, CODE, paired=PAIRED) is None
    assert bot.asked == [(None, 0), (5, 0)]
    assert bot.sent == []


async def test_another_code_or_a_button_pairs_nobody() -> None:
    press = Press(
        update_id=6, query_id="q", sender_id=OWNER, chat_id=OWNER, message_id=1, data="x:yes"
    )
    bot = FakeBot(waiting(written(5, "123456"), press))

    assert await pair(bot, CODE, paired=PAIRED) is None
    assert bot.sent == []


async def test_nothing_waiting_pairs_nobody_and_confirms_nothing() -> None:
    bot = FakeBot(Batch((), None))

    assert await pair(bot, CODE, paired=PAIRED) is None
    assert bot.asked == [(None, 0)]


async def test_a_conflict_is_asked_once_more_after_a_moment() -> None:
    """The assistant's own poll, stopped a moment ago for the settings list,
    can still be open on Telegram's side."""
    naps = Naps()
    bot = FakeBot(BotError("conflict", status=409), waiting(written(2, CODE)))

    owner = await pair(bot, CODE, paired=PAIRED, sleep=naps)

    assert owner == OWNER
    assert naps.taken == [PAIR_RETRY_SECONDS]


async def test_a_second_conflict_is_the_caller_s() -> None:
    bot = FakeBot(BotError("conflict", status=409), BotError("conflict", status=409))

    with pytest.raises(BotError) as raised:
        await pair(bot, CODE, paired=PAIRED, sleep=Naps())

    assert raised.value.kind == "conflict"


async def test_a_refused_token_is_not_asked_again() -> None:
    naps = Naps()
    bot = FakeBot(BotError("token", status=401))

    with pytest.raises(BotError):
        await pair(bot, CODE, paired=PAIRED, sleep=naps)

    assert naps.taken == []
    assert len(bot.asked) == 1
