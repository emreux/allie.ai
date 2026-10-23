"""The phone remote's channel (plan.md D57, D58): the owner's messages in,
one at a time, and the answers out.

The channel is driven with a fake bot - the polls fed by the test, every send
recorded - and a fake agent per conversation, so that what is tested is the
channel's own doing: who is listened to and in what order, what a question
does to the messages around it, how an answer goes back - a file as the
answer, the words under it - and what a failure turns into.

The real `ButtonQuestion` is used, with the pack's Turkish buttons: it has its
own suite (`test_remote_questions.py`), and here it is the thing the channel
hands its presses and its typed words to.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from pathlib import Path

import pytest
from loguru import logger

from allie.agent.text import Answer
from allie.live.base import AuthenticationError, ProviderError, Usage
from allie.remote.bot import Batch, BotError, Incoming, Press
from allie.remote.channel import TEXT, Outgoing, RemoteChannel
from allie.remote.questions import TEXT as QUESTION_TEXT
from allie.remote.questions import ButtonQuestion

OWNER = 42
STRANGER = 999
NOW = 1_761_200_000.0

QUESTION_WORDING = {**QUESTION_TEXT, "remote_yes": "Evet", "remote_no": "Hayır"}

Ask = Callable[[str], Awaitable[bool | None]]
Behaviour = Callable[[str, Ask], Awaitable[str]]


class FakeBot:
    """The bot, with the polls fed by the test: a poll with nothing in it
    waits, as Telegram's long poll does."""

    def __init__(self) -> None:
        self.polls: asyncio.Queue[Batch | BotError] = asyncio.Queue()
        self.offsets: list[int | None] = []
        self.sent: list[tuple[int, str, list[tuple[str, str]]]] = []
        self.documents: list[tuple[int, str, str]] = []
        self.refused_files: set[str] = set()
        self.typed: list[int] = []
        self.answered: list[tuple[str, str]] = []
        self.edits: list[tuple[int, int, str]] = []
        self.closed = False

    def deliver(self, *updates: Incoming | Press) -> None:
        last = max(update.update_id for update in updates)
        self.polls.put_nowait(Batch(tuple(updates), last + 1))

    def fail(self, error: BotError) -> None:
        self.polls.put_nowait(error)

    @property
    def texts(self) -> list[str]:
        return [text for _, text, _ in self.sent]

    async def updates(self, offset: int | None, *, wait: int = 50) -> Batch:
        self.offsets.append(offset)
        polled = await self.polls.get()
        if isinstance(polled, BotError):
            raise polled
        return polled

    async def send(self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()) -> int:
        self.sent.append((chat, text, list(buttons)))
        return len(self.sent)

    async def send_document(self, chat: int, path: Path, *, caption: str = "") -> int:
        if path.name in self.refused_files:
            raise BotError("refused", status=400)
        self.documents.append((chat, path.name, caption))
        return 1000 + len(self.documents)

    async def typing(self, chat: int) -> None:
        self.typed.append(chat)

    async def answer_button(self, query_id: str, text: str = "") -> None:
        self.answered.append((query_id, text))

    async def edit(self, chat: int, message_id: int, text: str) -> None:
        self.edits.append((chat, message_id, text))

    async def aclose(self) -> None:
        self.closed = True


class Conversation:
    """One fake conversation: what it was told, answered by `behave`."""

    def __init__(self, behave: Behaviour) -> None:
        self.behave = behave
        self.heard: list[tuple[str, str]] = []

    async def reply(self, said: str, *, turn_id: str, confirm: Ask) -> Answer:
        self.heard.append((said, turn_id))
        return Answer(text=await self.behave(said, confirm), usage=Usage(), finish_reason="STOP")


class Conversations:
    """`agent_for`: a new fake conversation every time it is asked for one."""

    def __init__(self, behave: Behaviour) -> None:
        self.behave = behave
        self.made: list[Conversation] = []

    def __call__(self) -> Conversation:
        conversation = Conversation(self.behave)
        self.made.append(conversation)
        return conversation

    @property
    def heard(self) -> list[str]:
        return [said for conversation in self.made for said, _ in conversation.heard]


def saying(words: str) -> Behaviour:
    async def behave(said: str, confirm: Ask) -> str:
        return words

    return behave


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


class Sleeps:
    """The back-off's sleep: recorded, and over at once."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0)


def message(
    update_id: int,
    text: str,
    *,
    sender: int = OWNER,
    chat: int | None = None,
    private: bool = True,
    age: float = 0.0,
) -> Incoming:
    return Incoming(
        update_id=update_id,
        chat_id=sender if chat is None else chat,
        private=private,
        sender_id=sender,
        text=text,
        date=int(NOW - age),
    )


def channel_with(
    bot: FakeBot,
    conversations: Conversations,
    *,
    outgoing: Outgoing | None = None,
    clock: Clock | None = None,
    sleeps: Sleeps | None = None,
    on_refused: Callable[[], None] | None = None,
    typing_every: float = 60.0,
) -> RemoteChannel:
    return RemoteChannel(
        bot,
        ButtonQuestion(bot, OWNER, wording=QUESTION_WORDING, seconds=5),
        owner_id=OWNER,
        agent_for=conversations,
        outgoing=outgoing if outgoing is not None else Outgoing(),
        wording=TEXT,
        stale_minutes=10,
        fresh_minutes=30,
        on_refused=on_refused if on_refused is not None else lambda: None,
        clock=clock if clock is not None else Clock(),
        sleep=sleeps if sleeps is not None else Sleeps(),
        typing_every=typing_every,
    )


@contextlib.asynccontextmanager
async def running(channel: RemoteChannel) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(channel.run())
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def until(condition: Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the channel never got there")


def asked(bot: FakeBot) -> tuple[int, str, list[tuple[str, str]]] | None:
    """The question on the phone - the one message sent with buttons."""
    return next((sent for sent in bot.sent if sent[2]), None)


def pressing(bot: FakeBot, answer: str, *, update_id: int) -> Press:
    question = asked(bot)
    assert question is not None
    data = next(data for _, data in question[2] if data.endswith(f":{answer}"))
    return Press(
        update_id=update_id,
        query_id=f"q-{update_id}",
        sender_id=OWNER,
        chat_id=OWNER,
        message_id=bot.sent.index(question) + 1,
        data=data,
    )


@pytest.fixture
def logged() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level} {message}", level="INFO")
    yield lines
    logger.remove(sink)


# --------------------------------------------------------------------------
# Who is listened to
# --------------------------------------------------------------------------


async def test_a_strangers_messages_and_a_groups_are_dropped_unanswered(
    logged: list[str],
) -> None:
    bot = FakeBot()
    conversations = Conversations(saying("Tamam."))

    async with running(channel_with(bot, conversations)):
        bot.deliver(
            message(1, "şarj kaç", sender=STRANGER),
            message(2, "bilgisayarı kapat", sender=STRANGER),
            message(3, "selam", chat=-100, private=False),
        )
        await until(lambda: bot.offsets[-1:] == [4])

    assert bot.sent == []
    assert conversations.made == []
    assert len([line for line in logged if str(STRANGER) in line]) == 1
    assert not [line for line in logged if "şarj" in line or "kapat" in line or "selam" in line]


async def test_a_strangers_press_is_not_even_answered() -> None:
    bot = FakeBot()

    async with running(channel_with(bot, Conversations(saying("Tamam.")))):
        bot.deliver(
            Press(
                update_id=1,
                query_id="q-1",
                sender_id=STRANGER,
                chat_id=STRANGER,
                message_id=1,
                data="1abcd:yes",
            )
        )
        await until(lambda: bot.offsets[-1:] == [2])

    assert bot.answered == []


async def test_each_poll_confirms_the_one_before() -> None:
    bot = FakeBot()

    async with running(channel_with(bot, Conversations(saying("Tamam.")))):
        bot.deliver(message(1, "bir"), message(3, "iki"))
        await until(lambda: len(bot.offsets) == 2)

    assert bot.offsets == [None, 4]


# --------------------------------------------------------------------------
# One at a time, in order, and what a question does
# --------------------------------------------------------------------------


async def test_the_owners_messages_are_answered_one_at_a_time_in_order(
    logged: list[str],
) -> None:
    release = asyncio.Event()
    started: list[str] = []

    async def behave(said: str, confirm: Ask) -> str:
        started.append(said)
        if said == "birinci":
            await release.wait()
        return f"{said} tamam"

    bot = FakeBot()
    conversations = Conversations(behave)
    async with running(channel_with(bot, conversations)):
        bot.deliver(message(1, "birinci"), message(2, "ikinci"))
        await until(lambda: started == ["birinci"])
        await asyncio.sleep(0.01)
        assert started == ["birinci"]
        release.set()
        await until(lambda: len(bot.texts) == 2)

    assert bot.texts == ["birinci tamam", "ikinci tamam"]
    turns = [turn for conversation in conversations.made for _, turn in conversation.heard]
    assert turns == ["telegram-1", "telegram-2"]
    assert not [line for line in logged if "birinci" in line or "ikinci" in line]


async def test_typing_shows_while_the_agent_works_and_not_while_a_question_waits() -> None:
    async def behave(said: str, confirm: Ask) -> str:
        await asyncio.sleep(0.05)
        return "Kapattım." if await confirm("Spotify'ı kapatayım mı?") else "Kapatmadım."

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave), typing_every=0.01)):
        bot.deliver(message(1, "spotify'ı kapat"))
        await until(lambda: asked(bot) is not None)
        typed_before_the_question = len(bot.typed)
        await asyncio.sleep(0.05)
        assert len(bot.typed) == typed_before_the_question
        bot.deliver(pressing(bot, "yes", update_id=2))
        await until(lambda: "Kapattım." in bot.texts)

    assert typed_before_the_question >= 2
    assert set(bot.typed) == {OWNER}


async def test_a_press_answers_the_open_question() -> None:
    async def behave(said: str, confirm: Ask) -> str:
        return "Kapattım." if await confirm("Spotify'ı kapatayım mı?") else "Kapatmadım."

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave))):
        bot.deliver(message(1, "spotify'ı kapat"))
        await until(lambda: asked(bot) is not None)
        bot.deliver(pressing(bot, "no", update_id=2))
        await until(lambda: "Kapatmadım." in bot.texts)

    assert bot.answered == [("q-2", "")]


async def test_typing_the_yes_word_answers_the_question_and_is_not_a_new_message() -> None:
    answers: list[bool | None] = []

    async def behave(said: str, confirm: Ask) -> str:
        answers.append(await confirm("Ben'e 'selam' göndereyim mi?"))
        return "Gönderdim."

    bot = FakeBot()
    conversations = Conversations(behave)
    async with running(channel_with(bot, conversations)):
        bot.deliver(message(1, "Ben'e selam yaz"))
        await until(lambda: asked(bot) is not None)
        bot.deliver(message(2, "evet"))
        await until(lambda: "Gönderdim." in bot.texts)

    assert answers == [True]
    assert conversations.heard == ["Ben'e selam yaz"]


async def test_anything_else_typed_closes_the_question_and_is_answered_next() -> None:
    answers: list[bool | None] = []

    async def behave(said: str, confirm: Ask) -> str:
        if said == "Ben'e selam yaz":
            answers.append(await confirm("Ben'e 'selam' göndereyim mi?"))
            return "Cevap gelmedi, göndermedim."
        return "Şarj yüzde 80."

    bot = FakeBot()
    conversations = Conversations(behave)
    async with running(channel_with(bot, conversations)):
        bot.deliver(message(1, "Ben'e selam yaz"))
        await until(lambda: asked(bot) is not None)
        bot.deliver(message(2, "şarj kaç"))
        await until(lambda: "Şarj yüzde 80." in bot.texts)

    assert answers == [None]
    assert conversations.heard == ["Ben'e selam yaz", "şarj kaç"]


async def test_messages_that_came_while_nobody_could_answer_are_counted_not_done() -> None:
    bot = FakeBot()
    conversations = Conversations(saying("Tamam."))

    async with running(channel_with(bot, conversations)):
        bot.deliver(
            message(1, "spotify'ı aç", age=5 * 3600),
            message(2, "şarkı aç", age=20 * 60),
            message(3, "şarj kaç", age=11 * 60),
        )
        await until(lambda: bot.texts != [])

    assert bot.texts == [TEXT["remote_stale"].format(count=3)]
    assert conversations.made == []


async def test_a_message_without_words_is_told_only_writing_is_read() -> None:
    bot = FakeBot()
    conversations = Conversations(saying("Tamam."))

    async with running(channel_with(bot, conversations)):
        bot.deliver(message(1, ""))
        await until(lambda: bot.texts != [])

    assert bot.texts == [TEXT["remote_text_only"]]
    assert conversations.made == []


async def test_the_conversation_is_kept_and_starts_afresh_after_a_long_silence() -> None:
    clock = Clock()
    bot = FakeBot()
    conversations = Conversations(saying("Tamam."))

    async with running(channel_with(bot, conversations, clock=clock)):
        bot.deliver(message(1, "bir"))
        await until(lambda: len(bot.texts) == 1)
        bot.deliver(message(2, "iki"))
        await until(lambda: len(bot.texts) == 2)
        clock.now += 31 * 60
        bot.deliver(message(3, "üç", age=-31 * 60))
        await until(lambda: len(bot.texts) == 3)

    assert [len(conversation.heard) for conversation in conversations.made] == [2, 1]


# --------------------------------------------------------------------------
# How the answer goes back
# --------------------------------------------------------------------------


async def test_an_answer_with_no_words_and_nothing_to_send_says_done() -> None:
    bot = FakeBot()

    async with running(channel_with(bot, Conversations(saying("  ")))):
        bot.deliver(message(1, "müziği durdur"))
        await until(lambda: bot.texts != [])

    assert bot.texts == [TEXT["remote_done"]]


async def test_a_queued_file_is_the_answer_with_the_words_under_it(tmp_path: Path) -> None:
    document = tmp_path / "Sözleşme.pdf"
    document.write_bytes(b"%PDF-1.7")
    outgoing = Outgoing()

    async def behave(said: str, confirm: Ask) -> str:
        outgoing.add(document)
        return "Ahmet'in sözleşmesi."

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave), outgoing=outgoing)):
        bot.deliver(message(1, "Ahmet'in sözleşmesini gönder"))
        await until(lambda: bot.documents != [])

    assert bot.documents == [(OWNER, "Sözleşme.pdf", "Ahmet'in sözleşmesi.")]
    assert bot.texts == []


async def test_two_files_go_with_the_words_under_the_last(tmp_path: Path) -> None:
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    for document in (first, second):
        document.write_bytes(b"%PDF-1.7")
    outgoing = Outgoing()

    async def behave(said: str, confirm: Ask) -> str:
        outgoing.add(first)
        outgoing.add(second)
        return "İkisi de burada."

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave), outgoing=outgoing)):
        bot.deliver(message(1, "iki dosyayı da gönder"))
        await until(lambda: len(bot.documents) == 2)

    assert bot.documents == [(OWNER, "a.pdf", ""), (OWNER, "b.pdf", "İkisi de burada.")]
    assert bot.texts == []


async def test_words_too_long_for_a_caption_come_after_the_file(tmp_path: Path) -> None:
    document = tmp_path / "a.pdf"
    document.write_bytes(b"%PDF-1.7")
    outgoing = Outgoing()
    words = "x" * 1100

    async def behave(said: str, confirm: Ask) -> str:
        outgoing.add(document)
        return words

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave), outgoing=outgoing)):
        bot.deliver(message(1, "gönder"))
        await until(lambda: bot.texts != [])

    assert bot.documents == [(OWNER, "a.pdf", "")]
    assert bot.texts == [words]


async def test_a_file_that_could_not_be_sent_is_said_and_the_words_still_come(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    for document in (first, second):
        document.write_bytes(b"%PDF-1.7")
    outgoing = Outgoing()

    async def behave(said: str, confirm: Ask) -> str:
        outgoing.add(first)
        outgoing.add(second)
        return "İkisi de burada."

    bot = FakeBot()
    bot.refused_files = {"b.pdf"}
    async with running(channel_with(bot, Conversations(behave), outgoing=outgoing)):
        bot.deliver(message(1, "iki dosyayı da gönder"))
        await until(lambda: len(bot.texts) == 2)

    assert bot.documents == [(OWNER, "a.pdf", "")]
    assert bot.texts == [TEXT["remote_file_failed"].format(file="b.pdf"), "İkisi de burada."]


# --------------------------------------------------------------------------
# What a failure turns into
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "key"),
    [
        (ProviderError("quota", kind="rate_limit"), "remote_rate_limit"),
        (ProviderError("no network", kind="unreachable"), "remote_unreachable"),
        (ProviderError("too slow", kind="timeout"), "remote_unreachable"),
        (AuthenticationError("bad key"), "remote_key_refused"),
        (ProviderError("odd", kind="refused"), "remote_failed"),
        (RuntimeError("boom"), "remote_failed"),
    ],
)
async def test_a_failed_turn_is_one_sentence_and_the_files_it_queued_stay_home(
    failure: Exception, key: str, tmp_path: Path
) -> None:
    document = tmp_path / "a.pdf"
    document.write_bytes(b"%PDF-1.7")
    outgoing = Outgoing()

    async def behave(said: str, confirm: Ask) -> str:
        outgoing.add(document)
        raise failure

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave), outgoing=outgoing)):
        bot.deliver(message(1, "şarj kaç"))
        await until(lambda: bot.texts != [])

    assert bot.texts == [TEXT[key]]
    assert bot.documents == []
    assert outgoing.drain() == []


async def test_an_unexpected_failure_is_logged_by_its_class_alone(logged: list[str]) -> None:
    async def behave(said: str, confirm: Ask) -> str:
        raise RuntimeError("the secret words of the turn")

    bot = FakeBot()
    async with running(channel_with(bot, Conversations(behave))):
        bot.deliver(message(1, "şarj kaç"))
        await until(lambda: bot.texts != [])

    errors = [line for line in logged if line.startswith("ERROR")]
    assert len(errors) == 1
    assert "RuntimeError" in errors[0]
    assert not [line for line in logged if "secret words" in line or "şarj" in line]


async def test_a_lost_network_is_waited_out_longer_each_time_and_said_once(
    logged: list[str],
) -> None:
    sleeps = Sleeps()
    bot = FakeBot()
    for _ in range(3):
        bot.fail(BotError("unreachable"))

    async with running(channel_with(bot, Conversations(saying("Tamam.")), sleeps=sleeps)):
        bot.deliver(message(1, "selam"))
        await until(lambda: bot.texts == ["Tamam."])

    assert sleeps.delays == [1, 2, 4]
    assert len([line for line in logged if line.startswith("WARNING")]) == 1
    assert [line for line in logged if line.startswith("INFO") and "again" in line]


async def test_another_reader_and_a_slow_down_are_waited_out_as_told() -> None:
    sleeps = Sleeps()
    bot = FakeBot()
    bot.fail(BotError("conflict", status=409))
    bot.fail(BotError("rate_limit", status=429, retry_after=7))

    async with running(channel_with(bot, Conversations(saying("Tamam.")), sleeps=sleeps)):
        bot.deliver(message(1, "selam"))
        await until(lambda: bot.texts == ["Tamam."])

    assert sleeps.delays == [30, 7]


async def test_a_refused_token_ends_the_channel_once_and_closes_the_bot(
    logged: list[str],
) -> None:
    told: list[str] = []
    bot = FakeBot()
    bot.fail(BotError("token", status=401))

    await asyncio.wait_for(
        channel_with(
            bot, Conversations(saying("Tamam.")), on_refused=lambda: told.append("refused")
        ).run(),
        timeout=1,
    )

    assert told == ["refused"]
    assert bot.closed
    assert [line for line in logged if line.startswith("ERROR")]


async def test_stopping_the_channel_closes_the_bot() -> None:
    bot = FakeBot()

    async with running(channel_with(bot, Conversations(saying("Tamam.")))):
        await until(lambda: bot.offsets != [])

    assert bot.closed


def test_outgoing_hands_its_files_over_once() -> None:
    outgoing = Outgoing()
    outgoing.add(Path("a.pdf"))

    assert outgoing.drain() == [Path("a.pdf")]
    assert outgoing.drain() == []
