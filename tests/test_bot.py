"""Telegram's Bot API, the few calls the phone remote makes (plan.md D57).

The client is tested against `httpx.MockTransport`: every test sees the
request the client made and answers it the way Telegram does - `{"ok": true,
"result": ...}` or `{"ok": false, ...}` - so nothing here reaches the network.
Three claims are worth the file.

**Plain text, always.** No `parse_mode` is ever sent, so nothing the model
writes - an asterisk, an underscore - can make Telegram refuse a message.

**The token never leaves in an error.** It is part of every request's
address; an error that carried the address would carry the token into a log
line or a traceback.

**An update the remote does not read still moves the offset**, or Telegram
would hand it over again at every poll.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from allie.remote import bot as bot_module
from allie.remote.bot import TEXT_LIMIT, BotApi, BotError, BotInfo, Incoming, Press, split_text

# Made up for the tests; the shape of a real one - the bot's id, a colon, the secret.
TOKEN = "123456:SECRET-token_value"  # noqa: S105  # a test's stand-in, not a secret


class Telegram:
    """A stand-in for api.telegram.org: records each request, answers from a script."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else ok(True)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def methods(self) -> list[str]:
        return [request.url.path.rsplit("/", 1)[-1] for request in self.requests]

    def body(self, index: int = -1) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[index].content)
        return loaded


def ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


def refused(status: int, description: str, **parameters: Any) -> httpx.Response:
    answer: dict[str, Any] = {"ok": False, "error_code": status, "description": description}
    if parameters:
        answer["parameters"] = parameters
    return httpx.Response(status, json=answer)


def bot_for(telegram: Telegram) -> BotApi:
    return BotApi(TOKEN, client=httpx.AsyncClient(transport=httpx.MockTransport(telegram)))


# --------------------------------------------------------------------------
# Who the bot is, and what was written to it
# --------------------------------------------------------------------------


async def test_me_names_the_bot() -> None:
    telegram = Telegram(
        ok({"id": 123456, "is_bot": True, "first_name": "allie", "username": "allie_test_bot"})
    )

    info = await bot_for(telegram).me()

    assert info == BotInfo(id=123456, username="allie_test_bot")
    assert str(telegram.requests[0].url) == f"https://api.telegram.org/bot{TOKEN}/getMe"


async def test_the_poll_asks_for_messages_and_presses_from_the_offset_on() -> None:
    telegram = Telegram(ok([]))

    batch = await bot_for(telegram).updates(7, wait=50)

    assert telegram.methods == ["getUpdates"]
    assert telegram.body() == {
        "offset": 7,
        "timeout": 50,
        "allowed_updates": ["message", "callback_query"],
    }
    assert batch.updates == ()
    assert batch.next_offset is None


async def test_the_first_poll_names_no_offset() -> None:
    telegram = Telegram(ok([]))

    await bot_for(telegram).updates(None, wait=0)

    assert "offset" not in telegram.body()


async def test_the_long_poll_is_given_longer_than_it_waits() -> None:
    """A read timeout shorter than the poll would end every quiet poll in a
    failure of our own making."""
    telegram = Telegram(ok([]))

    await bot_for(telegram).updates(None, wait=50)

    assert telegram.requests[0].extensions["timeout"]["read"] > 50


MESSAGE = {
    "update_id": 10,
    "message": {
        "message_id": 5,
        "date": 1761200000,
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 42, "is_bot": False, "first_name": "Emre"},
        "text": "şarj kaç",
    },
}
GROUP = {
    "update_id": 11,
    "message": {
        "message_id": 6,
        "date": 1761200001,
        "chat": {"id": -100, "type": "group"},
        "from": {"id": 42, "is_bot": False, "first_name": "Emre"},
        "text": "selam",
    },
}
PRESS = {
    "update_id": 12,
    "callback_query": {
        "id": "q-1",
        "from": {"id": 42, "is_bot": False, "first_name": "Emre"},
        "message": {"message_id": 9, "date": 1761200002, "chat": {"id": 42, "type": "private"}},
        "data": "a1b2:yes",
    },
}
EDITED = {
    "update_id": 13,
    "edited_message": {
        "message_id": 5,
        "date": 1761200003,
        "chat": {"id": 42, "type": "private"},
        "text": "şarj kaç?",
    },
}


async def test_a_message_and_a_press_are_read_and_anything_else_still_moves_the_offset() -> None:
    telegram = Telegram(ok([MESSAGE, GROUP, PRESS, EDITED]))

    batch = await bot_for(telegram).updates(None)

    assert batch.updates == (
        Incoming(
            update_id=10, chat_id=42, private=True, sender_id=42, text="şarj kaç", date=1761200000
        ),
        Incoming(
            update_id=11, chat_id=-100, private=False, sender_id=42, text="selam", date=1761200001
        ),
        Press(
            update_id=12, query_id="q-1", sender_id=42, chat_id=42, message_id=9, data="a1b2:yes"
        ),
    )
    assert batch.next_offset == 14


async def test_a_message_without_words_is_read_as_empty() -> None:
    """A photo or a voice note: there is a message, with no text in it."""
    photo = {
        "update_id": 20,
        "message": {
            "message_id": 1,
            "date": 1761200010,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False, "first_name": "Emre"},
            "photo": [{"file_id": "x", "width": 1, "height": 1}],
        },
    }

    batch = await bot_for(Telegram(ok([photo]))).updates(None)

    assert batch.updates == (
        Incoming(update_id=20, chat_id=42, private=True, sender_id=42, text="", date=1761200010),
    )


# --------------------------------------------------------------------------
# What the bot writes
# --------------------------------------------------------------------------


async def test_a_message_is_plain_text_and_answers_with_its_id() -> None:
    telegram = Telegram(ok({"message_id": 77}))

    sent = await bot_for(telegram).send(42, "*Şarj* %80_")

    assert sent == 77
    assert telegram.methods == ["sendMessage"]
    assert telegram.body() == {"chat_id": 42, "text": "*Şarj* %80_"}


async def test_buttons_go_under_the_message_in_one_row() -> None:
    telegram = Telegram(ok({"message_id": 78}))

    await bot_for(telegram).send(
        42, "Gönderilsin mi?", buttons=[("Evet", "q1:yes"), ("Hayır", "q1:no")]
    )

    assert telegram.body()["reply_markup"] == {
        "inline_keyboard": [
            [
                {"text": "Evet", "callback_data": "q1:yes"},
                {"text": "Hayır", "callback_data": "q1:no"},
            ]
        ]
    }


async def test_a_long_text_goes_in_pieces_with_the_buttons_on_the_last() -> None:
    first, second = "a" * 4000, "b" * 300
    telegram = Telegram(ok({"message_id": 1}), ok({"message_id": 2}))

    sent = await bot_for(telegram).send(42, f"{first}\n{second}", buttons=[("Evet", "q:yes")])

    assert [telegram.body(0)["text"], telegram.body(1)["text"]] == [first, second]
    assert "reply_markup" not in telegram.body(0)
    assert "reply_markup" in telegram.body(1)
    assert sent == 2


async def test_nothing_to_send_is_refused_before_telegram_is_asked() -> None:
    telegram = Telegram()

    with pytest.raises(ValueError):
        await bot_for(telegram).send(42, "  \n ")

    assert telegram.requests == []


def test_a_text_is_cut_at_a_line_break_then_a_space_then_anywhere() -> None:
    assert TEXT_LIMIT == 4096
    assert split_text("abc\ndef", limit=5) == ["abc", "def"]
    assert split_text("abc def", limit=5) == ["abc", "def"]
    assert split_text("abcdefgh", limit=5) == ["abcde", "fgh"]
    assert split_text("short", limit=5) == ["short"]
    assert split_text("   \n ", limit=5) == []


async def test_a_document_goes_as_a_file_under_its_own_name_with_the_caption(
    tmp_path: Path,
) -> None:
    document = tmp_path / "Sözleşme 2026.pdf"
    document.write_bytes(b"%PDF-1.7 tiny")
    telegram = Telegram(ok({"message_id": 90}))

    sent = await bot_for(telegram).send_document(42, document, caption="İşte sözleşme.")

    sent_bytes = telegram.requests[0].content
    assert sent == 90
    assert telegram.methods == ["sendDocument"]
    assert b'name="document"' in sent_bytes
    assert "Sözleşme 2026.pdf".encode() in sent_bytes
    assert b"%PDF-1.7 tiny" in sent_bytes
    assert b'name="chat_id"' in sent_bytes
    assert "İşte sözleşme.".encode() in sent_bytes


async def test_a_document_with_no_words_has_no_caption(tmp_path: Path) -> None:
    document = tmp_path / "fatura.pdf"
    document.write_bytes(b"%PDF-1.7 tiny")
    telegram = Telegram(ok({"message_id": 91}))

    await bot_for(telegram).send_document(42, document)

    assert b'name="caption"' not in telegram.requests[0].content


async def test_a_file_over_telegrams_limit_is_refused_before_anything_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "büyük.pdf"
    document.write_bytes(b"x" * 11)
    monkeypatch.setattr(bot_module, "MAX_FILE_BYTES", 10)
    telegram = Telegram()

    with pytest.raises(BotError) as refusal:
        await bot_for(telegram).send_document(42, document)

    assert refusal.value.kind == "too_big"
    assert telegram.requests == []


async def test_typing_a_tap_answered_and_a_message_rewritten() -> None:
    telegram = Telegram(ok(True), ok(True), ok(True), ok({"message_id": 9}))
    bot = bot_for(telegram)

    await bot.typing(42)
    await bot.answer_button("q-1")
    await bot.answer_button("q-2", "Bu soru kapandı.")
    await bot.edit(42, 9, "Gönderilsin mi?\n→ Evet")

    assert telegram.methods == [
        "sendChatAction",
        "answerCallbackQuery",
        "answerCallbackQuery",
        "editMessageText",
    ]
    assert telegram.body(0) == {"chat_id": 42, "action": "typing"}
    assert telegram.body(1) == {"callback_query_id": "q-1"}
    assert telegram.body(2) == {"callback_query_id": "q-2", "text": "Bu soru kapandı."}
    assert telegram.body(3) == {"chat_id": 42, "message_id": 9, "text": "Gönderilsin mi?\n→ Evet"}


# --------------------------------------------------------------------------
# What a failure is turned into
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        (refused(401, "Unauthorized"), "token"),
        (refused(404, "Not Found"), "token"),
        (
            refused(
                409,
                "Conflict: terminated by other getUpdates request; make sure that only one "
                "bot instance is running",
            ),
            "conflict",
        ),
        (refused(429, "Too Many Requests: retry after 7", retry_after=7), "rate_limit"),
        (refused(400, "Bad Request: chat not found"), "refused"),
        (httpx.Response(200, json={"ok": False, "description": "odd"}), "refused"),
        (refused(502, "Bad Gateway"), "unreachable"),
        (httpx.Response(502, text="<html>Bad Gateway</html>"), "unreachable"),
        (httpx.ConnectError("[Errno 11001] getaddrinfo failed"), "unreachable"),
        (httpx.ReadTimeout("The read operation timed out"), "unreachable"),
    ],
)
async def test_every_failure_has_a_kind_and_never_the_token(
    answer: httpx.Response | Exception, kind: str
) -> None:
    telegram = Telegram(answer)

    with pytest.raises(BotError) as failure:
        await bot_for(telegram).send(42, "selam")

    error = failure.value
    assert error.kind == kind
    for said in (str(error), repr(error)):
        assert "SECRET" not in said
        assert "api.telegram.org/bot" not in said
    # Nothing behind it that a traceback would print: the address is in there.
    assert error.__context__ is None
    assert error.__cause__ is None


async def test_slow_down_says_for_how_long() -> None:
    telegram = Telegram(refused(429, "Too Many Requests: retry after 7", retry_after=7))

    with pytest.raises(BotError) as failure:
        await bot_for(telegram).send(42, "selam")

    assert failure.value.retry_after == 7


async def test_a_client_handed_in_is_left_open() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(Telegram()))

    await BotApi(TOKEN, client=client).aclose()

    assert not client.is_closed
    await client.aclose()
