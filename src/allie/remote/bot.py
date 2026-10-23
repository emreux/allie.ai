"""Telegram's Bot API, the few calls the phone remote makes (plan.md D57).

**A bot, asked rather than listening.** The computer asks Telegram for what
was written to the bot (`getUpdates`) and Telegram holds the question open
for up to `POLL_SECONDS` until something comes - a long poll. Nothing on this
machine listens, no port is opened, no address is needed, and a message
written while the computer was away waits on Telegram's side for up to a day.
It is the Bot API over `httpx`, which the project already depends on: no
`api_id` from my.telegram.org and no session secret, only BotFather's token
(the spec of 2026-10-23, section 1, has the other routes and why not).

**Plain text, always.** No `parse_mode` is ever sent. Telegram reads `*` and
`_` as markup under one, and a model's answer with an odd asterisk in it
would be refused rather than shown.

**The token is in every request's address** (`/bot<token>/<method>`), so no
`httpx` exception and no address leaves this file: a failure becomes a
`BotError` with a `kind`, the status and Telegram's own `description`, raised
outside the `except` that caught it, so that a traceback has nothing behind
it to print. `httpx`'s own request lines go to the standard `logging`
module, which `logs.py` never routes to the log file.

The kinds are what the channel decides by: `token` (401/404 - revoked or
mistyped), `conflict` (409 - something else reads this bot's messages),
`rate_limit` (429, with Telegram's `retry_after`), `unreachable` (no network,
a timeout, or Telegram's own 5xx - all of which pass), `refused` (any other
4xx) and `too_big` (a file over the Bot API's 50 MB, refused before it is
read).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "API",
    "BOT_ENTRY",
    "CAPTION_LIMIT",
    "MAX_FILE_BYTES",
    "POLL_SECONDS",
    "TEXT_LIMIT",
    "Batch",
    "BotApi",
    "BotError",
    "BotInfo",
    "Incoming",
    "Press",
    "Update",
    "split_text",
]

API = "https://api.telegram.org"

# The Credential Manager entry the bot's token is stored under, beside
# Telegram's `telegram` and `telegram-session`; `allie purge --all` deletes it.
BOT_ENTRY = "telegram-bot"

# How long Telegram holds a poll open when nothing comes; the read timeout of
# that request is longer by `POLL_MARGIN_SECONDS`, or every quiet poll would
# end in a timeout of our own making.
POLL_SECONDS = 50
POLL_MARGIN_SECONDS = 10.0

# Every other call, and an upload, which may carry up to 50 MB.
CALL_SECONDS = 15.0
UPLOAD_SECONDS = 120.0

# Telegram's own limits: a message's text, a file's caption, a file a bot may
# send through this API.
TEXT_LIMIT = 4096
CAPTION_LIMIT = 1024
MAX_FILE_BYTES = 50 * 1024 * 1024

# What the remote reads: what is written to the bot, and its buttons pressed.
ALLOWED_UPDATES = ("message", "callback_query")


@dataclass(frozen=True, slots=True)
class BotInfo:
    """The bot, as `getMe` names it: the settings row shows `@username`."""

    id: int
    username: str


@dataclass(frozen=True, slots=True)
class Incoming:
    """A message written to the bot. `text` is empty for a message with no
    words in it - a photo, a voice note. `date` is Telegram's, in seconds."""

    update_id: int
    chat_id: int
    private: bool
    sender_id: int
    text: str
    date: int


@dataclass(frozen=True, slots=True)
class Press:
    """A button pressed under one of the bot's messages; `data` is what the
    button was sent with."""

    update_id: int
    query_id: str
    sender_id: int
    chat_id: int
    message_id: int
    data: str


Update = Incoming | Press


@dataclass(frozen=True, slots=True)
class Batch:
    """One poll's worth. `next_offset` confirms everything the poll held -
    the updates read and the ones skipped alike - and is `None` when it held
    nothing."""

    updates: tuple[Update, ...]
    next_offset: int | None


class BotError(Exception):
    """A call Telegram did not answer, by `kind` (the module's docstring).
    It never carries the request's address, which carries the token."""

    def __init__(
        self, kind: str, *, status: int = 0, description: str = "", retry_after: float = 0.0
    ) -> None:
        said = f"telegram: {kind}"
        if status:
            said = f"{said} ({status})"
        if description:
            said = f"{said}: {description}"
        super().__init__(said)
        self.kind = kind
        self.status = status
        self.description = description
        self.retry_after = retry_after


def split_text(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """`text` in pieces Telegram takes: cut at the last line break before the
    limit, else at the last space, else at the limit itself. Blank text is no
    piece at all, and no piece is ever blank."""
    pieces: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        piece = rest[:cut].rstrip()
        if piece:
            pieces.append(piece)
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


class BotApi:
    """The bot's calls. A client handed in is the caller's to close; one made
    here is closed by `aclose`."""

    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base = f"{API}/bot{token}/"
        self._client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def me(self) -> BotInfo:
        result = await self._call("getMe", {})
        return BotInfo(id=int(result["id"]), username=str(result.get("username", "")))

    async def updates(self, offset: int | None, *, wait: int = POLL_SECONDS) -> Batch:
        """What was written to the bot from `offset` on, waiting up to `wait`
        seconds for something to come."""
        body: dict[str, Any] = {"timeout": wait, "allowed_updates": list(ALLOWED_UPDATES)}
        if offset is not None:
            body = {"offset": offset, **body}
        result = await self._call(
            "getUpdates",
            body,
            seconds=httpx.Timeout(CALL_SECONDS, read=wait + POLL_MARGIN_SECONDS),
        )
        read: list[Update] = []
        last: int | None = None
        for update in result:
            update_id = int(update["update_id"])
            last = update_id if last is None else max(last, update_id)
            parsed = _parsed(update)
            if parsed is not None:
                read.append(parsed)
        return Batch(tuple(read), None if last is None else last + 1)

    async def send(self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()) -> int:
        """Writes `text` to `chat` - in pieces when it is longer than Telegram
        takes, the buttons under the last - and answers with the last piece's
        message id."""
        pieces = split_text(text)
        if not pieces:
            raise ValueError("there is nothing to send")
        sent = 0
        for number, piece in enumerate(pieces, start=1):
            body: dict[str, Any] = {"chat_id": chat, "text": piece}
            if buttons and number == len(pieces):
                body["reply_markup"] = {
                    "inline_keyboard": [
                        [{"text": label, "callback_data": data} for label, data in buttons]
                    ]
                }
            result = await self._call("sendMessage", body)
            sent = int(result["message_id"])
        return sent

    async def send_document(self, chat: int, path: Path, *, caption: str = "") -> int:
        """Sends the file at `path` as a document under its own name, with
        `caption` under it when there is one; answers with its message id."""
        status = await asyncio.to_thread(path.stat)
        if status.st_size > MAX_FILE_BYTES:
            raise BotError("too_big")
        content = await asyncio.to_thread(path.read_bytes)
        fields = {"chat_id": str(chat)}
        if caption:
            fields["caption"] = caption
        result = await self._call(
            "sendDocument",
            fields=fields,
            files={"document": (path.name, content)},
            seconds=UPLOAD_SECONDS,
        )
        return int(result["message_id"])

    async def typing(self, chat: int) -> None:
        """Shows "typing..." in the chat for a few seconds."""
        await self._call("sendChatAction", {"chat_id": chat, "action": "typing"})

    async def answer_button(self, query_id: str, text: str = "") -> None:
        """Tells Telegram a button press was received, with a line to show
        when there is one; until it is told, the button keeps spinning."""
        body: dict[str, Any] = {"callback_query_id": query_id}
        if text:
            body["text"] = text
        await self._call("answerCallbackQuery", body)

    async def edit(self, chat: int, message_id: int, text: str) -> None:
        """Rewrites one of the bot's messages; its buttons go with the edit."""
        await self._call(
            "editMessageText", {"chat_id": chat, "message_id": message_id, "text": text}
        )

    async def _call(
        self,
        method: str,
        body: Mapping[str, Any] | None = None,
        *,
        fields: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[str, bytes]] | None = None,
        seconds: float | httpx.Timeout = CALL_SECONDS,
    ) -> Any:
        """One request, and its `result` - or a `BotError` raised outside the
        `except` that caught the failure, so that nothing carrying the address
        rides along behind it."""
        failure: BotError
        try:
            if files is not None:
                response = await self._client.post(
                    self._base + method, data=fields, files=files, timeout=seconds
                )
            else:
                response = await self._client.post(
                    self._base + method, json=body or {}, timeout=seconds
                )
        except httpx.TimeoutException:
            failure = BotError("unreachable", description="timed out")
        except httpx.HTTPError as error:
            failure = BotError("unreachable", description=type(error).__name__)
        else:
            return _result(response)
        raise failure


def _result(response: httpx.Response) -> Any:
    """Telegram's `result`, or the failure its answer describes."""
    answer: Any = None
    try:
        answer = response.json()
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise BotError(_kind(response.status_code), status=response.status_code)
    if response.status_code == 200 and answer.get("ok") is True:
        return answer.get("result")
    parameters = answer.get("parameters")
    retry_after = parameters.get("retry_after", 0) if isinstance(parameters, dict) else 0
    raise BotError(
        _kind(response.status_code),
        status=response.status_code,
        description=str(answer.get("description", "")),
        retry_after=float(retry_after or 0),
    )


def _kind(status: int) -> str:
    """What a status means for the channel (the module's docstring)."""
    if status in (401, 404):
        return "token"
    if status == 409:
        return "conflict"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "unreachable"
    return "refused"


def _parsed(update: Mapping[str, Any]) -> Update | None:
    """A message or a press, or `None` for anything else the poll held."""
    update_id = int(update["update_id"])
    message = update.get("message")
    if isinstance(message, dict):
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        return Incoming(
            update_id=update_id,
            chat_id=int(chat.get("id", 0)),
            private=chat.get("type") == "private",
            sender_id=int(sender.get("id", 0)),
            text=str(message.get("text") or ""),
            date=int(message.get("date", 0)),
        )
    query = update.get("callback_query")
    if isinstance(query, dict):
        pressed_on = query.get("message") or {}
        sender = query.get("from") or {}
        return Press(
            update_id=update_id,
            query_id=str(query.get("id", "")),
            sender_id=int(sender.get("id", 0)),
            chat_id=int((pressed_on.get("chat") or {}).get("id", 0)),
            message_id=int(pressed_on.get("message_id", 0)),
            data=str(query.get("data") or ""),
        )
    return None
