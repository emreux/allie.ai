"""`send_email`: a mail from the user's own account, after asking (plan.md
D43, 27 Sep 2026).

**Asked, read and shown.** `risk="confirm"`: the gate reads the recipient,
the subject and the text aloud and puts the same question on the screen
(`Tool.shown`) - an address is easier to check by eye than by ear.

**The question reads the real recipient.** `prepare` runs before it: "me"
becomes the account's own address, and a reply's recipient and subject are
taken from the mail it answers (found by the `Id:` that `read_emails`
shows), not from the model's words. An address that is not one, an empty or
too long text and a missing subject are said to the model instead of asked
about.

**Nobody is added anywhere.** An address the user spells is used once; the
address book (`contacts.toml`) is neither read nor written (the owner:
"contacta eklemeyelim").

**SMTP over SSL with the IMAP app password.** One connection per mail, on a
thread. Gmail files what it sends in the Sent folder by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import smtplib
from collections.abc import Callable
from dataclasses import replace
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from typing import Annotated, Any, Protocol

from allie.tools.mail import NOT_SET_UP, Mailbox, MailError, Original
from allie.tools.registry import Tool, tool

__all__ = [
    "BAD_ADDRESS",
    "EMPTY",
    "MAX_SUBJECT_CHARS",
    "MAX_TEXT_CHARS",
    "NO_ORIGINAL",
    "NO_SUBJECT",
    "SMTP_SECONDS",
    "TEXT",
    "TOO_LONG",
    "Outbox",
    "SmtpOutbox",
    "send_email_for",
]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from this tool (D36: ends in "?", no no word).
TEXT: dict[str, str] = {
    "send_email_confirm": "Shall I send the mail '{subject}' - '{text}' - to {to}?",
}

SMTP_SECONDS = 15.0
MAX_TEXT_CHARS = 1_000
MAX_SUBJECT_CHARS = 150
ME = "me"

EMPTY = "The mail has no text. Ask the user what to write."
TOO_LONG = (
    "The text is {length} characters; the limit is {limit}, because it is read back to the "
    "user first. Shorten it, or ask the user to."
)
NO_SUBJECT = "A mail needs a subject: write a short one from the text and call again."
SUBJECT_TOO_LONG = "The subject is {length} characters; keep it under {limit}."
BAD_ADDRESS = (
    "{to!r} is not an email address. Ask the user to spell it - letter by letter if need "
    "be. Nothing was sent."
)
NO_ORIGINAL = (
    "No message with the Id {id!r} is in the mailbox. Call read_emails and pass the Id it shows."
)
UNREADABLE = "{failure} The mail being answered could not be read; nothing was sent."
SENT = "Sent to {to}: {subject!r}."
FAILED = "{failure} Nothing was sent; tell the user."

_ADDRESS = re.compile(r"^[^@\s<>,;\"]+@[^@\s<>,;\"]+\.[^@\s<>,;\".]{2,}$")


class Outbox(Protocol):
    def send(self, message: EmailMessage) -> None: ...


class SmtpClient(Protocol):
    def login(self, user: str, password: str) -> Any: ...

    def send_message(self, message: EmailMessage) -> Any: ...

    def quit(self) -> Any: ...


SmtpFactory = Callable[[str, int, float], SmtpClient]


def _ssl_client(host: str, port: int, timeout: float) -> SmtpClient:
    return smtplib.SMTP_SSL(host, port, timeout=timeout)


class SmtpOutbox:
    """One mail per connection, signed in with the IMAP app password."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        *,
        timeout: float = SMTP_SECONDS,
        client_factory: SmtpFactory = _ssl_client,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._timeout = timeout
        self._client_factory = client_factory

    def send(self, message: EmailMessage) -> None:
        """Blocking; `MailError` with the server's words when it refuses."""
        try:
            client = self._client_factory(self._host, self._port, self._timeout)
        except OSError as failure:
            raise MailError(
                f"{self._host}:{self._port} could not be reached ({type(failure).__name__})"
            ) from failure
        try:
            client.login(self._user, self._password)
            client.send_message(message)
        except smtplib.SMTPException as refusal:
            raise MailError(f"{self._host} refused: {refusal}") from refusal
        except OSError as failure:
            raise MailError(f"{self._host} failed: {failure}") from failure
        finally:
            with contextlib.suppress(smtplib.SMTPException, OSError):
                client.quit()


def send_email_for(
    outbox: Callable[[], Outbox] | None,
    open_mailbox: Callable[[], Mailbox] | None,
    *,
    own_address: str,
    confirm_prompt: str = TEXT["send_email_confirm"],
) -> Tool:
    """`send_email`, bound to the outgoing server, the mailbox a reply is
    looked up in, the account's own address and the question it asks."""
    originals: dict[str, Original] = {}

    async def resolve(to: str, subject: str, text: str, reply_to: str) -> dict[str, str] | str:
        if outbox is None:
            return NOT_SET_UP
        words = text.strip()
        if not words:
            return EMPTY
        if len(words) > MAX_TEXT_CHARS:
            return TOO_LONG.format(length=len(words), limit=MAX_TEXT_CHARS)
        title = " ".join(subject.split())
        answering = reply_to.strip()
        if answering:
            original = originals.get(answering)
            if original is None:
                if open_mailbox is None:
                    return NOT_SET_UP
                try:
                    original = await asyncio.to_thread(open_mailbox().original, answering)
                except MailError as failure:
                    return UNREADABLE.format(failure=failure)
                if original is None:
                    return NO_ORIGINAL.format(id=answering)
                originals[answering] = original
            address = original.address
            title = (
                original.subject
                if original.subject.casefold().startswith("re:")
                else f"Re: {original.subject}"
            )
        else:
            said = to.strip()
            address = own_address if said.casefold() in (ME, own_address.casefold()) else said
        if not title:
            return NO_SUBJECT
        if len(title) > MAX_SUBJECT_CHARS:
            return SUBJECT_TOO_LONG.format(length=len(title), limit=MAX_SUBJECT_CHARS)
        if not _is_address(address):
            return BAD_ADDRESS.format(to=address)
        return {"to": address, "subject": title, "text": words, "reply_to": answering}

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def send_email(
        to: Annotated[
            str,
            "The address, written out from what the user said ('ahmet nokta yilmaz at gmail "
            "nokta com' is ahmet.yilmaz@gmail.com), or 'me' for the user's own address.",
        ],
        subject: Annotated[str, "A short subject; write one from the text if the user gave none."],
        text: Annotated[str, "The mail itself, in the user's words, plain text."],
        reply_to: Annotated[
            str,
            "For a reply: the Id that read_emails showed for the mail being answered; the "
            "recipient and the subject then come from that mail.",
        ] = "",
    ) -> str:
        """Sends an email from the user's own account - to an address they
        said, to themselves ('me'), or as a reply to a mail read_emails
        showed. The user is asked first and hears and sees the recipient, the
        subject and the text, so pass them exactly; do not ask them yourself.
        Say whether it went, as the answer says."""
        prepared = await resolve(to, subject, text, reply_to)
        if isinstance(prepared, str):
            return prepared
        message = EmailMessage()
        message["From"] = own_address
        message["To"] = prepared["to"]
        message["Subject"] = prepared["subject"]
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid(domain=own_address.partition("@")[2] or None)
        original = originals.get(prepared["reply_to"]) if prepared["reply_to"] else None
        if original is not None:
            message["In-Reply-To"] = original.message_id
            message["References"] = f"{original.references} {original.message_id}".strip()
        message.set_content(prepared["text"])
        if outbox is None:  # `resolve` has said so already; this is for the type
            return NOT_SET_UP
        try:
            await asyncio.to_thread(outbox().send, message)
        except MailError as failure:
            return FAILED.format(failure=failure)
        return SENT.format(to=prepared["to"], subject=prepared["subject"])

    async def prepare(to: str, subject: str, text: str, reply_to: str = "") -> dict[str, str] | str:
        return await resolve(to, subject, text, reply_to)

    return replace(send_email, prepare=prepare, shown=True)


def _is_address(address: str) -> bool:
    return parseaddr(address)[1] == address and bool(_ADDRESS.match(address))
