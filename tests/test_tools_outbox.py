"""`send_email` (D43): the question - read and shown - with the real
recipient and subject; "me"; a reply's recipient taken from the mail it
answers; what is refused before anybody is asked; and the SMTP client.
Nothing here sends mail."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

import pytest

from allie.agent.policy import dispatch
from allie.live.base import ToolCall
from allie.tools.mail import NOT_SET_UP, MailError, Original
from allie.tools.outbox import (
    BAD_ADDRESS,
    EMPTY,
    NO_ORIGINAL,
    NO_SUBJECT,
    TEXT,
    TOO_LONG,
    SmtpOutbox,
    send_email_for,
)
from allie.tools.registry import Tool, ToolRegistry

ME = "emre@example.test"
ORIGINAL = Original(
    address="fatura@enerji.test",
    subject="Fatura",
    message_id="<abc@enerji.test>",
    references="<first@enerji.test>",
)


class FakeOutbox:
    def __init__(self, failure: MailError | None = None) -> None:
        self.sent: list[EmailMessage] = []
        self.failure = failure

    def send(self, message: EmailMessage) -> None:
        if self.failure is not None:
            raise self.failure
        self.sent.append(message)


class Mailbox:
    def original(self, message_id: str) -> Original | None:
        return ORIGINAL if message_id == ORIGINAL.message_id else None


class User:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.shown: list[str] = []

    async def confirm(self, question: str) -> bool:
        self.asked.append(question)
        return True


def tool(outbox: FakeOutbox | None) -> Tool:
    return send_email_for(
        (lambda: outbox) if outbox is not None else None,
        lambda: Mailbox(),
        own_address=ME,
        confirm_prompt="{to} | {subject} | {text}",
    )


async def through_the_gate(sending: Tool, user: User, **arguments: Any) -> str:
    return await dispatch(
        ToolCall(id="c1", name="send_email", arguments=arguments),
        turn_id="t1",
        registry=ToolRegistry([sending]),
        confirm=user.confirm,
        show=user.shown.append,
    )


def test_it_asks_first_and_shows_the_question() -> None:
    sending = tool(FakeOutbox())

    assert sending.risk == "confirm"
    assert sending.shown is True
    assert sending.prepare is not None
    assert sending.spec.parameters["required"] == ["to", "subject", "text"]
    assert TEXT["send_email_confirm"] == "Shall I send the mail '{subject}' - '{text}' - to {to}?"


async def test_me_is_the_users_own_address() -> None:
    outbox = FakeOutbox()
    user = User()

    said = await through_the_gate(
        tool(outbox), user, to="me", subject="Toplantı", text="Saat üçte."
    )

    assert user.asked == [f"{ME} | Toplantı | Saat üçte."]
    assert user.shown == user.asked
    [sent] = outbox.sent
    assert (sent["To"], sent["From"], sent["Subject"]) == (ME, ME, "Toplantı")
    assert sent.get_content().strip() == "Saat üçte."
    assert sent["Message-ID"].endswith("@example.test>")
    assert said == f"Sent to {ME}: 'Toplantı'."


async def test_a_spoken_address_is_used_as_it_is() -> None:
    outbox = FakeOutbox()

    await through_the_gate(
        tool(outbox), User(), to="ahmet.yilmaz@gmail.com", subject="Selam", text="Merhaba."
    )

    assert outbox.sent[0]["To"] == "ahmet.yilmaz@gmail.com"


@pytest.mark.parametrize("address", ["ahmet at gmail", "ahmet@gmail", "a b@c.com", "", "@x.com"])
async def test_an_address_that_is_not_one_is_refused_before_asking(address: str) -> None:
    outbox = FakeOutbox()
    user = User()

    said = await through_the_gate(tool(outbox), user, to=address, subject="S", text="T")

    assert said == BAD_ADDRESS.format(to=address)
    assert user.asked == [] and outbox.sent == []


async def test_empty_long_and_subjectless_mail_is_refused_before_asking() -> None:
    user = User()
    sending = tool(FakeOutbox())

    assert await through_the_gate(sending, user, to="me", subject="S", text="  ") == EMPTY
    assert (
        await through_the_gate(sending, user, to="me", subject="S", text="x" * 1001)
    ).startswith(TOO_LONG.split("{")[0])
    assert await through_the_gate(sending, user, to="me", subject=" ", text="T") == NO_SUBJECT
    assert user.asked == []


async def test_a_reply_goes_to_the_mail_it_answers_and_threads_under_it() -> None:
    outbox = FakeOutbox()
    user = User()

    await through_the_gate(
        tool(outbox),
        user,
        to="Enerji",
        subject="anything",
        text="Ödedim.",
        reply_to="<abc@enerji.test>",
    )

    assert user.asked == ["fatura@enerji.test | Re: Fatura | Ödedim."]
    [sent] = outbox.sent
    assert sent["To"] == "fatura@enerji.test"
    assert sent["In-Reply-To"] == "<abc@enerji.test>"
    assert sent["References"] == "<first@enerji.test> <abc@enerji.test>"


async def test_a_reply_to_a_mail_that_is_not_there_is_said() -> None:
    said = await through_the_gate(
        tool(FakeOutbox()), User(), to="x", subject="y", text="z", reply_to="<nope@x>"
    )

    assert said == NO_ORIGINAL.format(id="<nope@x>")


async def test_a_refusal_of_the_server_is_a_sentence_and_nothing_else() -> None:
    said = await through_the_gate(
        tool(FakeOutbox(MailError("smtp.x refused: 535"))),
        User(),
        to="me",
        subject="S",
        text="T",
    )

    assert said == "smtp.x refused: 535 Nothing was sent; tell the user."


async def test_without_mail_set_up_nothing_is_asked() -> None:
    user = User()

    assert await through_the_gate(tool(None), user, to="me", subject="S", text="T") == NOT_SET_UP
    assert user.asked == []


class FakeSmtp:
    def __init__(self, *, refuse: bool = False) -> None:
        self.calls: list[str] = []
        self.refuse = refuse

    def login(self, user: str, password: str) -> Any:
        self.calls.append(f"login {user}")
        if self.refuse:
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    def send_message(self, message: EmailMessage) -> Any:
        self.calls.append(f"send {message['To']}")

    def quit(self) -> Any:
        self.calls.append("quit")


def test_the_smtp_outbox_signs_in_sends_and_signs_out() -> None:
    client = FakeSmtp()
    dialled: list[tuple[str, int, float]] = []

    def dial(host: str, port: int, timeout: float) -> FakeSmtp:
        dialled.append((host, port, timeout))
        return client

    message = EmailMessage()
    message["To"] = ME
    SmtpOutbox("smtp.x.test", 465, ME, "pw", client_factory=dial).send(message)

    assert dialled == [("smtp.x.test", 465, 15.0)]
    assert client.calls == [f"login {ME}", f"send {ME}", "quit"]


def test_a_refused_sign_in_is_a_mail_error_and_the_connection_is_closed() -> None:
    client = FakeSmtp(refuse=True)

    with pytest.raises(MailError, match="refused"):
        SmtpOutbox("smtp.x.test", 465, ME, "pw", client_factory=lambda *_: client).send(
            EmailMessage()
        )

    assert client.calls[-1] == "quit"
