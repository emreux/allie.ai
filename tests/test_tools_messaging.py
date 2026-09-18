"""`send_message` (15 Sep 2026): one tool, two channels, and the user asked first.

The channels are fakes; the WhatsApp channel is tested over a fake
`WhatsApp` whose outcome the test chooses. Nothing here opens an app, and
the gate is `test_policy.py`'s business.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from assistant.messaging.contacts import AddressBook, Contact
from assistant.messaging.whatsapp import MAX_TEXT_CHARS, Outcome
from assistant.tools.messaging import (
    MATCHED,
    NO_CHANNEL,
    NO_CONTACT,
    NO_NUMBER,
    TEXT,
    TOO_LONG,
    NoRecipientError,
    Recipient,
    WhatsAppChannel,
    send_message_for,
)
from assistant.tools.registry import Tool

AHMET = Contact(name="Ahmet Yılmaz", aliases=("Ahmet",), phone="905320000000")
MEHMET = Contact(name="Mehmet Kaya", aliases=("Mehmet",))  # no number
BOOK = AddressBook([AHMET, MEHMET])


class Person:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeChannel:
    """Resolves the names it was given; records what it was asked to send."""

    def __init__(self, *known: str, near: Sequence[str] = ()) -> None:
        self.known = {name.casefold(): Person(name) for name in known}
        self.near = list(near)
        self.sent: list[tuple[str, str]] = []
        self.asked: list[str] = []

    async def resolve(self, spoken: str) -> Person | None:
        self.asked.append(spoken)
        return self.known.get(spoken.casefold())

    async def closest(self, spoken: str) -> list[str]:
        return self.near

    async def send(self, recipient: Recipient, text: str) -> str:
        self.sent.append((recipient.name, text))
        return f"Sent to {recipient.name}."


class FakeWhatsApp:
    def __init__(self, outcome: Outcome = "pressed", *, installed: bool = True) -> None:
        self.outcome = outcome
        self._installed = installed
        self.sent: list[tuple[str, str]] = []

    def installed(self) -> bool:
        return self._installed

    async def send(self, phone: str, text: str) -> Outcome:
        self.sent.append((phone, text))
        return self.outcome


@pytest.fixture
def whatsapp() -> FakeChannel:
    return FakeChannel("Ahmet Yılmaz", near=["Ahmet Yılmaz", "Mehmet Kaya"])


@pytest.fixture
def telegram() -> FakeChannel:
    return FakeChannel("Ada")


@pytest.fixture
def send_message(whatsapp: FakeChannel, telegram: FakeChannel) -> Tool:
    return send_message_for({"whatsapp": whatsapp, "telegram": telegram})


# --------------------------------------------------------------------------
# The tool as the model sees it
# --------------------------------------------------------------------------


def test_it_asks_first_and_the_question_carries_all_three_arguments(send_message: Tool) -> None:
    assert send_message.risk == "confirm"
    assert send_message.confirm_prompt == TEXT["send_message_confirm"]
    for field in ("{contact}", "{app}", "{text}"):
        assert field in TEXT["send_message_confirm"]
    assert send_message.spec.parameters["required"] == ["app", "contact", "text"]


def test_the_app_is_an_enum_of_the_two_names_as_people_say_them(send_message: Tool) -> None:
    """The question reads the enum's value out loud - "WhatsApp üzerinden" -
    so the value is the name and not a key."""
    assert send_message.spec.parameters["properties"]["app"]["enum"] == ["WhatsApp", "Telegram"]


def test_without_a_default_the_model_is_told_to_ask(send_message: Tool) -> None:
    described = send_message.spec.parameters["properties"]["app"]["description"]

    assert "Ask which one" in described


def test_with_a_default_the_model_is_told_which(whatsapp: FakeChannel) -> None:
    tool = send_message_for({"whatsapp": whatsapp}, default_app="WhatsApp")

    assert "use WhatsApp" in tool.spec.parameters["properties"]["app"]["description"]


def test_a_default_that_is_not_a_channel_is_refused_at_startup(whatsapp: FakeChannel) -> None:
    with pytest.raises(ValueError, match="Signal"):
        send_message_for({"whatsapp": whatsapp}, default_app="Signal")


def test_the_question_comes_from_the_pack(whatsapp: FakeChannel) -> None:
    tool = send_message_for({"whatsapp": whatsapp}, confirm_prompt="{text} -> {contact} ({app})")

    assert tool.confirm_prompt == "{text} -> {contact} ({app})"


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


async def test_the_message_goes_to_the_channel_the_model_named(
    send_message: Tool, whatsapp: FakeChannel, telegram: FakeChannel
) -> None:
    said = await send_message.run(app="WhatsApp", contact="Ahmet Yılmaz", text="yarın geliyorum")

    assert said == "Sent to Ahmet Yılmaz."
    assert whatsapp.sent == [("Ahmet Yılmaz", "yarın geliyorum")]
    assert telegram.sent == []


async def test_the_app_name_is_matched_without_regard_to_case(
    send_message: Tool, telegram: FakeChannel
) -> None:
    await send_message.run(app="telegram", contact="Ada", text="hi")

    assert telegram.sent == [("Ada", "hi")]


async def test_an_app_that_is_not_a_channel_is_refused_in_words(send_message: Tool) -> None:
    said = await send_message.run(app="Signal", contact="Ada", text="hi")

    assert said == NO_CHANNEL.format(app="Signal")


async def test_a_text_too_long_to_read_out_is_refused_before_anyone_is_looked_up(
    send_message: Tool, whatsapp: FakeChannel
) -> None:
    text = "a" * (MAX_TEXT_CHARS + 1)

    said = await send_message.run(app="WhatsApp", contact="Ahmet", text=text)

    assert said == TOO_LONG.format(length=MAX_TEXT_CHARS + 1, limit=MAX_TEXT_CHARS)
    assert whatsapp.asked == []


async def test_an_empty_text_is_refused(send_message: Tool, whatsapp: FakeChannel) -> None:
    said = await send_message.run(app="WhatsApp", contact="Ahmet", text="   ")

    assert "empty" in said
    assert whatsapp.sent == []


async def test_nobody_by_that_name_lists_the_closest_and_sends_nothing(
    send_message: Tool, whatsapp: FakeChannel
) -> None:
    said = await send_message.run(app="WhatsApp", contact="Ahmed", text="hi")

    assert said == NO_CONTACT.format(
        contact="Ahmed", closest="; closest names: Ahmet Yılmaz, Mehmet Kaya"
    )
    assert whatsapp.sent == []


async def test_nobody_and_nothing_near_is_said_plainly(telegram: FakeChannel) -> None:
    tool = send_message_for({"telegram": telegram})

    said = await tool.run(app="Telegram", contact="Zed", text="hi")

    assert said == NO_CONTACT.format(contact="Zed", closest="")


async def test_a_person_the_channel_cannot_reach_is_the_channels_own_sentence(
    send_message: Tool, whatsapp: FakeChannel
) -> None:
    async def refuse(spoken: str) -> Person | None:
        raise NoRecipientError(f"{spoken} cannot be reached this way.")

    whatsapp.resolve = refuse  # type: ignore[method-assign]

    assert await send_message.run(app="WhatsApp", contact="Ada", text="hi") == (
        "Ada cannot be reached this way."
    )


# --------------------------------------------------------------------------
# The WhatsApp channel over the address book
# --------------------------------------------------------------------------


async def test_the_whatsapp_channel_finds_a_person_in_the_book_by_alias() -> None:
    channel = WhatsAppChannel(FakeWhatsApp(), BOOK)

    assert await channel.resolve("Ahmet") == AHMET
    assert await channel.resolve("Zed") is None
    assert set(await channel.closest("Ahmet Kaya")) == {"Ahmet Yılmaz", "Mehmet Kaya"}


async def test_a_person_without_a_number_is_known_but_unreachable() -> None:
    channel = WhatsAppChannel(FakeWhatsApp(), BOOK)

    with pytest.raises(NoRecipientError, match=NO_NUMBER.format(name="Mehmet Kaya")):
        await channel.resolve("Mehmet")


async def test_the_channel_hands_the_digits_and_the_text_to_whatsapp() -> None:
    app = FakeWhatsApp()
    channel = WhatsAppChannel(app, BOOK)

    await channel.send(AHMET, "yarın geliyorum")

    assert app.sent == [("905320000000", "yarın geliyorum")]


@pytest.mark.parametrize(
    ("outcome", "said"),
    [
        ("pressed", "Handed to WhatsApp and Enter pressed"),
        ("placed", "Enter was not pressed"),
        ("no_window", "did not open a window"),
        ("not_installed", "not installed"),
    ],
)
async def test_each_outcome_is_a_sentence_the_model_can_repeat(outcome: Outcome, said: str) -> None:
    channel = WhatsAppChannel(FakeWhatsApp(outcome), BOOK)

    answer = await channel.send(AHMET, "hi")

    assert said in answer
    assert "Ahmet Yılmaz" in answer or outcome in {"no_window", "not_installed"}


async def test_a_guessed_person_is_named_and_not_sent_to_until_named_back() -> None:
    """Measured with the real model, 2026-09-15: "Mehmet'e selam yaz" with no
    Mehmet in the book went to Ahmet Yılmaz, and the question read
    "Mehmet". A guess - a case ending, a near miss - is answered with the
    person's full name; the second call, with that name, is the one that
    sends, and its question reads the name."""
    app = FakeWhatsApp("placed")
    tool = send_message_for({"whatsapp": WhatsAppChannel(app, BOOK)})

    said = await tool.run(app="WhatsApp", contact="ahmede", text="geç kalıyorum")

    assert said == MATCHED.format(contact="ahmede", name="Ahmet Yılmaz")
    assert app.sent == []

    said = await tool.run(app="WhatsApp", contact="Ahmet Yılmaz", text="geç kalıyorum")

    assert app.sent == [("905320000000", "geç kalıyorum")]
    assert "Ahmet Yılmaz" in said and "Enter was not pressed" in said


async def test_a_name_that_is_not_in_the_book_is_not_sent_to_a_near_one() -> None:
    """ "memet" is nine tenths like "mehmet" and does not start like it: the
    person is offered, not chosen, and nothing is sent."""
    app = FakeWhatsApp()
    tool = send_message_for({"whatsapp": WhatsAppChannel(app, BOOK)})

    said = await tool.run(app="WhatsApp", contact="Memet", text="selam")

    assert said == NO_CONTACT.format(
        contact="Memet", closest="; closest names: Mehmet Kaya, Ahmet Yılmaz"
    )
    assert app.sent == []


@pytest.mark.parametrize("spoken", ["Ahmet", "Ahmet Yılmaz", "yılmaz", "AHMET"])
async def test_a_listed_name_an_alias_or_a_word_of_one_sends_at_once(spoken: str) -> None:
    app = FakeWhatsApp()
    tool = send_message_for({"whatsapp": WhatsAppChannel(app, BOOK)})

    await tool.run(app="WhatsApp", contact=spoken, text="selam")

    assert app.sent == [("905320000000", "selam")]
