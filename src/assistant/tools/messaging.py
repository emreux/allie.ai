"""`send_message`: a text from the user's own account to a person they know,
through WhatsApp or Telegram, after asking (spec section 7.1, 2026-09-15).

**One tool, two channels.** Every tool's schema rides on every request
(the tool-count note of 2026-09-11), and "message Ahmet on WhatsApp" and
"message Ahmet on Telegram" are one request with one word changed. So the
app is a parameter - an enum whose values are the names as people say
them, `WhatsApp` and `Telegram`, because the confirm question reads the
value out loud ("... WhatsApp üzerinden gönderilecek") and a key would have
needed a lookup table to say.

**The user is asked first, and hears everything.** `risk="confirm"`: the
gate fills the question with the contact as the user said them, the app
and the text, word for word, before the tool runs (`agent/policy.py`). A
message to the wrong person is the worst failure this feature has, and a
recogniser's mistake in the text or the name is caught here, by the user's
ear, before anything is sent. The gate's own additions come for free: the
audit row, the "you already did this forty seconds ago" sentence when the
same person gets the same text twice, and the tool-count limit.

**What the channels share is small.** `resolve` the person the user named,
`closest` names to ask about when nobody matched, `send` to a resolved
person - three questions behind a protocol, so that this file knows nothing
of chat links, Enter keys or MTProto. A channel with a sentence instead of
a person - a contact with no WhatsApp number, a Telegram not logged in -
raises `NoRecipientError` with it, and that sentence is the tool's answer.

**A guessed person is put to the user by their full name first.** The
question the gate asks is filled from the model's arguments, so it reads
the name as the user *said* it. When that name found somebody only by
being close to theirs ("ahmede", or a "Mehmet" that is not in the book),
the question would read "Mehmet" while the message went to Ahmet - the
one failure this feature must not have (measured with the real model,
2026-09-15). So a match that is not a listed name, an alias or a whole
word of one is not sent: the answer names the person found and tells the
model to call again with that full name, and the second question reads it.
One extra round trip, only for guesses.

The answers are addressed to the model: English, one line, what happened
and what to do next. The one sentence the user hears, the question, comes
from the locale pack with `TEXT` as the end of the chain (section 3.12).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Any, Literal, Protocol

from assistant.messaging.contacts import AddressBook, Contact
from assistant.messaging.whatsapp import MAX_TEXT_CHARS, WAKE_SECONDS, Outcome
from assistant.tools.registry import Tool, tool

__all__ = [
    "MATCHED",
    "NO_CHANNEL",
    "NO_CONTACT",
    "NO_NUMBER",
    "TEXT",
    "TOO_LONG",
    "App",
    "BadDefaultAppError",
    "Channel",
    "NoRecipientError",
    "Recipient",
    "WhatsAppChannel",
    "WhatsAppSender",
    "send_message_for",
]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from this tool: the question before a message goes.
TEXT: dict[str, str] = {
    "send_message_confirm": "The message '{text}' will be sent to {contact} on {app}.",
}

# The answers, addressed to the model.
NO_CHANNEL = "There is no messaging app called {app!r}; the choices are WhatsApp and Telegram."
NO_CONTACT = "No contact called {contact!r}{closest}."
NO_NUMBER = "{name} has no WhatsApp number in contacts.toml."
MATCHED = (
    "{contact!r} is not a listed name; the closest contact is {name!r}. Nothing was sent. "
    "Call again now with contact={name!r} and the same text: the user is asked to confirm "
    "that name before anything is sent, so do not ask them yourself."
)
TOO_LONG = (
    "The message is {length} characters; the limit is {limit}, because it is read back "
    "to the user first."
)
EMPTY = "The message is empty. Ask the user what to say."

# What the WhatsApp channel answers for each outcome (`messaging/whatsapp.py`).
WHATSAPP_SAID: dict[Outcome, str] = {
    "pressed": (
        "Handed to WhatsApp and Enter pressed - the message to {name} should be on its way; "
        "WhatsApp gives no confirmation."
    ),
    "placed": (
        "The message to {name} is typed into the WhatsApp chat and waiting: WhatsApp did not "
        "come to the front, so Enter was not pressed - the user presses it."
    ),
    "no_window": "WhatsApp did not open a window within {seconds:.0f} seconds; nothing was sent.",
    "not_installed": "WhatsApp is not installed; open_app can offer it from the Store.",
}

App = Literal["WhatsApp", "Telegram"]
APPS: tuple[str, ...] = ("WhatsApp", "Telegram")

# What the model is told about `app`. The annotation below has to be a
# module constant - `get_type_hints` evaluates it in the module's namespace,
# not the closure's - so the version with the user's default is written
# into the schema afterwards (`send_message_for`).
APP_ASK = "The app to send from. Ask which one when the user did not say."
APP_DEFAULT = "The app to send from. When the user did not name one, use {default}."


class BadDefaultAppError(ValueError):
    """`[messaging] default_app` names an app that is not a channel.
    Fixable by the user, so named for `run` (`__main__`)."""


class NoRecipientError(Exception):
    """A person the channel knows but cannot deliver to; the message is the
    sentence for the model."""


class Recipient(Protocol):
    """Whoever a channel resolved: the one thing the tool reads off them."""

    @property
    def name(self) -> str: ...


class Channel[R: Recipient](Protocol):
    """One way of sending. `WhatsAppChannel` and `messaging/telegram.py`'s
    `Telegram` are the two; the tests' fakes are the rest."""

    async def resolve(self, spoken: str) -> R | None: ...

    async def closest(self, spoken: str) -> list[str]: ...

    async def send(self, recipient: R, text: str) -> str: ...


def send_message_for(
    channels: Mapping[str, Channel[Any]],
    *,
    default_app: str = "",
    confirm_prompt: str = TEXT["send_message_confirm"],
) -> Tool:
    """`send_message`, bound to the channels on offer and the question it asks.

    `channels` is keyed by the app's name in any case; `default_app` names
    the one to use when the user did not say, or is empty to have the model
    ask. A default that is not a channel is refused here, at startup, the
    way a bad locale code is - a typo in `config.toml` would otherwise be
    a refusal in the middle of every turn.
    """
    by_key = {name.casefold(): channel for name, channel in channels.items()}
    default = default_app.strip()
    if default and default.casefold() not in by_key:
        raise BadDefaultAppError(
            f"[messaging] default_app is {default!r}, which is not one of: {', '.join(APPS)}"
        )

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def send_message(
        app: Annotated[App, APP_ASK],
        contact: Annotated[
            str,
            "The person as the user said them - a first name, a full name or a nickname. "
            "Strip only the case ending ('Ahmet'e' is 'Ahmet'); never translate a name or "
            "add a surname.",
        ],
        text: Annotated[
            str,
            "The message itself, in the user's words with ordinary punctuation. Compose it "
            "yourself only when the user asked you to ('tell him I'll be late').",
        ],
    ) -> str:
        """Sends a text message from the user's own account to one of their
        contacts, through WhatsApp or Telegram. Use it for every request to
        message, text or write to a person. The user is asked to confirm
        first and hears the recipient and the text, so pass both exactly.
        When no contact matches, the answer lists the closest names: ask the
        user which they meant, then call again with that name. The answer
        says whether the message went - repeat that to the user as it is
        said."""
        channel = by_key.get(app.strip().casefold())
        if channel is None:
            return NO_CHANNEL.format(app=app)
        words = text.strip()
        if not words:
            return EMPTY
        if len(words) > MAX_TEXT_CHARS:
            return TOO_LONG.format(length=len(words), limit=MAX_TEXT_CHARS)

        try:
            person = await channel.resolve(contact)
        except NoRecipientError as why:
            return str(why)
        if person is None:
            near = await channel.closest(contact)
            closest = f"; closest names: {', '.join(near)}" if near else ""
            return NO_CONTACT.format(contact=contact, closest=closest)
        return await channel.send(person, words)

    if not default:
        return send_message
    parameters = copy.deepcopy(dict(send_message.spec.parameters))
    parameters["properties"]["app"]["description"] = APP_DEFAULT.format(default=default)
    return replace(send_message, spec=replace(send_message.spec, parameters=parameters))


class WhatsAppSender(Protocol):
    """The slice of `messaging/whatsapp.py::WhatsApp` this channel uses."""

    async def send(self, phone: str, text: str) -> Outcome: ...


class WhatsAppChannel:
    """WhatsApp over the address book: the person's number, the app's link."""

    def __init__(self, whatsapp: WhatsAppSender, book: AddressBook) -> None:
        self._whatsapp = whatsapp
        self._book = book

    async def resolve(self, spoken: str) -> Contact | None:
        found = self._book.find(spoken)
        if found is None:
            return None
        if not self._book.certain(spoken, found):
            raise NoRecipientError(MATCHED.format(contact=spoken, name=found.name))
        if not found.phone:
            raise NoRecipientError(NO_NUMBER.format(name=found.name))
        return found

    async def closest(self, spoken: str) -> list[str]:
        return self._book.closest(spoken)

    async def send(self, recipient: Contact, text: str) -> str:
        outcome = await self._whatsapp.send(recipient.phone, text)
        return WHATSAPP_SAID[outcome].format(name=recipient.name, seconds=WAKE_SECONDS)
