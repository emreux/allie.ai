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
gate fills the question with the person as the address book names them,
the app and the text, word for word, before the tool runs
(`agent/policy.py`). A
message to the wrong person is the worst failure this feature has, and a
recogniser's mistake in the text or the name is caught here, by the user's
ear, before anything is sent. The gate's own additions come for free: the
audit row, the "you already did this forty seconds ago" sentence when the
same person gets the same text twice, and the tool-count limit.

**What the channels share is small.** `resolve` the person the user named,
`closest` names to ask about when nobody matched, `send` to a resolved
person - three questions behind a protocol, so that this file knows nothing
of chat links, Send buttons or MTProto. A channel with a sentence instead of
a person - a contact with no WhatsApp number, a Telegram not logged in -
raises `NoRecipientError` with it, and that sentence is the tool's answer.

**The person is found before the question, and the question reads them.**
A question filled from the model's arguments reads the name as the user
*said* it; when that name found somebody only by being close to theirs
("ahmede", or a "Mehmet" that is not in the book), it would read "Mehmet"
while the message went to Ahmet - the one failure this feature must not
have (measured with the real model, 2026-09-15). Until 2026-09-26 a guess
was answered with the full name and a second call, so the user was asked
twice - "Beho", then "Behoo" - and a name nobody had was asked about
before "no contact" was said. Now the tool's `prepare` runs before the
gate asks: the question reads the name the message goes to, a person who
cannot be reached is said instead of asked about, and one question is
asked. The tool itself still refuses to send on a guess (`MATCHED`).

The answers are addressed to the model: English, one line, what happened
and what to do next. The one sentence the user hears, the question, comes
from the locale pack with `TEXT` as the end of the chain (section 3.12).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Any, Literal, Protocol

from allie.messaging.contacts import AddressBook, Contact
from allie.messaging.whatsapp import MAX_TEXT_CHARS, TYPE_SECONDS, WAKE_SECONDS, Outcome
from allie.tools.registry import Tool, tool

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
    "GuessedRecipientError",
    "NoRecipientError",
    "Recipient",
    "WhatsAppChannel",
    "WhatsAppSender",
    "send_message_for",
]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from this tool: the question before a message goes.
TEXT: dict[str, str] = {
    "send_message_confirm": "Shall I send '{text}' to {contact} on {app}?",
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
    "sent": "Sent to {name} on WhatsApp: the message left the chat box after Send.",
    "pressed": (
        "Handed to WhatsApp and its Send button pressed - the message to {name} should be on "
        "its way; WhatsApp gives no confirmation."
    ),
    "placed": (
        "The message to {name} is typed into the WhatsApp chat and waiting, not sent: the "
        "chat box held other text too, or WhatsApp's Send button could not be found or did "
        "not send it. The user looks at the chat and sends it."
    ),
    "unseen": (
        "WhatsApp did not show the message to {name} in the chat box within {type_seconds:.0f} "
        "seconds, so nothing was pressed. The user checks WhatsApp."
    ),
    "no_window": "WhatsApp did not open a window within {seconds:.0f} seconds; nothing was sent.",
    "not_installed": "WhatsApp is not installed; open_app can offer it from the Store.",
}

App = Literal["WhatsApp", "Telegram"]
APPS: tuple[str, ...] = ("WhatsApp", "Telegram")
# The names as people say them, by any spelling of case the model used:
# the question reads "WhatsApp" whatever came in.
_SAID_AS = {name.casefold(): name for name in APPS}

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


class GuessedRecipientError(NoRecipientError):
    """Somebody found only by being close to the name the user said.

    The message is `MATCHED`, what the model is told should the tool ever
    run on a guess; `name` is the person's full name, which `prepare` puts
    in the question instead (2026-09-26).
    """

    def __init__(self, spoken: str, name: str) -> None:
        super().__init__(MATCHED.format(contact=spoken, name=name))
        self.name = name


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
        refused = _unsendable(channel, app, text)
        if refused is not None or channel is None:
            return refused or NO_CHANNEL.format(app=app)
        try:
            person = await channel.resolve(contact)
        except NoRecipientError as why:
            return str(why)
        if person is None:
            return await _nobody(channel, contact)
        return await channel.send(person, text.strip())

    async def prepare(app: str, contact: str, text: str) -> dict[str, str] | str:
        """The arguments the question reads and the message goes with: the
        app as people say it, the person as the book names them - a guess
        by its full name - or the sentence that there is nobody to ask
        about."""
        channel = by_key.get(app.strip().casefold())
        refused = _unsendable(channel, app, text)
        if refused is not None or channel is None:
            return refused or NO_CHANNEL.format(app=app)
        try:
            person = await channel.resolve(contact)
        except GuessedRecipientError as guess:
            # Asked again by the full name: a guess the channel cannot
            # reach is said now, not after the user said yes to it.
            try:
                person = await channel.resolve(guess.name)
            except NoRecipientError as why:
                return str(why)
        except NoRecipientError as why:
            return str(why)
        if person is None:
            return await _nobody(channel, contact)
        said_as = _SAID_AS.get(app.strip().casefold(), app.strip())
        return {"app": said_as, "contact": person.name, "text": text.strip()}

    prepared = replace(send_message, prepare=prepare)
    if not default:
        return prepared
    parameters = copy.deepcopy(dict(send_message.spec.parameters))
    parameters["properties"]["app"]["description"] = APP_DEFAULT.format(default=default)
    return replace(prepared, spec=replace(send_message.spec, parameters=parameters))


def _unsendable(channel: Channel[Any] | None, app: str, text: str) -> str | None:
    """Why nothing can be sent before anybody is looked up, or `None`."""
    if channel is None:
        return NO_CHANNEL.format(app=app)
    words = text.strip()
    if not words:
        return EMPTY
    if len(words) > MAX_TEXT_CHARS:
        return TOO_LONG.format(length=len(words), limit=MAX_TEXT_CHARS)
    return None


async def _nobody(channel: Channel[Any], contact: str) -> str:
    """No contact by that name, with the closest the channel knows."""
    near = await channel.closest(contact)
    closest = f"; closest names: {', '.join(near)}" if near else ""
    return NO_CONTACT.format(contact=contact, closest=closest)


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
            raise GuessedRecipientError(spoken, found.name)
        if not found.phone:
            raise NoRecipientError(NO_NUMBER.format(name=found.name))
        return found

    async def closest(self, spoken: str) -> list[str]:
        return self._book.closest(spoken)

    async def send(self, recipient: Contact, text: str) -> str:
        outcome = await self._whatsapp.send(recipient.phone, text)
        return WHATSAPP_SAID[outcome].format(
            name=recipient.name, seconds=WAKE_SECONDS, type_seconds=TYPE_SECONDS
        )
