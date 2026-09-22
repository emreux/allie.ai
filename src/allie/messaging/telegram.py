"""Telegram, as the user, through Telegram's own API (spec section 6,
decision T1 of 2026-09-14).

**Why the user's account and not a bot.** A bot cannot write first, and a
message from a bot is a message from a bot. Telegram's API terms allow a
third-party client on a user account - "obtain your own api_id" - and
Telethon is that client, asyncio-native and maintained (Pyrogram was
archived in December 2024). The `api_id` and `api_hash` are the user's,
from my.telegram.org (two minutes, and consistent with bringing one's own
key); the hash "is secret and cannot be revoked", so it never ships in the
repository and lives, with the session, in the Credential Manager. A
`StringSession` is full access to the account, which is why.

**This program sends; it does not read.** `receive_updates=False`: no
update loop, nothing listened to, nothing of the user's conversations
ever reaches the model.

**Two protocols, one class.** `Client` is the slice of Telethon the channel
uses at run time (connect, contacts, resolve, send); `LoginClient` is the
slice `allie telegram login` uses once. `TelethonClient` is both, and
imports `telethon` inside its methods, the way `media/now_playing.py`
imports WinRT: a dependency that is missing or broken on a machine must
not stop the assistant from starting. Everything Telethon raises is turned
into a `TelegramError` with a `kind`, so that the channel's answers are
sentences and never tracebacks.

**Finding the person.** The address book first (`contacts.py`): a contact
with a `telegram` username, or with a phone the account's contact list
knows; then the contact list by name, through the same matcher as apps and
the book (`store/names.py`, `shared="nobody"`); a miss fetches the list
once more, for a contact added today. The list is fetched on first use and
kept: it is one request, and the names in it do not change mid-session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from allie.messaging.contacts import AddressBook, Contact, phone_digits
from allie.store.names import NameIndex
from allie.tools.messaging import MATCHED, NoRecipientError

__all__ = [
    "HASH_ENTRY",
    "SESSION_ENTRY",
    "TEXT",
    "Client",
    "Login",
    "LoginClient",
    "LoginError",
    "PasswordNeededError",
    "Person",
    "Prompter",
    "Telegram",
    "TelegramError",
    "TelethonClient",
    "login",
]

# Where the secrets go in the Credential Manager (`config.store_api_key`):
# the application hash, and the session string `login` produces.
HASH_ENTRY = "telegram"
SESSION_ENTRY = "telegram-session"

# The last link of the chain of section 3.12 for what `allie telegram
# login` says and asks. English here; Turkish in `tr.toml`.
TEXT: dict[str, str] = {
    "telegram_api_id": (
        "Your Telegram api_id (a number). Create one at https://my.telegram.org under "
        "'API development tools'"
    ),
    "telegram_api_hash": "Your Telegram api_hash (nothing is shown as you type)",
    "telegram_phone": "The phone number of your Telegram account, with the country code (+90...)",
    "telegram_code": "The code Telegram just sent to your Telegram app (not by SMS)",
    "telegram_password": "Your Telegram two-step verification password (nothing is shown)",
    "telegram_logged_in": (
        "Logged in to Telegram as {name}. The session is in the Credential Manager."
    ),
    "telegram_login_failed": "Telegram login failed: {reason}",
}

# The channel's answers, addressed to the model.
SENT = "Sent to {name} on Telegram."
NOT_LOGGED_IN = "Telegram is not logged in: run 'allie telegram login'."
NO_SUCH_USER = "No Telegram user called {username!r}; check the username in contacts.toml."
SAID: dict[str, str] = {
    "flood_wait": "Telegram asked to wait {seconds} seconds before the next message.",
    "peer_flood": "Telegram is limiting this account's messages to people it does not know.",
    "privacy": "{name}'s privacy settings do not allow messages from this account.",
    "network": "Telegram could not be reached.",
    "not_authorized": NOT_LOGGED_IN,
    "rpc": "Telegram refused: {reason}",
}


@dataclass(frozen=True, slots=True)
class Person:
    """One Telegram user as this account knows them."""

    id: int
    name: str  # "first last", stripped; the username when both are empty
    username: str  # without "@", "" when none
    phone: str  # digits, "" unless the contact shares it
    # Telethon's own User, carried opaque so that `send` needs no second
    # lookup. Left out of equality and hashing: it is not a fact about the
    # person, and Telethon's objects are not reliably hashable.
    entity: object = field(default=None, compare=False)


class TelegramError(Exception):
    """Telegram, or the network, would not do it. `kind` names why in one
    word; `seconds` and `reason` fill the sentence for that kind."""

    def __init__(self, kind: str, *, seconds: int = 0, reason: str = "") -> None:
        super().__init__(kind)
        self.kind = kind
        self.seconds = seconds
        self.reason = reason


class Client(Protocol):
    """The slice of Telethon the channel uses. The tests' fake is the other."""

    async def connect(self) -> None: ...

    async def authorized(self) -> bool: ...

    async def contacts(self) -> list[Person]: ...

    async def resolve(self, username: str) -> Person | None: ...

    async def send(self, person: Person, text: str) -> None: ...

    async def disconnect(self) -> None: ...


class Telegram:
    """Telegram over the address book and the account's own contact list."""

    def __init__(self, book: AddressBook, *, client: Client | None) -> None:
        self._book = book
        # `None` means not logged in: `resolve` says so, and nothing is
        # ever connected.
        self._client = client
        self._connected = False
        self._people: NameIndex[Person] | None = None
        self._listed: list[Person] = []

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def resolve(self, spoken: str) -> Person | None:
        """The Telegram user the user meant by `spoken`, or `None`."""
        client = await self._ready()
        contact = self._book.find(spoken)
        if contact is not None and not self._book.certain(spoken, contact):
            raise NoRecipientError(MATCHED.format(contact=spoken, name=contact.name))
        found = await self._from_book(client, contact) if contact is not None else None
        if found is None:
            found = await self._from_list(client, contact.name if contact else spoken)
        return found

    async def closest(self, spoken: str) -> list[str]:
        """The book's nearest names and the list's, the book's first."""
        near = self._book.closest(spoken)
        if self._people is not None:
            near.extend(name for name in self._people.closest(spoken) if name not in near)
        return near

    async def send(self, person: Person, text: str) -> str:
        client = await self._ready()
        try:
            await client.send(person, text)
        except TelegramError as failure:
            return _said(failure, name=person.name)
        return SENT.format(name=person.name)

    async def close(self) -> None:
        """Disconnects, once, at shutdown."""
        if self._client is not None and self._connected:
            self._connected = False
            await self._client.disconnect()

    async def _ready(self) -> Client:
        """The client, connected and logged in - or the sentence that it is not."""
        if self._client is None:
            raise NoRecipientError(NOT_LOGGED_IN)
        if not self._connected:
            try:
                await self._client.connect()
                if not await self._client.authorized():
                    raise NoRecipientError(NOT_LOGGED_IN)
            except TelegramError as failure:
                raise NoRecipientError(_said(failure)) from failure
            self._connected = True
        return self._client

    async def _from_book(self, client: Client, contact: Contact) -> Person | None:
        """The list's user with the contact's username or phone; a username
        the list lacks is asked of Telegram itself."""
        people = await self._list(client)
        if contact.telegram:
            wanted = contact.telegram.casefold()
            for person in people:
                if person.username.casefold() == wanted:
                    return person
            try:
                found = await client.resolve(contact.telegram)
            except TelegramError as failure:
                raise NoRecipientError(_said(failure)) from failure
            if found is None:
                raise NoRecipientError(NO_SUCH_USER.format(username=contact.telegram))
            return found
        if contact.phone:
            for person in people:
                if person.phone == contact.phone:
                    return person
        return None

    async def _from_list(self, client: Client, name: str) -> Person | None:
        people = await self._index(client)
        found = people.find(name)
        if found is None:
            # A contact added today: one more fetch before giving up.
            self._people = None
            people = await self._index(client, again=True)
            found = people.find(name)
        if found is not None and not people.certain(name, found):
            raise NoRecipientError(MATCHED.format(contact=name, name=found.name))
        return found

    async def _list(self, client: Client, *, again: bool = False) -> list[Person]:
        if again or not self._listed:
            try:
                self._listed = await client.contacts()
            except TelegramError as failure:
                raise NoRecipientError(_said(failure)) from failure
            logger.debug("telegram: {} contacts listed", len(self._listed))
        return self._listed

    async def _index(self, client: Client, *, again: bool = False) -> NameIndex[Person]:
        if self._people is None:
            people: NameIndex[Person] = NameIndex(shared="nobody")
            for person in await self._list(client, again=again):
                people.add(
                    person, person.name, aliases=(person.username,) if person.username else ()
                )
            self._people = people
        return self._people


def _said(failure: TelegramError, *, name: str = "") -> str:
    sentence = SAID.get(failure.kind, SAID["rpc"])
    return sentence.format(
        seconds=failure.seconds, name=name, reason=failure.reason or failure.kind
    )


# --------------------------------------------------------------------------
# Logging in: `allie telegram login`
# --------------------------------------------------------------------------


class PasswordNeededError(Exception):
    """The account has two-step verification: the code was right and a
    password is needed as well."""


class LoginError(Exception):
    """Telegram refused the login; the message is Telegram's own reason."""


class LoginClient(Protocol):
    """The slice of Telethon `login` uses, once."""

    async def connect(self) -> None: ...

    async def send_code(self, phone: str) -> str:
        """Asks Telegram to send the code; returns the hash the sign-in needs."""
        ...

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        """Signs in; returns the account's name. Raises `PasswordNeededError`."""
        ...

    async def sign_in_with_password(self, password: str) -> str: ...

    def session_string(self) -> str: ...

    async def disconnect(self) -> None: ...


class Prompter(Protocol):
    """What `login` needs from a terminal: `setup_wizard.Prompter`'s two
    questions, named by key, worded by the pack."""

    async def ask(self, key: str) -> str | None: ...

    async def secret(self, key: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class Login:
    """What a login produced: the two things to store, and the name to say."""

    api_id: int
    api_hash: str
    session: str
    name: str


LoginClientFactory = Callable[[int, str], LoginClient]


async def login(
    prompter: Prompter,
    *,
    api_id: int = 0,
    api_hash: str = "",
    client_factory: LoginClientFactory | None = None,
) -> Login | None:
    """Asks what Telegram needs, in order, and signs in once.

    The `api_id` and `api_hash` are asked only when not already stored;
    then the phone, the code (it arrives in the user's Telegram app, not
    by SMS), and the password only when the account turns out to have one.
    `None` when the user walked away from a question; `LoginError` with
    Telegram's own words when it refused.
    """
    factory = client_factory if client_factory is not None else _login_client
    if not api_id:
        answer = await prompter.ask("telegram_api_id")
        if answer is None:
            return None
        try:
            api_id = int(answer.strip())
        except ValueError as failure:
            raise LoginError(f"api_id must be a number, not {answer.strip()!r}") from failure
    if not api_hash:
        answer = await prompter.secret("telegram_api_hash")
        if answer is None:
            return None
        api_hash = answer.strip()

    phone = await prompter.ask("telegram_phone")
    if phone is None:
        return None

    client = factory(api_id, api_hash)
    await client.connect()
    try:
        code_hash = await client.send_code(phone.strip())
        code = await prompter.ask("telegram_code")
        if code is None:
            return None
        try:
            name = await client.sign_in(phone.strip(), code.strip(), code_hash=code_hash)
        except PasswordNeededError:
            password = await prompter.secret("telegram_password")
            if password is None:
                return None
            name = await client.sign_in_with_password(password)
        return Login(api_id=api_id, api_hash=api_hash, session=client.session_string(), name=name)
    finally:
        await client.disconnect()


def _login_client(api_id: int, api_hash: str) -> LoginClient:
    return TelethonClient(api_id, api_hash, "")


# --------------------------------------------------------------------------
# The real client
# --------------------------------------------------------------------------


class TelethonClient:
    """Telethon behind both protocols. `telethon` is imported inside the
    methods (see the module docstring); its objects are `Any` and are
    turned into `Person`s before anything else sees them."""

    def __init__(self, api_id: int, api_hash: str, session: str) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session = session
        self._client: Any = None

    async def connect(self) -> None:
        from telethon import TelegramClient
        from telethon.sessions import StringSession

        self._client = TelegramClient(
            StringSession(self._session or None),
            self._api_id,
            self._api_hash,
            receive_updates=False,
        )
        try:
            await self._client.connect()
        except OSError as failure:
            raise TelegramError("network", reason=str(failure)) from failure

    async def authorized(self) -> bool:
        return bool(await self._client.is_user_authorized())

    async def contacts(self) -> list[Person]:
        from telethon.tl.functions.contacts import GetContactsRequest

        listing = await self._call(self._client(GetContactsRequest(hash=0)))
        found = (_person(user) for user in getattr(listing, "users", None) or [])
        return [person for person in found if person is not None]

    async def resolve(self, username: str) -> Person | None:
        from telethon import errors

        try:
            entity = await self._client.get_entity(username)
        except (ValueError, errors.UsernameNotOccupiedError, errors.UsernameInvalidError):
            return None
        except errors.FloodWaitError as failure:
            raise TelegramError("flood_wait", seconds=int(failure.seconds)) from failure
        except errors.RPCError as failure:
            raise TelegramError("rpc", reason=str(failure)) from failure
        except OSError as failure:
            raise TelegramError("network", reason=str(failure)) from failure
        return _person(entity)

    async def send(self, person: Person, text: str) -> None:
        await self._call(self._client.send_message(person.entity, text))

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()

    # --- login ------------------------------------------------------------

    async def send_code(self, phone: str) -> str:
        sent = await self._login_call(self._client.send_code_request(phone))
        return str(sent.phone_code_hash)

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        from telethon import errors

        try:
            user = await self._client.sign_in(phone=phone, code=code, phone_code_hash=code_hash)
        except errors.SessionPasswordNeededError as needed:
            raise PasswordNeededError from needed
        except errors.RPCError as failure:
            raise LoginError(str(failure)) from failure
        return _display_name(user)

    async def sign_in_with_password(self, password: str) -> str:
        user = await self._login_call(self._client.sign_in(password=password))
        return _display_name(user)

    def session_string(self) -> str:
        return str(self._client.session.save())

    # --- the error map ----------------------------------------------------

    async def _call(self, request: Any) -> Any:
        from telethon import errors

        try:
            return await request
        except errors.FloodWaitError as failure:
            raise TelegramError("flood_wait", seconds=int(failure.seconds)) from failure
        except errors.PeerFloodError as failure:
            raise TelegramError("peer_flood") from failure
        except errors.UserPrivacyRestrictedError as failure:
            raise TelegramError("privacy") from failure
        except errors.UnauthorizedError as failure:
            raise TelegramError("not_authorized") from failure
        except errors.RPCError as failure:
            raise TelegramError("rpc", reason=str(failure)) from failure
        except OSError as failure:
            raise TelegramError("network", reason=str(failure)) from failure

    async def _login_call(self, request: Any) -> Any:
        from telethon import errors

        try:
            return await request
        except errors.RPCError as failure:
            raise LoginError(str(failure)) from failure
        except OSError as failure:
            raise LoginError(f"Telegram could not be reached: {failure}") from failure


def _person(user: Any) -> Person | None:
    """A `Person` from Telethon's `User`; `None` for a deleted account or a bot."""
    if user is None or getattr(user, "deleted", False) or getattr(user, "bot", False):
        return None
    username = str(getattr(user, "username", None) or "")
    name = _display_name(user) or username
    if not name:
        return None
    phone = str(getattr(user, "phone", None) or "")
    try:
        digits = phone_digits(phone) if phone else ""
    except ValueError:
        digits = ""
    return Person(
        id=int(getattr(user, "id", 0)), name=name, username=username, phone=digits, entity=user
    )


def _display_name(user: Any) -> str:
    first = str(getattr(user, "first_name", None) or "")
    last = str(getattr(user, "last_name", None) or "")
    return " ".join(part for part in (first, last) if part).strip()
