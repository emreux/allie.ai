"""The people the user can message, from `contacts.toml` (spec section 4.2).

**Why a file the user writes.** WhatsApp on Windows exposes no contact list
to another program, and the number a chat link needs has to come from
somewhere the user controls. `%APPDATA%\\allie\\contacts.toml` sits
beside `config.toml` and `memory.toml`, under the same rules (section 10):
data, editable by hand, following the user through a roaming profile,
never in the repository. Telegram adds the account's own contact list on
top (`telegram.py`); this file is what both channels start from.

**The worst failure is a message to the wrong person, so the file is
refused rather than repaired.** A number typed without its country code is
not guessed at - the country is not knowable from here (section 3.12) -
and is found before the first send, not after it. A key nobody meant
(`phonee`) would otherwise drop the number silently and the person would
be "found" with nothing to send to. Two entries that answer to the same
name would give the message to whichever came first. Each of these stops
the assistant with a sentence naming the file and the field, the way a
`memory.toml` that does not parse does (`MemoryFileError`).

The matching is the app catalogue's (`store/names.py`), built with
`shared="nobody"`: a first name two people share names neither of them,
and the tool is handed both names to ask about.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from allie.config import config_dir
from allie.store.names import NameIndex
from allie.store.normalize import normalize_search

__all__ = [
    "CONTACTS_FILE_NAME",
    "AddressBook",
    "Contact",
    "ContactsFileError",
    "contacts_path",
    "phone_digits",
]

CONTACTS_FILE_NAME = "contacts.toml"

# E.164: a country code and a subscriber number, 8 to 15 digits together.
MIN_DIGITS = 8
MAX_DIGITS = 15

# What a person may write around the digits, and nothing else.
_DECORATION = re.compile(r"[\s\-.()]")
_KNOWN_KEYS = frozenset({"name", "aliases", "phone", "telegram"})
_USERNAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class ContactsFileError(RuntimeError):
    """`contacts.toml` could not be trusted: not TOML, a field of the wrong
    shape, a number that cannot be dialled, two people who answer to one
    name. Fixable by the user, so named for `run` and never written over."""


def contacts_path() -> Path:
    """`%APPDATA%\\allie\\contacts.toml`, beside `config.toml`."""
    return config_dir() / CONTACTS_FILE_NAME


def phone_digits(text: str) -> str:
    """`"+90 (532) 000-00-00"` → `"905320000000"`: what a chat link takes.

    Spaces, dashes, dots and brackets are decoration; a leading `+` or `00`
    is the international prefix and comes off. What is left must be 8 to
    15 digits and must not start with `0` - that is a national number, and
    which country it is national to cannot be known from here.
    """
    written = text.strip()
    if not written:
        raise ValueError("the phone number is empty")
    bare = _DECORATION.sub("", written)
    if bare.startswith("+"):
        bare = bare[1:]
    elif bare.startswith("00"):
        bare = bare[2:]
    if not bare.isdigit():
        raise ValueError(
            f"{written!r} is not a phone number: only digits, spaces, dashes, dots and "
            "brackets are allowed, with + or 00 in front"
        )
    if bare.startswith("0"):
        raise ValueError(
            f"{written!r} starts with 0, which makes it a national number; write it with "
            "the country code, as in +90 532 000 00 00"
        )
    if not MIN_DIGITS <= len(bare) <= MAX_DIGITS:
        raise ValueError(
            f"{written!r} has {len(bare)} digits; an international number has "
            f"{MIN_DIGITS} to {MAX_DIGITS} digits"
        )
    return bare


@dataclass(frozen=True, slots=True)
class Contact:
    """One person as the user wrote them down."""

    name: str
    aliases: tuple[str, ...] = ()
    # Digits only, country code first, no plus: "905320000000". Empty when
    # the person has no WhatsApp number in the file.
    phone: str = ""
    # The Telegram username without the "@". Empty means "look the person
    # up on Telegram by name or phone" (`telegram.py`).
    telegram: str = ""


class AddressBook:
    """The people in `contacts.toml`, found by whatever the user calls them."""

    def __init__(self, contacts: Iterable[Contact] = ()) -> None:
        self._contacts = list(contacts)
        self._index: NameIndex[Contact] = NameIndex(shared="nobody")
        for contact in self._contacts:
            self._index.add(contact, contact.name, aliases=contact.aliases)

    @classmethod
    def load(cls, path: Path | None = None) -> AddressBook:
        """Reads the file; no file is an empty book and no error."""
        target = path if path is not None else contacts_path()
        if not target.is_file():
            return cls()

        try:
            data = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as failure:
            raise ContactsFileError(f"{target} could not be read: {failure}") from failure

        entries = data.get("contact", [])
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ContactsFileError(
                f"{target}: 'contact' must be a list of [[contact]] tables, one per person"
            )

        contacts = [_contact(target, number, entry) for number, entry in enumerate(entries, 1)]
        _refuse_shared_names(target, contacts)
        return cls(contacts)

    def __len__(self) -> int:
        return len(self._contacts)

    def names(self) -> list[str]:
        """Every name, then every alias, each once, in file order: what the
        recogniser is told to expect (spec A6)."""
        seen: dict[str, None] = {}
        for contact in self._contacts:
            seen.setdefault(contact.name, None)
        for contact in self._contacts:
            for alias in contact.aliases:
                seen.setdefault(alias, None)
        return list(seen)

    def find(self, spoken: str) -> Contact | None:
        """The person the user meant by `spoken`, or `None`."""
        return self._index.find(spoken)

    def certain(self, spoken: str, contact: Contact) -> bool:
        """Whether `spoken` is `contact`'s listed name, an alias or a whole
        word of one - not a prefix, not a guess."""
        return self._index.certain(spoken, contact)

    def closest(self, spoken: str, *, limit: int = 3) -> list[str]:
        """Names near `spoken`, for the model to ask about."""
        return self._index.closest(spoken, limit=limit)


def _contact(path: Path, number: int, entry: dict[str, object]) -> Contact:
    where = f"{path}: contact {number}"
    unknown = sorted(set(entry) - _KNOWN_KEYS)
    if unknown:
        raise ContactsFileError(
            f"{where} has a key this program does not read: {', '.join(unknown)}; "
            f"the keys are {', '.join(sorted(_KNOWN_KEYS))}"
        )

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ContactsFileError(f"{where} needs a name")

    aliases = entry.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise ContactsFileError(f"{where} ({name}): aliases must be a list of names")

    phone = entry.get("phone", "")
    if not isinstance(phone, str):
        raise ContactsFileError(f"{where} ({name}): phone must be written in quotes")
    digits = ""
    if phone.strip():
        try:
            digits = phone_digits(phone)
        except ValueError as failure:
            raise ContactsFileError(f"{where} ({name}): phone: {failure}") from failure

    telegram = entry.get("telegram", "")
    if not isinstance(telegram, str) or (telegram and not _USERNAME.match(telegram)):
        raise ContactsFileError(
            f"{where} ({name}): telegram must be the username without the @, "
            "letters, digits and underscores only"
        )

    return Contact(
        name=name.strip(),
        aliases=tuple(alias.strip() for alias in aliases),
        phone=digits,
        telegram=telegram,
    )


def _refuse_shared_names(path: Path, contacts: list[Contact]) -> None:
    """Two entries that fold to one name or alias would give the message to
    whichever came first, every time."""
    claimed: dict[str, str] = {}
    for contact in contacts:
        for called in (contact.name, *contact.aliases):
            key = normalize_search(called).strip()
            holder = claimed.setdefault(key, contact.name)
            if holder != contact.name:
                raise ContactsFileError(
                    f"{path}: {holder!r} and {contact.name!r} both answer to {key!r}; "
                    "give one of them a different name or alias"
                )
