"""`messaging/contacts.py` (15 Sep 2026): the address book in `contacts.toml`.

The worst failure of messaging is a message to the wrong person, so the
file is refused - with the field named - for anything that could send one:
a phone that cannot be read, a key nobody meant, two entries that answer to
the same name. Nothing here reads `%APPDATA%`; every book is written into
`tmp_path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.messaging.contacts import (
    CONTACTS_FILE_NAME,
    AddressBook,
    Contact,
    ContactsFileError,
    phone_digits,
)

BOOK = """\
# One [[contact]] per person.
[[contact]]
name = "Ahmet Yılmaz"
aliases = ["Ahmet", "abi"]
phone = "+90 532 000 00 00"
telegram = "ahmetyilmaz"

[[contact]]
name = "Şükrü Kaya"
phone = "0090 (533) 111-22-33"

[[contact]]
name = "Łukasz Nowak"
telegram = "lukasz"
"""


def book(tmp_path: Path, text: str = BOOK) -> AddressBook:
    path = tmp_path / CONTACTS_FILE_NAME
    path.write_text(text, encoding="utf-8")
    return AddressBook.load(path)


# --------------------------------------------------------------------------
# phone_digits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "digits"),
    [
        ("+90 532 000 00 00", "905320000000"),
        ("0090 532 000 00 00", "905320000000"),
        ("+90 (532) 000-00-00", "905320000000"),
        ("+1 415.555.0100", "14155550100"),
        ("447700900123", "447700900123"),
    ],
)
def test_the_forms_people_write_a_number_in_all_come_out_as_digits(
    written: str, digits: str
) -> None:
    assert phone_digits(written) == digits


@pytest.mark.parametrize(
    ("written", "reason"),
    [
        ("0532 000 00 00", "country code"),  # national: the country cannot be guessed
        ("+90 532", "8 to 15 digits"),
        ("+90 532 000 00 00 123 45", "8 to 15 digits"),
        ("+90 532 ABC", "only digits"),
        ("", "empty"),
    ],
)
def test_a_number_that_cannot_be_dialled_is_refused_with_the_reason(
    written: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        phone_digits(written)


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------


def test_the_file_is_read_into_contacts_with_digits_only_phones(tmp_path: Path) -> None:
    loaded = book(tmp_path)

    assert len(loaded) == 3
    assert loaded.find("Ahmet Yılmaz") == Contact(
        name="Ahmet Yılmaz",
        aliases=("Ahmet", "abi"),
        phone="905320000000",
        telegram="ahmetyilmaz",
    )
    assert loaded.find("Łukasz Nowak") == Contact(name="Łukasz Nowak", telegram="lukasz")


def test_no_file_is_an_empty_book_and_no_error(tmp_path: Path) -> None:
    loaded = AddressBook.load(tmp_path / CONTACTS_FILE_NAME)

    assert len(loaded) == 0
    assert loaded.find("Ahmet") is None
    assert loaded.names() == []


def test_the_default_path_is_beside_the_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIVE_ASSISTANT_CONFIG_DIR", str(tmp_path))
    (tmp_path / CONTACTS_FILE_NAME).write_text('[[contact]]\nname = "Ada"\n', encoding="utf-8")

    assert AddressBook.load().names() == ["Ada"]


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("[[contact]\nname = 'x'\n", "could not be read"),
        ('[[contact]]\nphone = "+90 532 000 00 00"\n', "name"),
        ('[[contact]]\nname = "Ahmet"\nphonee = "+90 532 000 00 00"\n', "phonee"),
        ('[[contact]]\nname = "Ahmet"\nphone = "0532 000 00 00"\n', "phone"),
        ('[[contact]]\nname = "Ahmet"\naliases = "abi"\n', "aliases"),
        ('[[contact]]\nname = "Ahmet"\ntelegram = "@ahmet"\n', "telegram"),
        ('[[contact]]\nname = ""\n', "name"),
        ("contact = 5\n", "contact"),
    ],
)
def test_a_file_that_could_send_to_the_wrong_person_stops_the_assistant(
    tmp_path: Path, text: str, names: str
) -> None:
    """A misspelt `phonee` would otherwise drop the number silently; a
    national number would be dialled in the wrong country."""
    with pytest.raises(ContactsFileError, match=names):
        book(tmp_path, text)


def test_two_entries_that_answer_to_the_same_name_are_refused(tmp_path: Path) -> None:
    """Whichever came first would get the message, every time."""
    text = (
        '[[contact]]\nname = "Ahmet Yılmaz"\naliases = ["abi"]\n\n'
        '[[contact]]\nname = "Mehmet"\naliases = ["ABİ"]\n'
    )

    with pytest.raises(ContactsFileError, match="abi"):
        book(tmp_path, text)


def test_the_same_name_folded_two_ways_is_the_same_name(tmp_path: Path) -> None:
    text = '[[contact]]\nname = "Şükrü"\n\n[[contact]]\nname = "sukru"\n'

    with pytest.raises(ContactsFileError, match="sukru"):
        book(tmp_path, text)


def test_the_error_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ContactsFileError, match=CONTACTS_FILE_NAME):
        book(tmp_path, "[[contact]]\nname = 5\n")


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    [
        "Ahmet",  # an alias
        "ahmede",  # the recogniser's case ending
        "abi",  # an alias that is one word
        "yılmaz",  # one word of the name
        "Ahmet Yılmaz",  # the whole name
        "AHMET YILMAZ",
    ],
)
def test_a_person_is_found_the_way_an_app_is(tmp_path: Path, spoken: str) -> None:
    found = book(tmp_path).find(spoken)

    assert found is not None and found.name == "Ahmet Yılmaz"


def test_accents_are_folded_on_both_sides(tmp_path: Path) -> None:
    loaded = book(tmp_path)

    assert loaded.find("sukru").name == "Şükrü Kaya"  # type: ignore[union-attr]
    assert loaded.find("lukasz nowak").name == "Łukasz Nowak"  # type: ignore[union-attr]


def test_two_people_called_ahmet_are_nobody_and_both_are_offered(tmp_path: Path) -> None:
    """No alias "Ahmet" on either: the first name alone is a word of both,
    and a word that two people share names neither of them."""
    text = '[[contact]]\nname = "Ahmet Yılmaz"\n\n[[contact]]\nname = "Ahmet Kaya"\n'
    loaded = book(tmp_path, text)

    assert loaded.find("Ahmet") is None
    assert loaded.closest("Ahmet") == ["Ahmet Kaya", "Ahmet Yılmaz"]


def test_a_name_not_in_the_book_is_not_the_nearest_person(tmp_path: Path) -> None:
    """ "Mehmet" is not Ahmet, however alike the words are (2026-09-15)."""
    assert book(tmp_path).find("Mehmet") is None


def test_certain_tells_a_listed_name_from_a_guess(tmp_path: Path) -> None:
    loaded = book(tmp_path)
    ahmet = loaded.find("Ahmet Yılmaz")
    assert ahmet is not None

    assert loaded.certain("Ahmet", ahmet)
    assert loaded.certain("abi", ahmet)
    assert not loaded.certain("ahmede", ahmet)


def test_closest_names_the_nearest_people(tmp_path: Path) -> None:
    assert book(tmp_path).closest("ahmet kaya") == ["Ahmet Yılmaz", "Şükrü Kaya"]


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------


def test_names_are_every_name_then_every_alias_each_once_in_file_order(tmp_path: Path) -> None:
    """What the recogniser is told to expect, people first (spec A6)."""
    text = (
        '[[contact]]\nname = "Ahmet Yılmaz"\naliases = ["Ahmet", "abi"]\n\n'
        '[[contact]]\nname = "Ahmet Kaya"\naliases = ["Kaya"]\n'
    )

    assert book(tmp_path, text).names() == ["Ahmet Yılmaz", "Ahmet Kaya", "Ahmet", "abi", "Kaya"]
