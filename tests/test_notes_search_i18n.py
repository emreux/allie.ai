"""The search folds every script by one rule (design.md section 3.7, 3.12;
CLAUDE.md invariant 4).

`anyascii` transliterates rather than decomposes, which is why the same
test passes for Polish, German, Greek and Cyrillic without a table per
language: `Łódź` is `lodz`, `STRASSE` and `Straße` are both `strasse`,
`Ελλάδα` is `ellada`, `Москва` is `moskva`. NFKD would leave `ł` and `ß`
as they are and the first two would fail (`store/normalize.py`).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from assistant.store.db import open_database
from assistant.store.repos import NotesRepo


@pytest.fixture
def notes() -> Iterator[NotesRepo]:
    connection: sqlite3.Connection = open_database(":memory:")
    try:
        yield NotesRepo(connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("kept", "spellings"),
    [
        ("Bilet: Łódź'e pazartesi, peron 3.", ["Łódź", "lodz", "LODZ", "Lódz"]),
        ("Adres: Hauptstraße 5, Berlin.", ["Hauptstraße", "hauptstrasse", "HAUPTSTRASSE"]),
        ("Yaz tatili: Ελλάδα, Rodos.", ["Ελλάδα", "ellada", "ELLADA"]),
        ("Toplantı Москва ofisiyle saat 15'te.", ["Москва", "moskva", "MOSKVA"]),
        ("Kahve: Đà Nẵng'dan getirilecek.", ["Đà Nẵng", "da nang", "DA NANG"]),
    ],
)
def test_a_note_is_found_in_its_own_script_and_in_plain_letters(
    notes: NotesRepo, kept: str, spellings: list[str]
) -> None:
    note = notes.add(kept)
    for other in ("Işık faturası", "Kira ödemesi", "Ahmet'i ara"):
        notes.add(other)

    for spelling in spellings:
        assert [found.id for found in notes.search(spelling)] == [note.id], spelling


def test_the_folding_does_not_mix_two_languages_up(notes: NotesRepo) -> None:
    """Folding removes accents and case, never letters: `Łódź` is not `Lodi`
    and `Straße` is not `Strasse` with a letter missing."""
    lodz = notes.add("Łódź")
    notes.add("Lodi")

    assert [found.id for found in notes.search("lodz")] == [lodz.id]
    assert [found.text for found in notes.search("lodi")] == ["Lodi"]
