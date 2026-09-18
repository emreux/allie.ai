"""Notes are found in whatever spelling they are asked for (design.md
section 3.7, phase 4.1; CLAUDE.md testing notes).

The claim of `store/normalize.py`, made against the real table: `Işık`,
`IŞIK`, `isik` and `ışık` are one word to the search, `ödeme` and `ODEME`
another, and a word finds the longer word it is part of. The index is
FTS5 with trigram tokens over the folded column, and `bm25` ranks.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from assistant.store.db import open_database
from assistant.store.repos import MIN_QUERY_CHARS, Note, NotesRepo, query_terms


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def notes(database: sqlite3.Connection) -> NotesRepo:
    moments = iter(range(1_700_000_000, 1_700_000_100))
    return NotesRepo(database, clock=lambda: next(moments))


@pytest.fixture
def kept(notes: NotesRepo) -> list[Note]:
    return [
        notes.add(text)
        for text in (
            "Işık faturası ödendi, makbuz çekmecede.",
            "Yarın ödeme günü, kira için para ayır.",
            "Ahmet'e doğum günü hediyesi: kitap.",
            "ISIK Sokak 12 numara, kapı kodu 4471.",
        )
    ]


def texts(found: list[Note]) -> list[str]:
    return [note.text for note in found]


# --------------------------------------------------------------------------
# One word, every spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["ışık", "Işık", "IŞIK", "isik", "ISIK", "İşık"])
def test_every_spelling_of_isik_finds_both_notes(
    notes: NotesRepo, kept: list[Note], spelling: str
) -> None:
    found = texts(notes.search(spelling))

    assert kept[0].text in found
    assert kept[3].text in found
    assert len(found) == 2


@pytest.mark.parametrize("spelling", ["ödeme", "odeme", "ÖDEME", "ODEME", "Ödeme"])
def test_every_spelling_of_odeme_finds_the_note_that_says_it(
    notes: NotesRepo, kept: list[Note], spelling: str
) -> None:
    found = texts(notes.search(spelling))

    # "ödendi" is not "ödeme": a substring search, not a stem search.
    assert found == [kept[1].text]


def test_a_word_finds_the_longer_word_it_is_part_of(notes: NotesRepo, kept: list[Note]) -> None:
    """Trigram tokens: "fatura" is inside "faturası", and a search is a
    substring, not a whole word."""
    assert texts(notes.search("fatura")) == [kept[0].text]
    assert texts(notes.search("hediye")) == [kept[2].text]


def test_every_word_of_the_query_is_required(notes: NotesRepo, kept: list[Note]) -> None:
    assert texts(notes.search("ışık makbuz")) == [kept[0].text]
    assert texts(notes.search("ışık kapı")) == [kept[3].text]
    assert notes.search("ışık hediye") == []


def test_a_note_that_says_it_more_ranks_first(notes: NotesRepo) -> None:
    first = notes.add("Kira kira kira: ayın birinde kira ödenecek.")
    notes.add("Kira sözleşmesi dosyada.")

    assert texts(notes.search("kira"))[0] == first.text


def test_the_note_reads_back_as_it_was_written(notes: NotesRepo, kept: list[Note]) -> None:
    """Folding is for the index only; what is stored and read back is the
    user's own spelling."""
    assert notes.get(kept[0].id) == kept[0]
    assert kept[0].text.startswith("Işık")


# --------------------------------------------------------------------------
# What a query has to be
# --------------------------------------------------------------------------


@pytest.mark.parametrize("short", ["ış", "ve", "a b", "", "  ", "12"])
def test_a_query_with_no_word_of_three_letters_is_refused(notes: NotesRepo, short: str) -> None:
    with pytest.raises(ValueError, match=str(MIN_QUERY_CHARS)):
        notes.search(short)


def test_short_words_are_dropped_and_the_rest_searched(notes: NotesRepo, kept: list[Note]) -> None:
    assert texts(notes.search("ve ışık da")) == texts(notes.search("ışık"))


def test_the_match_expression_is_folded_quoted_words_joined_by_and() -> None:
    assert query_terms("Işık faturası, ve") == '"isik" AND "faturasi"'
    assert query_terms("ve") == ""


def test_punctuation_in_the_query_cannot_break_the_expression(notes: NotesRepo) -> None:
    """Only the words reach FTS5; a quote or an operator typed by the user
    - or heard by the recogniser - is not syntax."""
    notes.add('Tırnak "içinde" not')

    assert query_terms('içinde" OR NOT x*') == '"icinde" AND "not"'
    assert texts(notes.search('"içinde"')) == ['Tırnak "içinde" not']


# --------------------------------------------------------------------------
# The rest of the table
# --------------------------------------------------------------------------


def test_the_latest_notes_come_newest_first(notes: NotesRepo, kept: list[Note]) -> None:
    assert texts(notes.latest(2)) == [kept[3].text, kept[2].text]
    assert notes.count() == 4


def test_a_deleted_note_is_gone_from_the_search_too(notes: NotesRepo, kept: list[Note]) -> None:
    assert notes.delete(kept[0].id)

    assert notes.get(kept[0].id) is None
    assert texts(notes.search("fatura")) == []
    assert not notes.delete(kept[0].id)
    assert notes.count() == 3


def test_times_are_utc_epoch_seconds(notes: NotesRepo) -> None:
    note = notes.add("x")

    assert note.created_at == 1_700_000_000
