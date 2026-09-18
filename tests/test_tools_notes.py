"""`add_note`, `search_notes`, `delete_note` (design.md phase 4.1; 17 Sep 2026).

The first two over the real table in memory; the third through the real
gate of `agent/policy.py`, because the claim about it is the gate's: a
note is not deleted without a yes, and the yes is to the note's own
words - a text the model paraphrased is refused before anything is
deleted, whatever the user answered.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from assistant.agent.core import Confirm
from assistant.agent.policy import DECLINED, dispatch
from assistant.live.base import ToolCall
from assistant.store.db import open_database
from assistant.store.repos import NotesRepo
from assistant.tools.notes import (
    EMPTY,
    MAX_NOTE_CHARS,
    NONE_STORED,
    TEXT,
    add_note_for,
    delete_note_for,
    search_notes_for,
)
from assistant.tools.registry import Tool, ToolRegistry


class FakeConfirm:
    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def __call__(self, question: str) -> bool:
        self.asked.append(question)
        return self.answer


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def notes(database: sqlite3.Connection) -> NotesRepo:
    return NotesRepo(database, clock=lambda: 1_758_000_000)


@pytest.fixture
def add_note(notes: NotesRepo) -> Tool:
    return add_note_for(notes)


@pytest.fixture
def search_notes(notes: NotesRepo) -> Tool:
    return search_notes_for(notes)


@pytest.fixture
def delete_note(notes: NotesRepo) -> Tool:
    return delete_note_for(notes, confirm_prompt="'{text}' notu silinecek.")


async def through_the_gate(tool: Tool, confirm: Confirm, **arguments: object) -> str:
    call = ToolCall(id="c1", name=tool.spec.name, arguments=dict(arguments))
    return await dispatch(call, turn_id="t1", registry=ToolRegistry([tool]), confirm=confirm)


# --------------------------------------------------------------------------
# add_note
# --------------------------------------------------------------------------


def test_add_and_search_are_safe_and_delete_asks(
    add_note: Tool, search_notes: Tool, delete_note: Tool
) -> None:
    assert (add_note.risk, search_notes.risk, delete_note.risk) == ("safe", "safe", "confirm")
    assert add_note.spec.parameters["required"] == ["text"]
    assert search_notes.spec.parameters["required"] == []
    assert delete_note.spec.parameters["required"] == ["note_id", "text"]
    assert delete_note.confirm_prompt == "'{text}' notu silinecek."
    assert TEXT["note_delete_confirm"] == "The note '{text}' will be deleted."


async def test_a_note_is_kept_as_said_and_numbered(add_note: Tool, notes: NotesRepo) -> None:
    said = await add_note.run(text="  Süt  al,\nekmek al. ")

    assert said == "Kept note #1: 'Süt al, ekmek al.'. 1 notes are stored."
    assert notes.get(1) is not None and notes.get(1).text == "Süt al, ekmek al."


async def test_an_empty_note_is_not_kept(add_note: Tool, notes: NotesRepo) -> None:
    assert await add_note.run(text="   ") == EMPTY
    assert notes.count() == 0


async def test_a_note_too_long_to_be_a_note_is_refused(add_note: Tool, notes: NotesRepo) -> None:
    said = await add_note.run(text="x" * (MAX_NOTE_CHARS + 1))

    assert said.startswith("Too long")
    assert str(MAX_NOTE_CHARS) in said
    assert notes.count() == 0


# --------------------------------------------------------------------------
# search_notes
# --------------------------------------------------------------------------


async def test_matching_notes_are_listed_with_their_numbers(
    add_note: Tool, search_notes: Tool
) -> None:
    await add_note.run(text="Işık faturası ödendi.")
    await add_note.run(text="Kira yarın.")

    said = await search_notes.run(query="isik")

    assert said.startswith("Notes matching 'isik', best first:\n#1 (")
    assert said.endswith("): Işık faturası ödendi.")
    assert "Kira" not in said


async def test_no_query_lists_the_latest_notes(add_note: Tool, search_notes: Tool) -> None:
    assert await search_notes.run() == NONE_STORED

    await add_note.run(text="bir")
    await add_note.run(text="iki")

    said = await search_notes.run(query="")

    assert said.startswith("The latest notes, newest first:\n#2 (")
    assert said.endswith("): bir")


async def test_a_query_nothing_mentions_is_said(add_note: Tool, search_notes: Tool) -> None:
    await add_note.run(text="bir")

    assert await search_notes.run(query="uzay mekiği") == "No note mentions 'uzay mekiği'."


async def test_a_query_too_short_to_search_is_explained(search_notes: Tool) -> None:
    said = await search_notes.run(query="ve")

    assert said.startswith("The search needs a word of at least 3 letters; 've' has none.")


# --------------------------------------------------------------------------
# delete_note, through the gate
# --------------------------------------------------------------------------


async def test_a_note_is_not_deleted_without_a_yes(delete_note: Tool, notes: NotesRepo) -> None:
    note = notes.add("Süt al.")
    user = FakeConfirm(answer=False)

    said = await through_the_gate(delete_note, user, note_id=note.id, text="Süt al.")

    assert said == DECLINED
    assert user.asked == ["'Süt al.' notu silinecek."]
    assert notes.get(note.id) is not None


async def test_a_note_is_deleted_after_a_yes_to_its_own_words(
    delete_note: Tool, notes: NotesRepo
) -> None:
    note = notes.add("Süt al.")
    notes.add("Ekmek al.")
    user = FakeConfirm(answer=True)

    said = await through_the_gate(delete_note, user, note_id=note.id, text="süt al")

    assert said == "Deleted note #1: 'Süt al.'. 1 notes remain."
    assert notes.get(note.id) is None


async def test_a_paraphrased_text_is_refused_even_after_a_yes(
    delete_note: Tool, notes: NotesRepo
) -> None:
    """The user heard the model's words, not the note's. Nothing is deleted
    and the model is told what the note actually says."""
    note = notes.add("Süt al, ekmek al.")
    user = FakeConfirm(answer=True)

    said = await through_the_gate(delete_note, user, note_id=note.id, text="Süt al.")

    assert said.startswith("Note #1 reads 'Süt al, ekmek al.', not 'Süt al.'. Nothing was deleted;")
    assert user.asked == ["'Süt al.' notu silinecek."]
    assert notes.get(note.id) is not None


async def test_a_number_that_is_no_note_is_said(delete_note: Tool, notes: NotesRepo) -> None:
    user = FakeConfirm(answer=True)

    said = await through_the_gate(delete_note, user, note_id=7, text="x")

    assert said == "There is no note #7. Call search_notes to find the one the user means."
