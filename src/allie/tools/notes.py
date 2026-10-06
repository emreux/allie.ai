"""Notes: kept, found, deleted (design.md section 3.7, phase 4.1; 17 Sep 2026).

The secretary of section 2 takes a note when told to and finds it again
when asked, in whatever words it is asked for: "ışık faturası" and "isik
faturasi" are the same note, and "fatura" alone finds it (`NotesRepo`,
`store/normalize.py`). Two tools, one table: `notes` keeps and finds (one
tool since D38, 27 Sep 2026; it was `add_note` and `search_notes`),
`delete_note` removes.

`notes` is `safe`: it writes what the user just said to keep, or reads.
`delete_note` is `confirm`, the same
class as `forget`: it removes the user's own words, and the user hears
which. **The question names the note's real text, not the model's
paraphrase.** The gate fills the question from the call's arguments
(`agent/policy.py`), and the model could put any words there - so the tool
takes both the id and the text, and deletes only when the text it was
given *is* the note under that id, read the way search reads. A model that
paraphrased is told to call again with the note's own words, which
`notes` with action 'find' gives it. What the user confirmed is then what is deleted,
by construction.

What the tools answer is addressed to the model - English, what happened,
what to do next. The one sentence the user hears comes from the locale
pack, `TEXT` below being the end of the chain (section 3.12).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from allie.store.normalize import normalize_search
from allie.store.repos import MIN_QUERY_CHARS, SEARCH_LIMIT, Note, NotesRepo
from allie.tools.registry import Tool, tool

__all__ = ["MAX_NOTE_CHARS", "TEXT", "UNKNOWN_ACTION", "delete_note_for", "notes_for"]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from these tools. `{text}` is the note as it was written.
TEXT: dict[str, str] = {
    "note_delete_confirm": "Shall I delete the note '{text}'?",
}

# A note is a sentence or two said out loud, not a document.
MAX_NOTE_CHARS = 500

# The answers, addressed to the model.
KEPT = "Kept note #{id}: {text!r}. {count} notes are stored."
EMPTY = "Nothing to keep: the note was empty."
TOO_LONG = "Too long: a note is at most {limit} characters. Keep its essence and call again."
NONE_STORED = "No notes are stored."
LATEST = "The latest notes, newest first:\n{notes}"
FOUND = "Notes matching {query!r}, best first:\n{notes}"
NOT_FOUND = "No note mentions {query!r}."
SHORT_QUERY = (
    "The search needs a word of at least {limit} letters; {query!r} has none. "
    "Ask for more of the note, or leave the words empty for the latest notes."
)
DELETED = "Deleted note #{id}: {text!r}. {count} notes remain."
NO_SUCH_NOTE = (
    "There is no note #{id}. Call notes with action 'find' to find the one the user means."
)
UNKNOWN_ACTION = "No action {action!r}; the actions are 'add' and 'find'."

NoteAction = Literal["add", "find"]
MISMATCH = (
    "Note #{id} reads {stored!r}, not {given!r}. Nothing was deleted; call again with the "
    "note's own text so that the user hears what will be deleted."
)

_WORD = re.compile(r"\w+")


def notes_for(repo: NotesRepo) -> Tool:
    """`notes`, bound to the table it keeps and finds in (D38: it was
    `add_note` and `search_notes`)."""

    @tool(risk="safe")
    async def notes(
        action: Annotated[NoteAction, "'add' keeps a note; 'find' finds notes."],
        text: Annotated[
            str,
            "For add: the note in the user's own words, as they said it. For find: a few "
            "words from the note, in any spelling - empty for the latest notes.",
        ] = "",
    ) -> str:
        """Keeps and finds the user's notes. add: a note they asked you to
        take - "not al", "note this down", "write down that..." - kept in
        their words, not summarised; not for facts about the user themselves
        (remember keeps those) and not for reminders at a time (reminders
        keeps those). find: the notes mentioning some words, or the latest
        ones when no words are given - when they ask what they noted or to
        read a note back. Each line found starts with the note's number,
        which delete_note needs."""
        # A plain string: the schema offers two actions, and a model can
        # still send a third.
        chosen: str = action
        if chosen == "add":
            return _add(repo, text)
        if chosen == "find":
            return _find(repo, text)
        return UNKNOWN_ACTION.format(action=chosen)

    return notes


def _add(repo: NotesRepo, text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return EMPTY
    if len(cleaned) > MAX_NOTE_CHARS:
        return TOO_LONG.format(limit=MAX_NOTE_CHARS)
    note = repo.add(cleaned)
    return KEPT.format(id=note.id, text=note.text, count=repo.count())


def _find(repo: NotesRepo, text: str) -> str:
    wanted = " ".join(text.split())
    if not wanted:
        found = repo.latest(SEARCH_LIMIT)
        return LATEST.format(notes=_lines(found)) if found else NONE_STORED
    try:
        found = repo.search(wanted, limit=SEARCH_LIMIT)
    except ValueError:
        return SHORT_QUERY.format(limit=MIN_QUERY_CHARS, query=wanted)
    if not found:
        return NOT_FOUND.format(query=wanted)
    return FOUND.format(query=wanted, notes=_lines(found))


def delete_note_for(notes: NotesRepo, *, confirm_prompt: str = TEXT["note_delete_confirm"]) -> Tool:
    """`delete_note`, bound to the table and to the question it asks."""

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def delete_note(
        note_id: Annotated[int, "The note's number, as notes (find) listed it."],
        text: Annotated[str, "The note's text exactly as notes (find) listed it."],
    ) -> str:
        """Deletes one of the user's notes, once they have confirmed out
        loud. Call notes with action 'find' first and pass the number and
        the text of the note as listed there: the user hears the text before deciding,
        and a text that does not match the note is not deleted."""
        stored = notes.get(note_id)
        if stored is None:
            return NO_SUCH_NOTE.format(id=note_id)
        if _phrase(stored.text) != _phrase(text):
            return MISMATCH.format(id=note_id, stored=stored.text, given=text)
        notes.delete(note_id)
        return DELETED.format(id=note_id, text=stored.text, count=notes.count())

    return delete_note


def _lines(found: list[Note]) -> str:
    """One note per line: `#12 (2026-09-17): the text`."""
    return "\n".join(f"#{note.id} ({_date(note.created_at)}): {note.text}" for note in found)


def _date(created_at: int) -> str:
    """The day the note was taken, in local time, as the model reads dates."""
    return datetime.fromtimestamp(created_at).astimezone().date().isoformat()


def _phrase(text: str) -> str:
    """`text` as the words in it, folded the way search folds - the same
    reading `store/memory.py` gives a fact."""
    return " ".join(_WORD.findall(normalize_search(text)))
