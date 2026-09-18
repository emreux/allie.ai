"""Finding the thing the user named, by whatever they called it (design.md
section 3.6; extracted from `tools/system.py` on 15 September 2026).

The name the user said is not the name on the list. "krom" is Google
Chrome; "IŞIK" and "isik" are one word; the recogniser writes "pay charm"
for PyCharm and "ahmede" for "Ahmet'e". The app catalogue solved this once,
for apps; the address book of `messaging/contacts.py` needed the same
answer for people, and the second time a thing is written a helper is born
(python-guide rule 10) - copying the eighty lines would have given
people-name matching a different set of bugs from app-name matching.

**The order of the questions is the whole design.** Both sides folded with
`normalize_search` first (`store/normalize.py`); then the whole name, then
one word of a name ("chrome" is Google Chrome, "yılmaz" is Ahmet Yılmaz),
then the start of either at three characters or more ("spot"), and last
`difflib`'s closest match - only when one is *clearly* closest. Every step
is cheaper and surer than the one after it, which is why they are tried in
that order.

**The constants are measurements, not taste.** `CLOSE_ENOUGH` at 0.6 opened
Sticky Notes for "spotify" and the Command Prompt for "krom" on a machine
with 173 apps (2026-09-09); at 0.7 both are left for the model to ask
about, and "notpad", "kalkulator", "vscode" still land. Several words said
are measured against whole names only, because "Text Editor" opened the
Registry Editor when the word "editor" alone was seven tenths of the
phrase (2026-09-13). Opening the wrong app - or writing to the wrong
person - is worse than asking.

**A word two names share means two different things.** "chrome" is a word of
"Google Chrome" and of "Chrome Remote Desktop", and it means the first: the
product is named by the word, the other product merely contains it, and the
catalogue's order - the Start menu's, then the user's own history - already
puts the likelier one first. "ahmet" is a word of "Ahmet Yılmaz" and of
"Ahmet Kaya", and it means *neither*: nothing about a person's first name
says which of them is meant, and a message to the first Ahmet listed is a
message to the wrong person half the time. So an index is built with
`shared="first"` for things where the order knows more (apps), and
`shared="nobody"` for things where it knows nothing (people): under
"nobody", a word two items share is nobody's, a prefix two items share is
nobody's, and the model is handed both names to ask about.

**For people, a guess has to look like the name.** Measured 2026-09-15 with
the real model: "Mehmet'e selam yaz" with no Mehmet in the book went to
Ahmet Yılmaz, because "mehmet" is 0.727 like "ahmet" - the same score as
"ahmede", which is "Ahmet'e" as the recogniser writes it. The score cannot
tell them apart; the *front* of the word can. The recogniser's mistake is a
case ending glued to the name, so under "nobody" the closest-match step is
kept to keys that share their first `len(key) - 1` characters with what
was said: "ahmede" starts with "ahme", "mehmet" does not. A "Memet" for
Mehmet is then not found and is offered by `closest` instead, which is the
right side to err on. Apps keep the plain ratio: nobody is harmed by
"kalkulator" opening the calculator.

`certain` says whether a find was by a listed name, an alias or a whole
word of one - as opposed to a prefix or a guess - so that a channel can
put a guessed person's full name to the user before sending anything.

What is added under a name can be anything hashable: an `AppEntry`, a
`Contact`, a Telegram user. `NameIndex` never looks inside it.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Iterator
from difflib import SequenceMatcher
from typing import Literal

from assistant.store.normalize import normalize_search

__all__ = ["CLOSE_ENOUGH", "NEAR_ENOUGH", "PREFIX_CHARS", "NameIndex", "Shared"]

# Whom a word or a prefix that two items share belongs to (see above).
Shared = Literal["first", "nobody"]

# Below this ratio `difflib` is guessing rather than matching (see above).
CLOSE_ENOUGH = 0.7

# The bar for a *suggestion* is lower: when nothing matched, the model is
# better off with three names that were nearly it than with none.
NEAR_ENOUGH = 0.4

# A prefix shorter than this matches too much: "a" starts half the catalogue.
PREFIX_CHARS = 3


class NameIndex[T: Hashable]:
    """Items by the names people call them, found the way people say them."""

    def __init__(self, *, shared: Shared = "first") -> None:
        self._shared = shared
        self._by_name: dict[str, T] = {}
        self._by_word: dict[str, T] = {}
        # Under "nobody": the words two items claimed, kept out of `_by_word`
        # for good - a third item may not have them either.
        self._disputed: set[str] = set()
        # The name each item was added under: what `closest` answers with.
        self._names: dict[T, str] = {}

    def add(self, item: T, name: str, *, spoken: str = "", aliases: Iterable[str] = ()) -> None:
        """Files `item` under `name`, its spoken form and its aliases.

        The folded `name` always maps to `item`: a listed name finds its own
        thing even when another thing is *said* the same way and was added
        first ("Outlook (classic)" before "Outlook"). The spoken form and
        the aliases only stand where nothing stands yet. Single words come
        from `spoken or name` and from every alias, so that "yılmaz" and
        "abi" both find Ahmet Yılmaz - and not from a listed name whose
        spoken form differs, so that "2026" finds nothing.
        """
        key = normalize_search(name).strip()
        if not key:
            return
        self._names[item] = name
        self._by_name[key] = item
        said = normalize_search(spoken).strip()
        if said and said != key:
            self._by_name.setdefault(said, item)
        for alias in aliases:
            called = normalize_search(alias).strip()
            if called and called != key:
                self._by_name.setdefault(called, item)
            for word in _words(called):
                self._claim(word, item)
        for word in _words(said or key):
            self._claim(word, item)

    def _claim(self, word: str, item: T) -> None:
        if word in self._disputed:
            return
        holder = self._by_word.setdefault(word, item)
        if holder != item and self._shared == "nobody":
            del self._by_word[word]
            self._disputed.add(word)

    def clear(self) -> None:
        self._by_name.clear()
        self._by_word.clear()
        self._disputed.clear()
        self._names.clear()

    def __len__(self) -> int:
        return len(self._names)

    def find(self, spoken: str) -> T | None:
        """The item the user meant by `spoken`, or `None`."""
        wanted = normalize_search(spoken).strip()
        if not wanted:
            return None

        found = self._by_name.get(wanted) or self._by_word.get(wanted)
        if found is None and len(wanted) >= PREFIX_CHARS:
            found = self._starting_with(wanted)
        if found is None:
            found = self._one_close_enough(wanted)
        return found

    def _starting_with(self, wanted: str) -> T | None:
        """The item whose name or word starts with `wanted`: the first one
        listed, or - for people - the only one, else nobody."""
        started = [item for key, item in self._keys(words=True) if key.startswith(wanted)]
        if not started:
            return None
        if self._shared == "nobody" and any(item != started[0] for item in started):
            return None
        return started[0]

    def certain(self, spoken: str, item: T) -> bool:
        """Whether `spoken` is a listed name, an alias or a whole word of
        `item`'s - the matches that need no second look."""
        wanted = normalize_search(spoken).strip()
        return self._by_name.get(wanted) == item or self._by_word.get(wanted) == item

    def closest(self, spoken: str, *, limit: int = 3) -> list[str]:
        """Names near `spoken`, nearest first, for the model to offer when
        nothing matched."""
        wanted = normalize_search(spoken).strip()
        return [self._names[item] for _, item in self._ranked(wanted, cutoff=NEAR_ENOUGH)[:limit]]

    def _one_close_enough(self, wanted: str) -> T | None:
        """The closest item when one is clearly closest, otherwise nothing.

        Two things equally like what was said - "chrome" and "prompt" are
        both six tenths of "krom" - is a question for the user, not a coin
        toss.
        """
        ranked = self._ranked(wanted, cutoff=CLOSE_ENOUGH)
        if self._shared == "nobody":
            ranked = [(score, item) for score, item in ranked if self._looks_like(wanted, item)]
        if not ranked:
            return None
        (best_score, best), *others = ranked
        if others and others[0][0] == best_score:
            return None
        return best

    def _looks_like(self, wanted: str, item: T) -> bool:
        """Whether some key of `item` shares its front with `wanted` - all
        but its last character, at least (the docstring above)."""
        for key, holder in self._keys(words=" " not in wanted):
            if holder != item:
                continue
            front = key[: max(1, len(key) - 1)]
            if wanted.startswith(front):
                return True
        return False

    def _ranked(self, wanted: str, *, cutoff: float) -> list[tuple[float, T]]:
        """Every item with a key at least `cutoff` like `wanted`, the most
        alike first, each item once - scored by its best key, so that a
        name and its words do not fill the list with one item.

        One word said is measured against single words too ("krom" against
        "chrome"); several words are measured against whole names only
        ("Text Editor", above).
        """
        matcher = SequenceMatcher()
        matcher.set_seq2(wanted)
        best: dict[T, float] = {}
        for key, item in self._keys(words=" " not in wanted):
            matcher.set_seq1(key)
            # The two cheap upper bounds first, as `get_close_matches` does;
            # the real ratio is the expensive one, and this runs on the loop.
            if matcher.real_quick_ratio() < cutoff or matcher.quick_ratio() < cutoff:
                continue
            score = matcher.ratio()
            if score >= cutoff and score > best.get(item, 0.0):
                best[item] = score
        return sorted(((score, item) for item, score in best.items()), key=lambda pair: -pair[0])

    def _keys(self, *, words: bool) -> Iterator[tuple[str, T]]:
        yield from self._by_name.items()
        if words:
            yield from self._by_word.items()


def _words(key: str) -> list[str]:
    # The key is ASCII already; a single letter is not a word anyone asks for.
    return [word for word in re.findall(r"[a-z0-9]+", key) if len(word) > 1]
