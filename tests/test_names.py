"""`store/names.py` (15 Sep 2026): the one matcher, for apps and for people.

Every claim here was true of `AppCatalog.find` before the matcher moved out
of it - `test_app_catalog.py` is the proof that nothing changed for apps -
and is now also true of a contact found by an alias. The measurements the
constants rest on are in the module's docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.store.names import CLOSE_ENOUGH, NEAR_ENOUGH, PREFIX_CHARS, NameIndex


@dataclass(frozen=True, slots=True)
class Thing:
    label: str


def index(*names: str, **spoken: str) -> NameIndex[Thing]:
    """Things added under their names; `spoken` maps a name to its spoken form."""
    found: NameIndex[Thing] = NameIndex()
    for name in names:
        found.add(Thing(name), name, spoken=spoken.get(name, ""))
    return found


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------


def test_the_listed_name_always_finds_its_own_item() -> None:
    """ "Outlook (classic)" is *said* "Outlook" and listed first; the app
    listed as "Outlook" still wins its own name."""
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("classic"), "Outlook (classic)", spoken="Outlook")
    found.add(Thing("new"), "Outlook")

    assert found.find("Outlook") == Thing("new")
    assert found.find("Outlook (classic)") == Thing("classic")


def test_a_spoken_form_or_an_alias_only_fills_a_gap() -> None:
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("first"), "Ahmet Yılmaz", aliases=("abi",))
    found.add(Thing("second"), "Mehmet", aliases=("abi",))

    assert found.find("abi") == Thing("first")


def test_words_come_from_the_spoken_form_and_from_every_alias() -> None:
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("ide"), "PyCharm 2026.2.1", spoken="PyCharm")
    found.add(Thing("ahmet"), "Ahmet Yılmaz", aliases=("Ahmet abi",))

    assert found.find("pycharm") == Thing("ide")
    # "2026" is a word of the listed name, not of the spoken one.
    assert found.find("2026") is None
    assert found.find("yılmaz") == Thing("ahmet")
    assert found.find("abi") == Thing("ahmet")


def test_clear_forgets_everything() -> None:
    found = index("Spotify")
    found.clear()

    assert len(found) == 0
    assert found.find("Spotify") is None


def test_len_counts_items_not_keys() -> None:
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("a"), "Ahmet Yılmaz", aliases=("Ahmet", "abi"))

    assert len(found) == 1


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


def test_the_whole_name_wins_over_a_word_of_another() -> None:
    found = index("Google Chrome", "Chrome Remote Desktop")

    assert found.find("chrome remote desktop") == Thing("Chrome Remote Desktop")


def test_one_word_of_a_name_is_enough() -> None:
    assert index("Google Chrome").find("chrome") == Thing("Google Chrome")


def test_a_prefix_of_three_letters_matches_and_two_do_not() -> None:
    found = index("Spotify")

    assert found.find("spo") == Thing("Spotify")
    assert PREFIX_CHARS == 3
    assert found.find("sp") is None


def test_case_accents_and_the_recognisers_spelling_do_not_decide() -> None:
    found = index("Şükrü Kaya", "Łukasz Nowak")

    assert found.find("SUKRU KAYA") == Thing("Şükrü Kaya")
    assert found.find("lukasz nowak") == Thing("Łukasz Nowak")


def test_a_misspelling_lands_when_it_is_close_enough() -> None:
    found = index("Notepad", "Calculator")

    assert found.find("notpad") == Thing("Notepad")
    assert found.find("kalkulator") == Thing("Calculator")
    assert CLOSE_ENOUGH == 0.7


def test_two_items_equally_close_are_nobodys() -> None:
    """Measured 2026-09-09 on 173 apps: at 0.6 "krom" opened the Command
    Prompt. Two equally likely answers is a question for the user."""
    found = index("Ahmet Kaya", "Ahmet Kayo")

    # Nine tenths like both, a prefix of neither: nobody's.
    assert found.find("ahmet kayx") is None
    assert found.closest("ahmet kayx") == ["Ahmet Kaya", "Ahmet Kayo"]


def test_a_word_two_apps_share_belongs_to_the_first_listed() -> None:
    """ "chrome" names the product; "Chrome Remote Desktop" merely contains
    it, and the catalogue's order already puts the likelier one first."""
    found = index("Google Chrome", "Chrome Remote Desktop")

    assert found.find("chrome") == Thing("Google Chrome")


def test_a_word_two_people_share_belongs_to_nobody() -> None:
    """Two Ahmets: nothing about the first name says which, and the
    recogniser's "ahmede" must not land on either by being close to the
    shared word."""
    found: NameIndex[Thing] = NameIndex(shared="nobody")
    found.add(Thing("yilmaz"), "Ahmet Yılmaz")
    found.add(Thing("kaya"), "Ahmet Kaya")

    assert found.find("Ahmet") is None
    assert found.find("ahme") is None  # a prefix of both
    assert found.find("ahmede") is None
    assert found.find("ahmet yilmaz") == Thing("yilmaz")
    assert found.find("kaya") == Thing("kaya")
    assert found.closest("Ahmet") == ["Ahmet Kaya", "Ahmet Yılmaz"]


def test_a_disputed_word_stays_nobodys_when_a_third_person_has_it() -> None:
    found: NameIndex[Thing] = NameIndex(shared="nobody")
    found.add(Thing("a"), "Ahmet Yılmaz")
    found.add(Thing("b"), "Ahmet Kaya")
    found.add(Thing("c"), "Ahmet Demir")

    assert found.find("Ahmet") is None


def test_an_alias_written_down_for_one_person_wins_the_word_another_merely_has() -> None:
    """The book refuses two people with the same alias; one person with the
    alias "Ahmet" and another merely called Ahmet Kaya is allowed, and the
    alias - a whole name the user wrote down - wins."""
    found: NameIndex[Thing] = NameIndex(shared="nobody")
    found.add(Thing("yilmaz"), "Ahmet Yılmaz", aliases=("Ahmet",))
    found.add(Thing("kaya"), "Ahmet Kaya")

    assert found.find("Ahmet") == Thing("yilmaz")


def test_several_words_are_measured_against_whole_names_only() -> None:
    """Measured 2026-09-13: "Text Editor" opened the Registry Editor,
    because "editor" alone is seven tenths of the phrase."""
    found = index("Registry Editor", "Notepad")

    assert found.find("Text Editor") is None


def test_the_recognisers_case_ending_is_close_enough() -> None:
    """ "Ahmet'e" is transcribed "ahmede" often enough to matter."""
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("ahmet"), "Ahmet Yılmaz", aliases=("Ahmet",))

    assert found.find("ahmede") == Thing("ahmet")


def test_nothing_is_found_for_nothing() -> None:
    assert index("Spotify").find("   ") is None


# --------------------------------------------------------------------------
# closest
# --------------------------------------------------------------------------


def test_closest_answers_with_the_names_the_items_were_added_under() -> None:
    found: NameIndex[Thing] = NameIndex()
    found.add(Thing("ahmet"), "Ahmet Yılmaz", aliases=("abi",))
    found.add(Thing("mehmet"), "Mehmet Yıldız")

    # "hmet yildiz" is eleven letters of Mehmet Yıldız; the alias "abi" is
    # a key of Ahmet's, but the answer is the name, never the alias.
    assert found.closest("ahmet yildiz") == ["Mehmet Yıldız", "Ahmet Yılmaz"]


def test_closest_is_nearest_first_and_bounded() -> None:
    found = index("Notepad", "Notepad++", "Notion", "Spotify")

    assert found.closest("notepad", limit=2) == ["Notepad", "Notepad++"]
    assert NEAR_ENOUGH == 0.4


def test_closest_is_empty_when_nothing_is_near() -> None:
    assert index("Spotify").closest("zzzzzz") == []


# --------------------------------------------------------------------------
# People: a guess has to look like the name (2026-09-15)
# --------------------------------------------------------------------------


def test_for_people_a_near_miss_that_does_not_start_like_the_name_is_nobody() -> None:
    """ "mehmet" and "ahmede" are both 0.727 like "ahmet"; only the second
    is Ahmet with a case ending on. Measured with the real model: the first
    sent Ahmet a message meant for a Mehmet who was not in the book."""
    people: NameIndex[Thing] = NameIndex(shared="nobody")
    people.add(Thing("ahmet"), "Ahmet Yılmaz", aliases=("Ahmet",))

    assert people.find("ahmede") == Thing("ahmet")
    assert people.find("mehmet") is None
    assert people.closest("mehmet") == ["Ahmet Yılmaz"]  # offered, not chosen


def test_for_apps_the_plain_ratio_stands() -> None:
    assert index("Calculator").find("kalkulator") == Thing("Calculator")


def test_certain_is_a_name_an_alias_or_a_word_and_not_a_guess() -> None:
    people: NameIndex[Thing] = NameIndex(shared="nobody")
    people.add(Thing("ahmet"), "Ahmet Yılmaz", aliases=("abi",))

    assert people.certain("Ahmet Yılmaz", Thing("ahmet"))
    assert people.certain("abi", Thing("ahmet"))
    assert people.certain("yılmaz", Thing("ahmet"))
    assert not people.certain("ahmede", Thing("ahmet"))
    assert not people.certain("ahm", Thing("ahmet"))
    assert not people.certain("Ahmet Yılmaz", Thing("somebody else"))
