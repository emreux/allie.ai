"""Locale packs: everything that depends on the language, kept out of the code.

Section 3.12 turns "which language does the product speak" into a
configuration dimension, and two claims make that true rather than decorative.

The first is the fallback chain - the pack that was asked for, then `en`, then
the English constant in the code that says the sentence. It is what makes an
unsupported language *usable* instead of broken, and it is tested against packs
written into `tmp_path`: the mechanism is the point, and a test that read the
real `tr.toml` would fail every time a sentence was reworded.

The second is that the packs this project ships answer everything phase 1 asks
of them. A missing key is survivable by design, which is exactly why nobody
would ever notice one.
"""

from __future__ import annotations

import string
import tomllib
from pathlib import Path
from typing import Any

from allie import __main__ as cli
from allie import app, locales, setup_wizard
from allie.agent import policy
from allie.live.probe import QUESTION
from allie.locales import FALLBACK_CODE, available, iso_code, load, system_code
from allie.messaging import telegram
from allie.scheduler import runner as scheduler
from allie.tools import mail as mail_tools
from allie.tools import memory as memory_tools
from allie.tools import messaging as messaging_tools
from allie.tools import notes as notes_tools
from allie.tools import reminders as reminder_tools
from allie.tools import store as store_tools
from allie.tools import system as system_tools
from allie.ui import status, tray, window

PACKAGED = Path(locales.__file__).parent

# A pack has one `[ui]` table, and six modules say things: the wizard asks the
# questions, `app.py` says what went wrong in a turn, the status line says what
# is happening, the command line says whether it can start at all and what was
# spent, the gate adds a sentence to a risky tool's question, and the memory
# tools ask one before a fact is forgotten. What a pack may translate is the
# six of them together.
TABLES = {
    "setup_wizard": setup_wizard.TEXT,
    "app": app.TEXT,
    "ui.status": status.TEXT,
    "ui.tray": tray.TEXT,
    "ui.window": window.TEXT,
    "__main__": cli.TEXT,
    "__main__.doctor": cli.DOCTOR_TEXT,
    "agent.policy": policy.TEXT,
    "tools.memory": memory_tools.TEXT,
    "tools.system": system_tools.TEXT,
    "tools.store": store_tools.TEXT,
    "tools.messaging": messaging_tools.TEXT,
    "tools.mail": mail_tools.TEXT,
    "tools.notes": notes_tools.TEXT,
    "tools.reminders": reminder_tools.TEXT,
    "scheduler.runner": scheduler.TEXT,
    "messaging.telegram": telegram.TEXT,
}
SENTENCES = {key: text for table in TABLES.values() for key, text in table.items()}


def shipped() -> list[Path]:
    return sorted(PACKAGED.glob("*.toml"))


def read(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def write(directory: Path, code: str, body: str) -> None:
    (directory / f"{code}.toml").write_text(body, encoding="utf-8")


def fields(sentence: str) -> set[str]:
    """The `{name}` placeholders a sentence expects to be given."""
    return {name for _, name, _, _ in string.Formatter().parse(sentence) if name}


# --------------------------------------------------------------------------
# The packs this project ships
# --------------------------------------------------------------------------


def test_phase_one_ships_the_two_languages_it_promises() -> None:
    """A list rather than a set: a file that is not a language still loads as
    one, and would sit in the menu as a second, identical English."""
    assert [pack.code for pack in available()] == ["en", "tr"]


def test_the_template_is_a_scaffold_and_not_a_language() -> None:
    """It exists to be copied. Offered in the menu it would be a language that
    speaks nothing but placeholders."""
    assert (PACKAGED / "_template.toml").is_file()
    assert not any(pack.code.startswith("_") for pack in available())


def test_every_shipped_pack_answers_what_phase_one_asks_of_it() -> None:
    for pack in available():
        assert pack.name.strip(), f"{pack.code} has no name to show in a menu"
        assert pack.stt_language, f"{pack.code} gives the speech recogniser no hint"
        assert pack.voice("sapi"), f"{pack.code} names no Windows voice"


def test_a_pack_is_filed_under_the_name_of_its_own_file() -> None:
    """The file name is what the loader trusts. A `code` that disagrees with it
    means `config.toml` and the pack are quietly talking about two languages."""
    for path in shipped():
        if path.stem.startswith("_"):
            continue
        assert read(path).get("code") == path.stem, path.name


def test_english_is_not_written_down_twice() -> None:
    """`en.toml` carries no sentences: English is already the constant the
    chain ends at, and a second copy is a second thing to keep in step. The
    same goes for the yes and no words the window listens for."""
    assert "ui" not in read(PACKAGED / "en.toml")
    assert "speech" not in read(PACKAGED / "en.toml")
    assert "probe" not in read(PACKAGED / "en.toml")


def test_the_template_offers_every_sentence_a_translator_has_to_write() -> None:
    """A key the template forgets is a sentence nobody ever translates, and the
    product answers it in English for good without anyone noticing."""
    assert set(read(PACKAGED / "_template.toml")["ui"]) == set(SENTENCES)


def test_no_two_modules_claim_the_same_sentence() -> None:
    """One table, one key, one sentence. Two modules using the same key for two
    different sentences means the translation of one becomes the other."""
    for name, table in TABLES.items():
        others = {key for other, t in TABLES.items() if other != name for key in t}

        assert not set(table) & others, f"{name} shares {set(table) & others}"


def test_a_pack_translates_only_sentences_the_product_actually_says() -> None:
    """A key left behind by a reworded question is dead weight that reads like
    finished work."""
    for path in shipped():
        offered = set(read(path).get("ui", {}))
        assert offered <= set(SENTENCES), f"{path.name} translates {offered - set(SENTENCES)}"


def test_turkish_is_complete_because_it_is_one_of_the_two_that_is_promised() -> None:
    """The fallback chain would hide a gap, and an English sentence in the
    middle of a Turkish wizard is a defect, not a graceful degradation."""
    assert set(read(PACKAGED / "tr.toml")["ui"]) == set(SENTENCES)


def test_a_translation_asks_for_the_same_fields_as_the_sentence_it_replaces() -> None:
    """The sentence is formatted with what the caller passes. A renamed
    placeholder is a `KeyError` in front of the user, halfway through setup."""
    for path in shipped():
        for key, sentence in read(path).get("ui", {}).items():
            assert fields(sentence) == fields(SENTENCES[key]), f"{path.name}: {key}"


# --------------------------------------------------------------------------
# The fallback chain: the pack, then English, then the code
# --------------------------------------------------------------------------


def test_the_pack_that_was_asked_for_outranks_english(tmp_path: Path) -> None:
    """The first link of the chain. Both packs answer this key, and the whole
    point of asking for one of them is that its answer is used."""
    write(tmp_path, "en", '[ui]\nmodel = "Which model should answer?"\n')
    write(tmp_path, "de", '[ui]\nmodel = "Welches Modell?"\n')

    assert load("de", directory=tmp_path).say("model", "constant") == "Welches Modell?"


def test_a_sentence_the_pack_forgets_is_said_in_english(tmp_path: Path) -> None:
    write(tmp_path, "en", '[ui]\nmodel = "Which model should answer?"\n')
    write(tmp_path, "de", '[ui]\nwelcome = "Willkommen."\n')

    said = load("de", directory=tmp_path).say("model", "constant in the code")

    assert said == "Which model should answer?"


def test_a_sentence_no_pack_offers_falls_back_to_the_constant_in_the_code(tmp_path: Path) -> None:
    write(tmp_path, "en", 'name = "English"\n')

    assert load("de", directory=tmp_path).say("model", "Which model?") == "Which model?"


def test_a_translation_left_blank_is_not_a_translation(tmp_path: Path) -> None:
    write(tmp_path, "de", '[ui]\nmodel = ""\n')

    assert load("de", directory=tmp_path).say("model", "Which model?") == "Which model?"


def test_a_language_with_no_pack_at_all_is_still_a_working_locale(tmp_path: Path) -> None:
    """Section 3.12's promise: an unsupported language gets an English
    interface, not a broken product."""
    pack = load("de", directory=tmp_path)

    assert (pack.code, pack.name, pack.stt_language) == ("de", "de", "de")
    assert pack.say("welcome", "Hello.") == "Hello."


def test_english_does_not_lend_its_voice_to_another_language(tmp_path: Path) -> None:
    """Sentences fall back; identity does not. An English voice reading German
    is worse than no preference at all."""
    write(tmp_path, "en", '[tts.voice]\nsapi = "Zira"\n')

    assert load("de", directory=tmp_path).voice("sapi") is None


def test_a_voice_left_blank_is_no_preference_at_all(tmp_path: Path) -> None:
    """The template ships the key with a placeholder in it. Emptying it is how
    a translator says their language has no voice worth naming."""
    write(tmp_path, "de", '[tts.voice]\nsapi = ""\n')

    assert load("de", directory=tmp_path).voice("sapi") is None


def test_english_does_not_lend_its_speech_language_either(tmp_path: Path) -> None:
    """Whisper told to expect English would return English-shaped nonsense for
    every German sentence, and the model would never see the German."""
    write(tmp_path, "en", '[stt]\nlanguage = "en"\n')

    assert load("de", directory=tmp_path).stt_language == "de"


# --------------------------------------------------------------------------
# Asking for a pack
# --------------------------------------------------------------------------


def test_the_region_and_the_case_are_not_part_of_the_question(tmp_path: Path) -> None:
    """`tr-TR` comes out of Windows, `TR` out of a settings file somebody
    typed by hand; both name the same pack."""
    write(tmp_path, "tr", 'name = "Turkce"\n')

    assert load("tr-TR", directory=tmp_path).name == "Turkce"
    assert load("TR", directory=tmp_path).name == "Turkce"
    # The code is carried around afterwards - as the speech hint, and as what
    # the pack is called. Windows finds `TR.toml` whatever the case, so the
    # name alone would not notice this.
    assert load("TR", directory=tmp_path).code == "tr"


def test_asking_for_no_language_at_all_gives_the_fallback(tmp_path: Path) -> None:
    write(tmp_path, "en", 'name = "English"\n')

    assert load(None, directory=tmp_path).code == FALLBACK_CODE
    assert load("", directory=tmp_path).code == FALLBACK_CODE


def test_a_code_that_could_name_a_file_somewhere_else_is_refused(tmp_path: Path) -> None:
    """The code comes out of a settings file. A language is letters; anything
    else is a path, and reading one would be this module's own fault."""
    write(tmp_path, "en", 'name = "English"\n')

    assert load("../secrets", directory=tmp_path).code == FALLBACK_CODE


def test_the_languages_are_offered_in_a_settled_order(tmp_path: Path) -> None:
    """A menu that reshuffles itself between runs is a menu nobody learns."""
    for code in ("tr", "en", "de"):
        write(tmp_path, code, f'name = "{code}"\n')

    assert [pack.code for pack in available(directory=tmp_path)] == ["de", "en", "tr"]


# --------------------------------------------------------------------------
# Packs written by other people
# --------------------------------------------------------------------------


def test_a_pack_that_does_not_parse_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """A contributor's typo must not stop the assistant from starting."""
    write(tmp_path, "en", 'name = "English"\n[ui]\nwelcome = "Hello."\n')
    write(tmp_path, "de", "this is not toml at all\n")

    assert load("de", directory=tmp_path).say("welcome", "constant") == "Hello."
    assert [pack.code for pack in available(directory=tmp_path)] == ["en"]


def test_a_pack_of_the_wrong_shape_is_read_as_far_as_it_makes_sense(tmp_path: Path) -> None:
    """Every value here is the wrong kind of thing, and each one falls back on
    its own rather than taking the whole pack down with it."""
    write(tmp_path, "de", 'name = 5\nstt = "German"\n[tts]\nvoice = 3\n[ui]\nmodel = 7\n')

    pack = load("de", directory=tmp_path)

    assert (pack.code, pack.name, pack.stt_language) == ("de", "de", "de")
    assert pack.voice("sapi") is None
    assert pack.say("model", "Which model?") == "Which model?"


# --------------------------------------------------------------------------
# The language Windows itself is in
# --------------------------------------------------------------------------


def test_windows_language_identifiers_are_read_as_iso_codes() -> None:
    assert iso_code(0x041F) == "tr"
    assert iso_code(0x0409) == "en"


def test_a_language_windows_will_not_name_is_read_as_the_fallback() -> None:
    """Unlike a voice, which may honestly have no language, the interface has
    to be in *some* language before the first question is asked."""
    assert iso_code(0xFFFF) == FALLBACK_CODE


def test_the_language_windows_is_in_can_be_asked_for() -> None:
    assert system_code().isalpha()
    assert system_code() == system_code().casefold()


# --------------------------------------------------------------------------
# What the recogniser is told to expect (D3, D10; 2026-09-22)
# --------------------------------------------------------------------------


def test_no_shipped_pack_writes_a_recogniser_prompt_of_its_own() -> None:
    """There was a `[stt] prompt` here - a sentence with the installed apps'
    names in it - while every sentence the user spoke went through the local
    decoder. The live model hears those, so the prompt is built from the words
    of the confirmation window instead (`app.confirm_prompt`), and a key that
    fed nothing would have translators filling it in for years."""
    for path in shipped():
        assert "prompt" not in read(path).get("stt", {}), path.name


def test_the_turkish_pack_has_the_words_the_store_install_asks_with() -> None:
    """`install_app`'s question (2026-09-13), with both of its blanks, and
    the word for a publisher the Store did not name."""
    pack = load("tr")

    question = pack.say("store_install_confirm", "")
    assert "{name}" in question
    assert "{publisher}" in question
    assert pack.say("unknown_publisher", "")


# --------------------------------------------------------------------------
# The yes and no words (2.3)
# --------------------------------------------------------------------------


def test_turkish_knows_its_yes_and_no() -> None:
    """The window of section 3.1 rule 2 listens for these; a pack without
    them would have the assistant say "evet ya da hayır de" and hear neither."""
    pack = load("tr")

    assert pack.yes_words and pack.no_words
    assert not set(pack.yes_words) & set(pack.no_words)


def test_the_template_offers_the_three_lists_a_translator_has_to_fill() -> None:
    assert set(read(PACKAGED / "_template.toml")["speech"]) == {"yes_words", "no_words", "filler"}


def test_turkish_has_something_to_say_while_a_tool_takes_its_time() -> None:
    """The filler of section 4 (2.8): a pack without one is heard saying
    the English one in the middle of a Turkish answer."""
    assert load("tr").fillers


def test_the_fillers_come_from_the_pack_in_the_order_they_are_listed(tmp_path: Path) -> None:
    write(tmp_path, "de", '[speech]\nfiller = [" Moment... ", "Ich schaue nach..."]\n')

    assert load("de", directory=tmp_path).fillers == ("Moment...", "Ich schaue nach...")


def test_a_pack_without_fillers_leaves_them_to_the_code(tmp_path: Path) -> None:
    write(tmp_path, "de", '[speech]\nyes_words = ["ja"]\n')

    assert load("de", directory=tmp_path).fillers == ()


def test_the_yes_and_no_words_come_from_the_pack(tmp_path: Path) -> None:
    write(tmp_path, "de", '[speech]\nyes_words = ["ja", " jawohl "]\nno_words = ["nein"]\n')

    pack = load("de", directory=tmp_path)

    assert (pack.yes_words, pack.no_words) == (("ja", "jawohl"), ("nein",))


def test_a_pack_without_them_leaves_the_words_to_the_code(tmp_path: Path) -> None:
    """Like a sentence: the English words live beside the code that listens
    for them, with the English hint that names them, and not in `en.toml`."""
    write(tmp_path, "en", '[speech]\nyes_words = ["yes"]\n')

    assert load("de", directory=tmp_path).yes_words == ()
    assert load("de", directory=tmp_path).no_words == ()


def test_words_of_the_wrong_shape_are_read_as_far_as_they_make_sense(tmp_path: Path) -> None:
    write(tmp_path, "de", '[speech]\nyes_words = "ja"\nno_words = ["nein", 3, ""]\n')

    pack = load("de", directory=tmp_path)

    assert (pack.yes_words, pack.no_words) == ((), ("nein",))


# --------------------------------------------------------------------------
# The short commands of the old fast path are gone (plan.md D7): no pack lists
# them, and the loader stopped carrying them on 2026-09-22.
# --------------------------------------------------------------------------


def test_no_shipped_pack_lists_short_commands() -> None:
    """ "Saat kaç" goes to the model and "dur" is what talking over it does;
    a table nothing answers would read as finished work."""
    for path in shipped():
        assert "intents" not in read(path), path.name


# --------------------------------------------------------------------------
# The probe question (2.6)
# --------------------------------------------------------------------------


def test_turkish_asks_the_probe_question_in_turkish() -> None:
    """Section 3.2 wanted one request to check tool calling and the user's
    language together; the pack is what keeps that true without a Turkish
    sentence in the code."""
    assert load("tr").probe_question
    assert load("tr").probe_question != QUESTION


def test_the_template_offers_the_probe_question_a_translator_has_to_write() -> None:
    assert set(read(PACKAGED / "_template.toml")["probe"]) == {"question"}


def test_the_probe_question_comes_from_the_pack(tmp_path: Path) -> None:
    write(tmp_path, "de", '[probe]\nquestion = " Wie spät ist es in Istanbul? "\n')

    assert load("de", directory=tmp_path).probe_question == "Wie spät ist es in Istanbul?"


def test_a_pack_without_one_leaves_the_question_to_the_code(tmp_path: Path) -> None:
    """Like a sentence: English lends nothing, the constant beside the code
    that asks (`live/probe.py`) answers."""
    write(tmp_path, "en", '[probe]\nquestion = "What time is it?"\n')

    assert load("de", directory=tmp_path).probe_question == ""


def test_a_question_of_the_wrong_shape_is_no_question(tmp_path: Path) -> None:
    write(tmp_path, "de", '[probe]\nquestion = ["a", "b"]\n')
    write(tmp_path, "fr", 'probe = "not a table"\n')

    assert load("de", directory=tmp_path).probe_question == ""
    assert load("fr", directory=tmp_path).probe_question == ""


# --------------------------------------------------------------------------
# The two hints the live model needs (plan.md D19, ADR-001 section 6)
# --------------------------------------------------------------------------


def test_turkish_tells_the_live_model_which_language_it_will_hear() -> None:
    """Measured 2026-09-18: without the two hints a short or quiet "Saat
    kaç" is heard as Hindi and answered in it. The BCP-47 code goes to the
    session, the sentence to the prompt - both from the pack, so that no
    language is named in the code (section 3.12)."""
    pack = load("tr")

    assert pack.language_code == "tr-TR"
    assert pack.user_language_rule
    assert "Turkish" in pack.user_language_rule


def test_english_leaves_both_hints_to_the_server_and_the_prompt() -> None:
    """ADR-001: both empty in `en.toml` - the server's default language and
    the prompt's own mirroring rule are what an English pack means."""
    pack = load("en")

    assert (pack.language_code, pack.user_language_rule) == ("", "")


def test_the_template_offers_the_two_hints_a_translator_has_to_write() -> None:
    assert set(read(PACKAGED / "_template.toml")["live"]) == {
        "language_code",
        "user_language_rule",
    }


def test_the_hints_come_from_the_pack_that_was_asked_for(tmp_path: Path) -> None:
    write(
        tmp_path,
        "de",
        '[live]\nlanguage_code = " de-DE "\nuser_language_rule = " The user speaks German. "\n',
    )

    pack = load("de", directory=tmp_path)

    assert (pack.language_code, pack.user_language_rule) == ("de-DE", "The user speaks German.")


def test_english_does_not_lend_its_hints_to_another_language(tmp_path: Path) -> None:
    """Identity, not a sentence: an English hint on a German pack would make
    the server hear German as English, which is the failure the hints
    exist to prevent."""
    write(tmp_path, "en", '[live]\nlanguage_code = "en-US"\nuser_language_rule = "English."\n')

    pack = load("de", directory=tmp_path)

    assert (pack.language_code, pack.user_language_rule) == ("", "")


def test_hints_of_the_wrong_shape_are_no_hints(tmp_path: Path) -> None:
    write(tmp_path, "de", '[live]\nlanguage_code = 5\nuser_language_rule = ["a"]\n')
    write(tmp_path, "fr", 'live = "not a table"\n')

    assert load("de", directory=tmp_path).language_code == ""
    assert load("de", directory=tmp_path).user_language_rule == ""
    assert load("fr", directory=tmp_path).language_code == ""
