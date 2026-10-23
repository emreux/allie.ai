"""The gate's question on the phone: two buttons (plan.md D59).

By voice a tool that asks first is answered aloud and judged by the referee
(D54). On the phone the question is a message with two buttons, and a tap is
not ambiguous. What these tests hold to:

**No answer is not a no** (D36). A question nobody answered in time, or one
the owner wrote past, is `None` - unheard - and the message says nothing was
done.

**Typing the button's own word is pressing it**, and anything else typed
closes the question unanswered and is handed back, to be read next. These
are the buttons' two labels, not a word list.

**A press belongs to one question.** Each question's buttons carry its own
code; a press on an old one is told the question is closed and moves nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from allie.remote.bot import BotError, Press
from allie.remote.questions import TEXT, ButtonQuestion

CHAT = 42
ASKED = "Spotify'ı kapatayım mı?"

WORDING = {
    **TEXT,
    "remote_yes": "Evet",
    "remote_no": "Hayır",
    "remote_answered_yes": "→ Evet",
    "remote_answered_no": "→ Hayır",
    "remote_unanswered": "→ Cevap gelmedi, yapmadım.",
    "remote_question_closed": "Bu soru kapandı.",
}


class Phone:
    """Stands in for the bot: what was sent, edited and answered."""

    def __init__(self, *, refuse: bool = False, refuse_edit: bool = False) -> None:
        self.sent: list[tuple[int, str, list[tuple[str, str]]]] = []
        self.edits: list[tuple[int, int, str]] = []
        self.answered: list[tuple[str, str]] = []
        self.refuse = refuse
        self.refuse_edit = refuse_edit

    async def send(self, chat: int, text: str, *, buttons: Sequence[tuple[str, str]] = ()) -> int:
        if self.refuse:
            raise BotError("unreachable")
        self.sent.append((chat, text, list(buttons)))
        return 100 + len(self.sent)

    async def edit(self, chat: int, message_id: int, text: str) -> None:
        if self.refuse_edit:
            raise BotError("unreachable")
        self.edits.append((chat, message_id, text))

    async def answer_button(self, query_id: str, text: str = "") -> None:
        self.answered.append((query_id, text))


async def asked(question: ButtonQuestion, phone: Phone) -> asyncio.Task[bool | None]:
    """The question put, and on the phone."""
    task = asyncio.create_task(question(ASKED))
    for _ in range(100):
        if phone.sent:
            return task
        await asyncio.sleep(0)
    raise AssertionError("the question never reached the phone")


def pressing(phone: Phone, answer: str, *, query: str = "q-1") -> Press:
    """A tap on the last question's `answer` button."""
    chat, _, buttons = phone.sent[-1]
    data = next(data for _, data in buttons if data.endswith(f":{answer}"))
    return Press(
        update_id=1,
        query_id=query,
        sender_id=CHAT,
        chat_id=chat,
        message_id=100 + len(phone.sent),
        data=data,
    )


async def test_a_yes_tap_answers_yes_and_says_so_under_the_question() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    task = await asked(question, phone)
    assert question.open
    await question.press(pressing(phone, "yes"))

    assert await task is True
    assert not question.open
    assert phone.edits == [(CHAT, 101, f"{ASKED}\n\n→ Evet")]
    assert phone.answered == [("q-1", "")]


async def test_a_no_tap_answers_no() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    task = await asked(question, phone)
    await question.press(pressing(phone, "no"))

    assert await task is False
    assert phone.edits == [(CHAT, 101, f"{ASKED}\n\n→ Hayır")]


async def test_no_answer_in_time_is_unheard_and_not_a_no() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=0.05)

    assert await question(ASKED) is None
    assert not question.open
    assert phone.edits == [(CHAT, 101, f"{ASKED}\n\n→ Cevap gelmedi, yapmadım.")]


async def test_the_buttons_are_yes_then_no_and_carry_their_own_questions_code() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=0.01)

    await question("bir")
    await question("iki")

    first, second = phone.sent[0][2], phone.sent[1][2]
    assert [label for label, _ in first] == ["Evet", "Hayır"]
    assert [data.split(":")[1] for _, data in first] == ["yes", "no"]
    assert all(len(data.encode()) <= 64 for _, data in first + second)
    assert first[0][1].split(":")[0] != second[0][1].split(":")[0]


@pytest.mark.parametrize(("typed", "answer"), [("evet", True), ("  Hayır. ", False)])
async def test_typing_a_buttons_word_is_pressing_it(typed: str, answer: bool) -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    task = await asked(question, phone)

    assert question.offer_text(typed) is True
    assert await task is answer


async def test_anything_else_typed_closes_the_question_unanswered_and_is_handed_back() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    task = await asked(question, phone)

    assert question.offer_text("bir de şarkı aç") is False
    assert await task is None
    assert phone.edits == [(CHAT, 101, f"{ASKED}\n\n→ Cevap gelmedi, yapmadım.")]


def test_a_text_with_no_question_open_is_not_taken() -> None:
    question = ButtonQuestion(Phone(), CHAT, wording=WORDING, seconds=5)

    assert question.offer_text("evet") is False


async def test_a_press_on_an_old_question_is_told_it_is_closed_and_moves_nothing() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)
    task = await asked(question, phone)

    old = Press(
        update_id=2, query_id="q-old", sender_id=CHAT, chat_id=CHAT, message_id=55, data="0f00:yes"
    )
    await question.press(old)

    assert phone.answered == [("q-old", "Bu soru kapandı.")]
    assert question.open
    await question.press(pressing(phone, "no", query="q-2"))
    assert await task is False


async def test_a_press_with_nothing_open_is_told_it_is_closed() -> None:
    phone = Phone()
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    await question.press(
        Press(update_id=3, query_id="q-3", sender_id=CHAT, chat_id=CHAT, message_id=9, data="a:yes")
    )

    assert phone.answered == [("q-3", "Bu soru kapandı.")]


async def test_a_question_that_could_not_be_sent_is_unheard() -> None:
    phone = Phone(refuse=True)
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    assert await question(ASKED) is None
    assert not question.open


async def test_the_answer_stands_when_the_message_could_not_be_rewritten() -> None:
    phone = Phone(refuse_edit=True)
    question = ButtonQuestion(phone, CHAT, wording=WORDING, seconds=5)

    task = await asked(question, phone)
    await question.press(pressing(phone, "yes"))

    assert await task is True
