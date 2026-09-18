"""`messaging/telegram.py` (15 Sep 2026): the channel over a fake client,
and the login over a fake prompter. No network, no Telethon object is ever
built; `TelethonClient` is only imported.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from assistant.messaging.contacts import AddressBook, Contact
from assistant.messaging.telegram import (
    NO_SUCH_USER,
    NOT_LOGGED_IN,
    SAID,
    TEXT,
    Login,
    LoginError,
    PasswordNeededError,
    Person,
    Telegram,
    TelegramError,
    login,
)
from assistant.tools.messaging import NoRecipientError

AHMET = Person(id=1, name="Ahmet Yılmaz", username="ahmetyilmaz", phone="905320000000")
AYSE = Person(id=2, name="Ayşe Demir", username="", phone="")
MEHMET = Person(id=3, name="Mehmet Kaya", username="mkaya", phone="905331112233")

BOOK = AddressBook(
    [
        Contact(name="Ahmet Yılmaz", aliases=("abi",), telegram="ahmetyilmaz"),
        Contact(name="Mehmet Kaya", phone="+90 533 111 22 33"),
        Contact(name="Zeynep", telegram="zeynep_z"),
        Contact(name="Can", telegram="nobody_here"),
    ]
)


class FakeClient:
    """Scripted: whether it is logged in, whom it lists, what it refuses."""

    def __init__(
        self,
        *people: Person,
        authorized: bool = True,
        resolvable: dict[str, Person] | None = None,
        failing: TelegramError | None = None,
    ) -> None:
        self.people = list(people)
        self._authorized = authorized
        self.resolvable = resolvable or {}
        self.failing = failing
        self.sent: list[tuple[str, str]] = []
        self.connected = 0
        self.disconnected = 0
        self.listed = 0
        self.resolved: list[str] = []

    async def connect(self) -> None:
        self.connected += 1

    async def authorized(self) -> bool:
        return self._authorized

    async def contacts(self) -> list[Person]:
        self.listed += 1
        if self.failing is not None:
            raise self.failing
        return list(self.people)

    async def resolve(self, username: str) -> Person | None:
        self.resolved.append(username)
        return self.resolvable.get(username)

    async def send(self, person: Person, text: str) -> None:
        if self.failing is not None:
            raise self.failing
        self.sent.append((person.name, text))

    async def disconnect(self) -> None:
        self.disconnected += 1


# --------------------------------------------------------------------------
# Not logged in
# --------------------------------------------------------------------------


async def test_without_a_client_nothing_is_configured_and_the_answer_says_how_to_log_in() -> None:
    channel = Telegram(BOOK, client=None)

    assert channel.configured is False
    with pytest.raises(NoRecipientError, match="live-assistant telegram login"):
        await channel.resolve("Ahmet")
    with pytest.raises(NoRecipientError, match=NOT_LOGGED_IN):
        await channel.send(AHMET, "hi")


async def test_a_client_that_is_not_authorized_says_the_same_and_sends_nothing() -> None:
    client = FakeClient(AHMET, authorized=False)
    channel = Telegram(BOOK, client=client)

    with pytest.raises(NoRecipientError, match=NOT_LOGGED_IN):
        await channel.resolve("Ahmet")
    assert client.sent == []


async def test_it_connects_once_on_first_use_and_disconnects_once_at_close() -> None:
    client = FakeClient(AHMET)
    channel = Telegram(BOOK, client=client)

    await channel.resolve("Ahmet")
    await channel.resolve("Ahmet")
    await channel.close()
    await channel.close()

    assert client.connected == 1
    assert client.disconnected == 1


# --------------------------------------------------------------------------
# Resolving, in the order the spec gives
# --------------------------------------------------------------------------


async def test_the_books_username_finds_the_person_in_the_list() -> None:
    client = FakeClient(AHMET, AYSE)

    found = await Telegram(BOOK, client=client).resolve("abi")

    assert found == AHMET
    assert client.resolved == []  # the list had them; Telegram was not asked


async def test_the_books_phone_finds_the_person_in_the_list() -> None:
    client = FakeClient(AHMET, MEHMET)

    assert await Telegram(BOOK, client=client).resolve("Mehmet") == MEHMET


async def test_a_username_the_list_lacks_is_asked_of_telegram() -> None:
    zeynep = Person(id=9, name="Zeynep Z", username="zeynep_z", phone="")
    client = FakeClient(AHMET, resolvable={"zeynep_z": zeynep})

    assert await Telegram(BOOK, client=client).resolve("Zeynep") == zeynep
    assert client.resolved == ["zeynep_z"]


async def test_a_username_nobody_has_is_a_sentence_naming_it() -> None:
    client = FakeClient(AHMET)

    with pytest.raises(NoRecipientError, match=NO_SUCH_USER.format(username="nobody_here")):
        await Telegram(BOOK, client=client).resolve("Can")


async def test_a_name_not_in_the_book_is_found_in_the_list() -> None:
    client = FakeClient(AHMET, AYSE)

    assert await Telegram(BOOK, client=client).resolve("Ayşe") == AYSE
    assert await Telegram(BOOK, client=client).resolve("ayse demir") == AYSE


async def test_a_username_said_out_loud_is_found_in_the_list() -> None:
    client = FakeClient(MEHMET)

    assert await Telegram(AddressBook(), client=client).resolve("mkaya") == MEHMET


async def test_a_miss_fetches_the_list_once_more_before_giving_up() -> None:
    """A contact added today is in the list Telegram holds, not in the one
    fetched at the first send."""
    client = FakeClient(AHMET)
    channel = Telegram(AddressBook(), client=client)
    assert await channel.resolve("Ayşe") is None
    assert client.listed == 2

    client.people.append(AYSE)
    assert await channel.resolve("Ayşe") == AYSE


async def test_a_book_contact_without_username_or_phone_is_looked_up_by_their_full_name() -> None:
    book = AddressBook([Contact(name="Ayşe Demir", aliases=("ablam",))])
    client = FakeClient(AYSE)

    assert await Telegram(book, client=client).resolve("ablam") == AYSE


async def test_a_guessed_person_in_the_book_or_the_list_is_named_back_not_sent_to() -> None:
    """The same rule as WhatsApp's (2026-09-15): a case ending or a near
    miss is answered with the full name, and the model calls again."""
    client = FakeClient(AHMET, AYSE)
    channel = Telegram(BOOK, client=client)

    with pytest.raises(NoRecipientError, match="Ahmet Yılmaz"):
        await channel.resolve("ahmede")  # the book's Ahmet, guessed
    with pytest.raises(NoRecipientError, match="Ayşe Demir"):
        await channel.resolve("ayseye")  # the list's Ayşe, guessed
    assert await channel.resolve("Ayşe Demir") == AYSE


async def test_two_people_in_the_list_with_one_first_name_are_nobody() -> None:
    client = FakeClient(AHMET, Person(id=5, name="Ahmet Kaya", username="", phone=""))

    assert await Telegram(AddressBook(), client=client).resolve("Ahmet") is None


async def test_closest_merges_the_book_and_the_list_the_book_first() -> None:
    client = FakeClient(AHMET, AYSE)
    channel = Telegram(BOOK, client=client)
    await channel.resolve("Ayşe")  # fetches the list

    near = await channel.closest("ahmet")

    assert near[0] == "Ahmet Yılmaz"
    assert len(near) == len(set(near))


async def test_a_list_that_cannot_be_fetched_is_a_sentence() -> None:
    client = FakeClient(failing=TelegramError("network"))

    with pytest.raises(NoRecipientError, match="could not be reached"):
        await Telegram(AddressBook(), client=client).resolve("Ayşe")


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------


async def test_sent_is_said_with_the_persons_name() -> None:
    client = FakeClient(AHMET)

    said = await Telegram(BOOK, client=client).send(AHMET, "yarın geliyorum")

    assert said == "Sent to Ahmet Yılmaz on Telegram."
    assert client.sent == [("Ahmet Yılmaz", "yarın geliyorum")]


@pytest.mark.parametrize(
    ("failure", "said"),
    [
        (TelegramError("flood_wait", seconds=30), "wait 30 seconds"),
        (TelegramError("peer_flood"), "people it does not know"),
        (TelegramError("privacy"), "Ahmet Yılmaz's privacy settings"),
        (TelegramError("network"), "could not be reached"),
        (TelegramError("not_authorized"), "live-assistant telegram login"),
        (TelegramError("rpc", reason="CHAT_WRITE_FORBIDDEN"), "CHAT_WRITE_FORBIDDEN"),
    ],
)
async def test_every_kind_of_refusal_is_its_own_sentence(failure: TelegramError, said: str) -> None:
    client = FakeClient(AHMET, failing=failure)

    answer = await Telegram(BOOK, client=client).send(AHMET, "hi")

    assert said in answer
    assert set(SAID) >= {"flood_wait", "peer_flood", "privacy", "network", "not_authorized"}


# --------------------------------------------------------------------------
# Logging in
# --------------------------------------------------------------------------


class ScriptedPrompter:
    def __init__(self, **answers: str | None) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def ask(self, key: str) -> str | None:
        self.asked.append(key)
        return self.answers.get(key)

    async def secret(self, key: str) -> str | None:
        self.asked.append(key)
        return self.answers.get(key)


class FakeLoginClient:
    built: ClassVar[list[tuple[int, str]]] = []

    def __init__(
        self, api_id: int, api_hash: str, *, password: bool = False, refuse: str = ""
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self._password = password
        self._refuse = refuse
        self.calls: list[str] = []
        FakeLoginClient.built.append((api_id, api_hash))

    async def connect(self) -> None:
        self.calls.append("connect")

    async def send_code(self, phone: str) -> str:
        self.calls.append(f"send_code:{phone}")
        return "hash-1"

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        self.calls.append(f"sign_in:{phone}:{code}:{code_hash}")
        if self._refuse:
            raise LoginError(self._refuse)
        if self._password:
            raise PasswordNeededError
        return "Emre"

    async def sign_in_with_password(self, password: str) -> str:
        self.calls.append(f"password:{password}")
        return "Emre"

    def session_string(self) -> str:
        return "1BVtsOK...session"

    async def disconnect(self) -> None:
        self.calls.append("disconnect")


@pytest.fixture(autouse=True)
def _no_clients_left_over() -> None:
    FakeLoginClient.built.clear()


class Factory:
    """Builds fake login clients the way `login` asks, and keeps them."""

    def __init__(self, *, password: bool = False, refuse: str = "") -> None:
        self._password = password
        self._refuse = refuse
        self.made: list[FakeLoginClient] = []

    def __call__(self, api_id: int, api_hash: str) -> FakeLoginClient:
        client = FakeLoginClient(api_id, api_hash, password=self._password, refuse=self._refuse)
        self.made.append(client)
        return client


def factory(*, password: bool = False, refuse: str = "") -> Factory:
    return Factory(password=password, refuse=refuse)


async def test_the_questions_come_in_order_and_the_login_carries_what_to_store() -> None:
    prompter = ScriptedPrompter(
        telegram_api_id="123456",
        telegram_api_hash="abcdef",
        telegram_phone="+90 532 000 00 00",
        telegram_code="12345",
    )
    build = factory()

    got = await login(prompter, client_factory=build)

    assert got == Login(api_id=123456, api_hash="abcdef", session="1BVtsOK...session", name="Emre")
    assert prompter.asked == [
        "telegram_api_id",
        "telegram_api_hash",
        "telegram_phone",
        "telegram_code",
    ]
    assert build.made[0].calls == [
        "connect",
        "send_code:+90 532 000 00 00",
        "sign_in:+90 532 000 00 00:12345:hash-1",
        "disconnect",
    ]


async def test_a_stored_id_and_hash_are_not_asked_again() -> None:
    prompter = ScriptedPrompter(telegram_phone="+90...", telegram_code="1")
    build = factory()

    got = await login(prompter, api_id=7, api_hash="h", client_factory=build)

    assert got is not None and (got.api_id, got.api_hash) == (7, "h")
    assert prompter.asked == ["telegram_phone", "telegram_code"]
    assert FakeLoginClient.built == [(7, "h")]


async def test_the_password_is_asked_only_when_the_account_has_one() -> None:
    prompter = ScriptedPrompter(
        telegram_phone="+90...",
        telegram_code="1",
        telegram_password="pw",  # noqa: S106  # a test's stand-in, not a secret
    )
    build = factory(password=True)

    got = await login(prompter, api_id=7, api_hash="h", client_factory=build)

    assert got is not None and got.name == "Emre"
    assert prompter.asked[-1] == "telegram_password"
    assert build.made[0].calls[-2:] == ["password:pw", "disconnect"]


async def test_a_wrong_code_is_telegrams_reason_and_the_client_is_still_disconnected() -> None:
    prompter = ScriptedPrompter(telegram_phone="+90...", telegram_code="0")
    build = factory(refuse="The phone code entered was invalid (caused by SignInRequest)")

    with pytest.raises(LoginError, match="phone code entered was invalid"):
        await login(prompter, api_id=7, api_hash="h", client_factory=build)
    assert build.made[0].calls[-1] == "disconnect"


async def test_an_id_that_is_not_a_number_is_refused_before_anything_connects() -> None:
    prompter = ScriptedPrompter(telegram_api_id="abc")
    build = factory()

    with pytest.raises(LoginError, match="must be a number"):
        await login(prompter, client_factory=build)
    assert build.made == []


@pytest.mark.parametrize("walk_away_at", ["telegram_api_id", "telegram_phone", "telegram_code"])
async def test_walking_away_from_any_question_is_none_and_nothing_stored(walk_away_at: str) -> None:
    answers = {
        "telegram_api_id": "1",
        "telegram_api_hash": "h",
        "telegram_phone": "+90...",
        "telegram_code": "1",
    }
    answers[walk_away_at] = None  # type: ignore[assignment]
    build = factory()

    assert await login(ScriptedPrompter(**answers), client_factory=build) is None


def test_the_sentences_a_translator_has_to_write_are_the_ones_the_login_asks() -> None:
    assert set(TEXT) == {
        "telegram_api_id",
        "telegram_api_hash",
        "telegram_phone",
        "telegram_code",
        "telegram_password",
        "telegram_logged_in",
        "telegram_login_failed",
    }
    assert "{name}" in TEXT["telegram_logged_in"]
    assert "{reason}" in TEXT["telegram_login_failed"]


def test_the_real_client_is_built_without_telethon_being_imported() -> None:
    """`telethon` is imported inside the methods, like WinRT in
    `media/now_playing.py`: a broken install must not stop the assistant."""
    import sys

    from assistant.messaging.telegram import TelethonClient

    for name in [module for module in sys.modules if module.startswith("telethon")]:
        del sys.modules[name]
    TelethonClient(1, "h", "")

    assert not any(module.startswith("telethon") for module in sys.modules)
