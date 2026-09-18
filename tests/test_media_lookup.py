"""Looking a song, a video and a Spotify search up, without a network.

Nothing here reaches YouTube. The results page arrives through an
`httpx.MockTransport`, the music catalogue is a stand-in for `ytmusicapi`, and
the `no_network` fixture below turns a forgotten stub into a failed test
rather than a slow one that quietly asks Google.

What is actually being proved is one claim, from four sides: **the model never
supplies an identifier**. A `videoId` is either read out of a real answer or
the lookup fails in words - it is never guessed, never half-built, and never
turned into a `watch?v=` address that would open "This video isn't available
anymore".
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from assistant.media import youtube
from assistant.media.deezer import Deezer
from assistant.media.spotify import APP_HOME, WEB_HOME, Spotify, app_installed
from assistant.media.track import SearchError
from assistant.media.youtube import YouTube, YouTubeMusic

MAIN = threading.current_thread()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgot its stub fails here rather than on YouTube."""

    async def refuse(*_: object, **__: object) -> httpx.Response:
        raise AssertionError("a test reached the real network")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(
        youtube,
        "_ytmusic",
        lambda: (_ for _ in ()).throw(AssertionError("a test built a real music catalogue")),
    )


# --------------------------------------------------------------------------
# Building the answers YouTube would have given
# --------------------------------------------------------------------------


def renderer(video_id: str, title: str, owner: str = "") -> dict[str, Any]:
    """One row of a results page, in YouTube's own shape."""
    row: dict[str, Any] = {"videoId": video_id, "title": {"runs": [{"text": title}]}}
    if owner:
        row["ownerText"] = {"runs": [{"text": owner}]}
    return row


def results_page(*rows: dict[str, Any], data: object | None = None) -> str:
    """A page with `ytInitialData` in it, nested the way YouTube nests it."""
    blob = (
        data
        if data is not None
        else {
            "contents": {
                "twoColumnSearchResultsRenderer": {
                    "primaryContents": {
                        "sectionListRenderer": {
                            "contents": [
                                {
                                    "itemSectionRenderer": {
                                        "contents": [{"videoRenderer": row} for row in rows]
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    )
    return (
        "<!DOCTYPE html><html><body><script>"
        f"var ytInitialData = {json.dumps(blob)};</script>"
        "<script>var other = 1;</script></body></html>"
    )


class Served:
    """One prepared answer, and what was asked to get it."""

    def __init__(self, page: str = "", status: int = 200, raises: Exception | None = None) -> None:
        self.page = page
        self.status = status
        self.raises = raises
        self.requests: list[httpx.Request] = []
        self.threads: list[threading.Thread] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.threads.append(threading.current_thread())
        if self.raises is not None:
            raise self.raises
        return httpx.Response(self.status, text=self.page)

    @property
    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(str(self.requests[-1].url)).query)


class Catalogue:
    """What `ytmusicapi` would have answered, and what it was asked."""

    def __init__(
        self,
        songs: list[dict[str, Any]] | None = None,
        home: list[dict[str, Any]] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.songs = songs or []
        self.home = home or []
        self.raises = raises
        self.asked: list[str] = []
        self.threads: list[threading.Thread] = []

    def search(self, query: str, *_: object, **__: object) -> list[dict[str, Any]]:
        self.asked.append(query)
        self.threads.append(threading.current_thread())
        if self.raises is not None:
            raise self.raises
        return self.songs

    def get_home(self, *_: object, **__: object) -> list[dict[str, Any]]:
        self.threads.append(threading.current_thread())
        if self.raises is not None:
            raise self.raises
        return self.home


def song(video_id: str, title: str, artist: str = "") -> dict[str, Any]:
    return {"videoId": video_id, "title": title, "artists": [{"name": artist}] if artist else []}


def music(catalogue: Catalogue) -> YouTubeMusic:
    return YouTubeMusic(catalogue=lambda: catalogue)


# --------------------------------------------------------------------------
# YouTube: the video the user described
# --------------------------------------------------------------------------


async def test_the_words_reach_the_search_box_unchanged() -> None:
    served = Served(results_page(renderer("outny_anbdo", "izle öğren")))

    await YouTube(client=served.client()).first_video("skill bilmeyen yakında işsiz kalır")

    assert served.query["search_query"] == ["skill bilmeyen yakında işsiz kalır"]


async def test_no_language_or_country_is_sent_with_the_search() -> None:
    """Invariant 4: nothing in this code carries a locale (section 3.12)."""
    served = Served(results_page(renderer("outny_anbdo", "izle öğren")))

    await YouTube(client=served.client()).first_video("bir video")

    assert "hl" not in served.query
    assert "gl" not in served.query
    assert "accept-language" not in served.requests[-1].headers


async def test_the_first_ordinary_result_is_the_one_opened() -> None:
    served = Served(
        results_page(
            renderer("outny_anbdo", '"Skill bilmeyen yakında işsiz kalır" - izle öğren', "İzle"),
            renderer("bbbbbbbbbbb", "Something else"),
        )
    )

    found = await YouTube(client=served.client()).first_video("skill bilmeyen")

    assert found.target == "https://www.youtube.com/watch?v=outny_anbdo"
    assert found.title == '"Skill bilmeyen yakında işsiz kalır" - izle öğren'
    assert found.artist == "İzle"


async def test_a_result_without_a_real_identifier_is_skipped_not_opened() -> None:
    """A short, empty or malformed id would build an address that fails."""
    served = Served(
        results_page(
            renderer("too-short", "A shelf header"),
            renderer("", "No identifier at all"),
            renderer("goodgoodgoo", "The real one"),
        )
    )

    found = await YouTube(client=served.client()).first_video("anything")

    assert found.target == "https://www.youtube.com/watch?v=goodgoodgoo"


async def test_a_title_written_as_simple_text_is_read_too() -> None:
    page = results_page({"videoId": "goodgoodgoo", "title": {"simpleText": "Plain title"}})

    found = await YouTube(client=Served(page).client()).first_video("anything")

    assert found.title == "Plain title"


async def test_a_page_with_no_results_is_reported_rather_than_guessed() -> None:
    served = Served(results_page())

    with pytest.raises(SearchError, match="no video"):
        await YouTube(client=served.client()).first_video("nothing at all")


async def test_a_consent_wall_is_reported_rather_than_parsed() -> None:
    served = Served("<html><body>Before you continue to YouTube</body></html>")

    with pytest.raises(SearchError, match="Nothing was opened"):
        await YouTube(client=served.client()).first_video("anything")


async def test_a_blob_that_is_not_json_is_reported_rather_than_half_read() -> None:
    served = Served("<script>var ytInitialData = {not json at all;</script>")

    with pytest.raises(SearchError, match="could not be read"):
        await YouTube(client=served.client()).first_video("anything")


async def test_a_refused_page_is_a_failure_the_tool_can_read() -> None:
    served = Served(status=429)

    with pytest.raises(SearchError, match="429"):
        await YouTube(client=served.client()).first_video("anything")


async def test_a_connection_that_never_answers_is_a_failure_not_a_hang() -> None:
    served = Served(raises=httpx.ConnectTimeout("too slow"))

    with pytest.raises(SearchError, match="did not answer within"):
        await YouTube(client=served.client(), seconds=3).first_video("anything")


async def test_a_line_that_is_down_becomes_a_sentence_not_a_traceback() -> None:
    served = Served(raises=httpx.ConnectError("no route"))

    with pytest.raises(SearchError, match="could not be reached"):
        await YouTube(client=served.client()).first_video("anything")


async def test_the_page_is_parsed_without_the_event_loop() -> None:
    """A megabyte of JSON on the loop would stall audio (section 3.1 rule 4)."""
    seen: list[threading.Thread] = []
    real = youtube._first_video

    def watched(page: str, words: str) -> Any:
        seen.append(threading.current_thread())
        return real(page, words)

    served = Served(results_page(renderer("goodgoodgoo", "Title")))
    original, youtube._first_video = youtube._first_video, watched
    try:
        await YouTube(client=served.client()).first_video("anything")
    finally:
        youtube._first_video = original

    assert seen and seen[0] is not MAIN


async def test_one_connection_serves_every_search_of_a_session() -> None:
    """A fresh handshake per search measured half a second of a spoken turn."""
    built: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    def counted(**options: Any) -> httpx.AsyncClient:
        client = real(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=page)))
        built.append(client)
        return client

    page = results_page(renderer("goodgoodgoo", "Title"))
    service = YouTube()
    original, httpx.AsyncClient = httpx.AsyncClient, counted  # type: ignore[assignment, misc]
    try:
        await service.first_video("one")
        await service.first_video("two")
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]

    assert len(built) == 1
    await service.aclose()
    assert built[0].is_closed


async def test_a_client_that_was_handed_in_is_not_closed_underneath_its_owner() -> None:
    served = Served(results_page(renderer("goodgoodgoo", "Title")))
    client = served.client()

    service = YouTube(client=client)
    await service.first_video("anything")
    await service.aclose()

    assert not client.is_closed


# --------------------------------------------------------------------------
# YouTube Music: the song
# --------------------------------------------------------------------------


async def test_a_song_is_addressed_to_youtube_music_not_to_youtube() -> None:
    """The address is the whole point: this one plays, youtube.com does not."""
    catalogue = Catalogue([song("UXK9s54VmxQ", "Kumralım", "Yaşar")])

    found = await music(catalogue).song("kumralım")

    assert found.target == "https://music.youtube.com/watch?v=UXK9s54VmxQ"
    assert found.name == "Yaşar - Kumralım"
    assert catalogue.asked == ["kumralım"]


async def test_the_catalogue_keeps_its_own_order_when_nothing_matches_better() -> None:
    catalogue = Catalogue(
        [song("aaaaaaaaaaa", "First", "Someone"), song("bbbbbbbbbbb", "Second", "Another")]
    )

    found = await music(catalogue).song("a song nobody named an artist for")

    assert found.target.endswith("aaaaaaaaaaa")


async def test_the_named_artist_beats_a_cover_listed_above_them() -> None:
    catalogue = Catalogue(
        [
            song("ccccccccccc", "Kumralım", "A Cover Band"),
            song("ddddddddddd", "Kumralım", "Someone Else"),
            song("UXK9s54VmxQ", "Kumralım", "Yaşar"),
        ]
    )

    found = await music(catalogue).song("yaşar kumralım")

    assert found.target.endswith("UXK9s54VmxQ")


async def test_an_artist_whose_name_is_inside_a_longer_word_does_not_count() -> None:
    catalogue = Catalogue(
        [song("aaaaaaaaaaa", "First", "Someone"), song("bbbbbbbbbbb", "Second", "ya")]
    )

    found = await music(catalogue).song("yaşar kumralım")

    assert found.target.endswith("aaaaaaaaaaa")


async def test_a_song_result_with_no_usable_identifier_is_never_played() -> None:
    catalogue = Catalogue([song("short", "Bad"), song("", "Worse"), song("goodgoodgoo", "Good")])

    found = await music(catalogue).song("anything")

    assert found.target.endswith("goodgoodgoo")


async def test_a_search_with_no_playable_song_is_not_turned_into_one() -> None:
    with pytest.raises(SearchError, match="no song"):
        await music(Catalogue([])).song("something nobody recorded")


async def test_music_with_nothing_named_takes_the_front_page_first_song() -> None:
    catalogue = Catalogue(
        home=[
            {"title": "Listen again", "contents": [{"playlistId": "PL123", "title": "A playlist"}]},
            {"title": "Quick picks", "contents": [song("UXK9s54VmxQ", "Kumralım", "Yaşar")]},
        ]
    )

    found = await music(catalogue).anything()

    assert found.target.endswith("UXK9s54VmxQ")


async def test_a_front_page_with_nothing_playable_asks_rather_than_guesses() -> None:
    with pytest.raises(SearchError, match="front page"):
        await music(Catalogue(home=[{"title": "Empty", "contents": []}])).anything()


async def test_the_music_client_is_asked_off_the_event_loop() -> None:
    """`ytmusicapi` is synchronous and talks to the network (rule 4)."""
    catalogue = Catalogue([song("goodgoodgoo", "Good")])

    await music(catalogue).song("anything")

    assert catalogue.threads and catalogue.threads[0] is not MAIN


async def test_one_catalogue_serves_every_search_of_a_session() -> None:
    built: list[int] = []
    catalogue = Catalogue([song("goodgoodgoo", "Good")])

    def build() -> Catalogue:
        built.append(1)
        return catalogue

    service = YouTubeMusic(catalogue=build)

    await service.song("one")
    await service.song("two")

    assert len(built) == 1


async def test_a_catalogue_that_raises_becomes_a_sentence_not_a_traceback() -> None:
    with pytest.raises(SearchError, match="could not be searched"):
        await music(Catalogue(raises=RuntimeError("the API moved"))).song("anything")


async def test_a_cancelled_search_is_not_reported_as_a_failed_one() -> None:
    """A turn the user stopped is not a search that went wrong."""
    with pytest.raises(asyncio.CancelledError):
        await music(Catalogue(raises=asyncio.CancelledError())).song("anything")


# --------------------------------------------------------------------------
# Deezer: the recording's ISRC, which is the one thing Spotify's search
# understands without a key
# --------------------------------------------------------------------------


class DeezerServed:
    """Deezer's two answers - the search, then the track - and what was asked."""

    def __init__(
        self,
        search: object = None,
        track: object = None,
        status: int = 200,
        raises: Exception | None = None,
    ) -> None:
        self.search = search if search is not None else {"data": [], "total": 0}
        self.track = track
        self.status = status
        self.raises = raises
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        body = self.track if request.url.path.startswith("/track/") else self.search
        return httpx.Response(self.status, json=body)

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    @property
    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(str(self.requests[0].url)).query)


def hit(track_id: int, title: str, artist: str) -> dict[str, Any]:
    """One row of a Deezer search: no ISRC yet, that takes the second call."""
    return {"id": track_id, "type": "track", "title": title, "artist": {"name": artist}}


def recording(isrc: str, title: str, artist: str) -> dict[str, Any]:
    """The track itself, the way `GET /track/<id>` answers."""
    return {"id": 1, "type": "track", "isrc": isrc, "title": title, "artist": {"name": artist}}


KUMRALIM_SEARCH = {"data": [hit(3135556, "Kumralım", "Yaşar")], "total": 1}
KUMRALIM_TRACK = recording("TR2240596102", "Kumralım", "Yaşar")


async def test_the_words_are_searched_on_deezer_unchanged() -> None:
    served = DeezerServed(KUMRALIM_SEARCH, KUMRALIM_TRACK)

    await Deezer(client=served.client()).recording("yaşar kumralım")

    assert served.query["q"] == ["yaşar kumralım"]


async def test_the_first_hit_s_isrc_names_the_recording() -> None:
    served = DeezerServed(KUMRALIM_SEARCH, KUMRALIM_TRACK)

    found = await Deezer(client=served.client()).recording("kumralım")

    assert found.isrc == "TR2240596102"
    assert found.name == "Yaşar - Kumralım"
    assert served.paths == ["/search", "/track/3135556"]


async def test_deezer_s_own_spelling_of_the_name_is_what_is_reported() -> None:
    """The user said "manga"; the answer names the band the way Deezer does."""
    served = DeezerServed(
        {"data": [hit(7, "Bir Kadın Çizeceksin", "maNga")]},
        recording("TRA160400089", "Bir Kadın Çizeceksin", "maNga"),
    )

    found = await Deezer(client=served.client()).recording("manga bir kadın çizeceksin")

    assert found.name == "maNga - Bir Kadın Çizeceksin"


async def test_no_language_or_country_is_sent_to_deezer() -> None:
    """Invariant 4: nothing in this code carries a locale (section 3.12)."""
    served = DeezerServed(KUMRALIM_SEARCH, KUMRALIM_TRACK)

    await Deezer(client=served.client()).recording("kumralım")

    assert "accept-language" not in served.requests[0].headers
    assert set(served.query) == {"q", "limit"}


async def test_no_hit_is_a_sentence_and_no_second_request() -> None:
    served = DeezerServed({"data": [], "total": 0})

    with pytest.raises(SearchError, match="no recording"):
        await Deezer(client=served.client()).recording("something nobody recorded")

    assert served.paths == ["/search"]


async def test_deezer_s_own_error_object_is_a_sentence_not_a_key_error() -> None:
    """Deezer answers a bad request with HTTP 200 and an `error` object."""
    served = DeezerServed({"error": {"type": "DataException", "message": "no data", "code": 800}})

    with pytest.raises(SearchError, match="Deezer"):
        await Deezer(client=served.client()).recording("anything")


async def test_a_hit_without_an_isrc_is_reported_rather_than_searched_for_with_nothing() -> None:
    served = DeezerServed(KUMRALIM_SEARCH, recording("", "Kumralım", "Yaşar"))

    with pytest.raises(SearchError, match="ISRC"):
        await Deezer(client=served.client()).recording("kumralım")


async def test_an_isrc_that_is_not_one_is_refused_not_passed_on() -> None:
    """Whatever ends up after `isrc:` in a Spotify search has to be an ISRC."""
    served = DeezerServed(KUMRALIM_SEARCH, recording("not an isrc", "Kumralım", "Yaşar"))

    with pytest.raises(SearchError, match="ISRC"):
        await Deezer(client=served.client()).recording("kumralım")


async def test_an_isrc_written_with_dashes_is_read_the_way_spotify_wants_it() -> None:
    served = DeezerServed(KUMRALIM_SEARCH, recording("tr-224-05-96102", "Kumralım", "Yaşar"))

    found = await Deezer(client=served.client()).recording("kumralım")

    assert found.isrc == "TR2240596102"


async def test_a_refused_request_is_a_failure_the_tool_can_read() -> None:
    served = DeezerServed(status=503)

    with pytest.raises(SearchError, match="503"):
        await Deezer(client=served.client()).recording("anything")


async def test_a_deezer_that_never_answers_is_a_failure_not_a_hang() -> None:
    served = DeezerServed(raises=httpx.ReadTimeout("slow"))

    with pytest.raises(SearchError, match="seconds"):
        await Deezer(client=served.client(), seconds=2.0).recording("anything")


async def test_a_deezer_that_cannot_be_reached_becomes_a_sentence() -> None:
    served = DeezerServed(raises=httpx.ConnectError("no route"))

    with pytest.raises(SearchError, match="could not be reached"):
        await Deezer(client=served.client()).recording("anything")


async def test_an_answer_that_is_not_json_is_a_sentence_not_a_traceback() -> None:
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(html))

    with pytest.raises(SearchError, match="could not be read"):
        await Deezer(client=client).recording("anything")


async def test_a_deezer_client_handed_in_is_not_closed_underneath_its_owner() -> None:
    served = DeezerServed(KUMRALIM_SEARCH, KUMRALIM_TRACK)
    client = served.client()

    service = Deezer(client=client)
    await service.recording("kumralım")
    await service.aclose()

    assert not client.is_closed


# --------------------------------------------------------------------------
# Spotify: a search, and never a claim that something started
# --------------------------------------------------------------------------


def test_the_application_is_installed_when_windows_names_one_for_the_scheme() -> None:
    """The Store build registers no `shell\\open\\command` for `spotify:` -
    measured 2026-09-14 - so the question goes to the association Windows
    itself resolves, which answers with the application's name."""
    assert app_installed(named=lambda scheme: "Spotify" if scheme == "spotify" else None)


def test_the_application_is_not_installed_when_nothing_answers_the_scheme() -> None:
    assert not app_installed(named=lambda scheme: None)


def test_an_isrc_opens_the_one_exact_recording_in_the_application() -> None:
    """`isrc:` is a filter Spotify's own search box understands."""
    exact = Spotify(installed=lambda: True).exact("TR2240596102")

    assert exact == "spotify:search:isrc%3ATR2240596102"


def test_without_the_application_the_exact_recording_is_searched_on_the_website() -> None:
    exact = Spotify(installed=lambda: False).exact("TR2240596102")

    assert exact == "https://open.spotify.com/search/isrc%3ATR2240596102"


def test_the_installed_application_is_handed_a_search_uri() -> None:
    assert Spotify(installed=lambda: True).search("Billie Jean") == "spotify:search:Billie%20Jean"


def test_without_the_application_the_website_takes_the_search() -> None:
    address = Spotify(installed=lambda: False).search("Billie Jean")

    assert address == "https://open.spotify.com/search/Billie%20Jean"


def test_a_search_escapes_what_the_user_said() -> None:
    """Unescaped, a slash or a Turkish letter is a different search."""
    escaped = Spotify(installed=lambda: True).search("Yaşar / Kumralım")

    assert " " not in escaped
    assert "/" not in escaped.removeprefix("spotify:search:")
    assert escaped.startswith("spotify:search:")


def test_opening_spotify_prefers_the_installed_application() -> None:
    assert Spotify(installed=lambda: True).home() == APP_HOME


def test_opening_spotify_without_the_application_uses_its_website() -> None:
    assert Spotify(installed=lambda: False).home() == WEB_HOME


def test_the_machine_is_asked_every_time_rather_than_once() -> None:
    """Spotify may be installed while the assistant is running."""
    answers = iter([False, True])
    spotify = Spotify(installed=lambda: next(answers))

    assert spotify.home() == WEB_HOME
    assert spotify.home() == APP_HOME
