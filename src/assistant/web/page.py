"""A web page, fetched and reduced to the words a person would read
(design.md 3.1, 17 Sep 2026).

"Şu siteyi özetle" is the first example task of section 2, and it is two
jobs: getting the page, and getting the page *out of the way*. A news
article is a few hundred words inside forty kilobytes of navigation, cookie
banners, scripts and footers, and every one of those kilobytes would be
paid for as tokens and read out as nothing. So the page is stripped before
the model sees it, in three steps that are each a rule rather than a
heuristic: what a browser would not show is removed (`script`, `style`,
anything hidden), what is furniture is removed (`nav`, `header`, `footer`,
`aside`), and what the page itself marks as its content (`main`,
`article`) is taken alone when it is marked.

**The page is cut, and the cut is said.** `MAX_PAGE_CHARS` is about three
to four thousand tokens - the design's thirty thousand would cost more per
turn than a day of ordinary use with Flash-Lite - and a page longer than
that is handed over cut, with a sentence saying so, so that the model can
tell the user rather than pretend to have read the end. `focus` moves the
paragraphs that mention what the user asked about to the front before the
cut, which is how a question about one section of a long page is answered
from that section.

**Only what a browser could show as text is read.** A PDF, an image, a
download: each is refused in a sentence rather than fed to the parser as
noise. The body is read at most `MAX_BYTES` deep, so that a page that never
ends does not end the turn.

Nothing here decides what the words mean, and nothing here marks them as
foreign - that is `tools/untrusted.py`, applied by the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from selectolax.parser import HTMLParser, Node

from assistant.store.normalize import normalize_search

__all__ = [
    "FETCH_SECONDS",
    "MAX_BYTES",
    "MAX_PAGE_CHARS",
    "Page",
    "PageError",
    "PageReader",
    "address",
    "focused",
    "read_html",
]

# How long a page may take to start answering and to be read. Longer than
# the weather's eight: a page is one request against the whole of the web.
FETCH_SECONDS = 10.0

# How much of a page is read before the rest is left unread. Two megabytes
# is every article there is and no video.
MAX_BYTES = 2_000_000

# How many characters of the page reach the model: about three to four
# thousand tokens, a few cents a turn on the priced models and the whole of
# most articles.
MAX_PAGE_CHARS = 12_000

# The user agent a page is asked with. A bare library name is answered with
# a 403 by many sites; a browser's shape is answered with the page.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0 Safari/537.36 assistant/0.4"
)
ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"

# What a browser would not show, or shows as furniture rather than as the
# page. Removed with everything inside them.
_DROPPED = (
    "script, style, noscript, template, iframe, svg, canvas, "
    "nav, header, footer, aside, button, input, select, textarea, "
    "[hidden], [aria-hidden='true']"
)

# Where the words of the page are, when the page says so.
_CONTENT = ("main", "[role='main']")

# Elements that end a line when read aloud. A line break is written around
# each so that inline markup - a link in the middle of a sentence - stays in
# its sentence, and a paragraph ends where the page ends it.
_BLOCKS = frozenset(
    {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "tr", "br", "hr",
        "div", "section", "article", "blockquote", "pre", "figcaption", "table",
        "ul", "ol", "main", "body",
    }
)  # fmt: skip

# What the parser calls a text node.
_TEXT = "-text"

_SPACES = re.compile(r"[ \t\r\f\v ]+")
_WORD = re.compile(r"\w+")

# A word of the focus shorter than this matches too much to move anything.
MIN_FOCUS_WORD = 3


class PageError(Exception):
    """A page that could not be read, worded so the tool can pass it on."""


@dataclass(frozen=True, slots=True)
class Page:
    """What was read: where it came from in the end, what it called itself,
    and its words, one paragraph per line."""

    url: str
    title: str
    text: str


class PageReader:
    """Fetches pages over one kept connection, the way `OpenMeteo` asks
    for the weather."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        seconds: float = FETCH_SECONDS,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds
        self._max_bytes = max_bytes

    async def fetch(self, url: str) -> Page:
        """The page at `url`, reduced to its words.

        Redirects are followed, and the address the page was finally read
        from is the one reported. Anything that is not text - by the
        server's own account - is refused in a sentence.
        """
        target = address(url)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._seconds, follow_redirects=True)

        try:
            async with self._client.stream(
                "GET",
                target,
                headers={"User-Agent": USER_AGENT, "Accept": ACCEPT},
                timeout=self._seconds,
                follow_redirects=True,
            ) as response:
                if response.status_code >= 400:
                    raise PageError(
                        f"The server answered {response.status_code} for {target}; "
                        "the page could not be read."
                    )
                kind = response.headers.get("content-type", "").partition(";")[0].strip()
                if not _readable(kind):
                    raise PageError(
                        f"{target} is {kind or 'not a text page'}, which cannot be read as text."
                    )
                body = await _read(response, self._max_bytes)
                final = str(response.url)
        except httpx.TimeoutException as failure:
            raise PageError(
                f"The page did not answer within {self._seconds:.0f} seconds."
            ) from failure
        except httpx.HTTPError as failure:
            raise PageError(f"The page could not be reached: {failure}") from failure

        if kind == "text/plain":
            return Page(url=final, title="", text=_lines(body.decode("utf-8", errors="replace")))
        title, text = read_html(body)
        return Page(url=final, title=title, text=text)

    async def aclose(self) -> None:
        """Gives back the connection, at shutdown. A borrowed client is left alone."""
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None


def address(url: str) -> str:
    """`url` as something that can be fetched: `https://` in front of a bare
    host, and anything that is not the web refused."""
    wanted = " ".join(url.split())
    if not wanted:
        raise PageError("No address was given.")
    if "://" not in wanted:
        wanted = f"https://{wanted}"
    parts = urlsplit(wanted)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise PageError(f"Only http and https addresses can be read; {url!r} is neither.")
    return urlunsplit(parts)


def read_html(html: bytes | str) -> tuple[str, str]:
    """The title of a page and its words, one paragraph per line.

    Everything a browser would not show is dropped first, then the
    furniture around the content; what the page marks as its content is
    read alone when it marks any. The bytes are decoded by the parser from
    the page's own `meta` charset, which is the one thing about a page's
    encoding that is reliably true.
    """
    tree = HTMLParser(html)
    named = tree.css_first("title")
    title = " ".join((named.text() if named is not None else "").split())

    for node in tree.css(_DROPPED):
        node.decompose()
    for node in tree.css("[style]"):
        if _hidden_by_style(node):
            node.decompose()

    content = next((found for selector in _CONTENT if (found := tree.css_first(selector))), None)
    if content is None:
        articles = tree.css("article")
        content = articles[0] if len(articles) == 1 else tree.body
    if content is None:
        return title, ""
    return title, _lines(_words_of(content))


def focused(text: str, focus: str) -> tuple[str, bool]:
    """`text` with the paragraphs that mention `focus` moved to the front,
    and whether any did.

    Read the way search reads (`store/normalize.py`): "ödeme" finds
    "ÖDEME" and "odeme". Words shorter than `MIN_FOCUS_WORD` are not
    looked for, since "ve" is in every paragraph.
    """
    wanted = [
        word for word in _WORD.findall(normalize_search(focus)) if len(word) >= MIN_FOCUS_WORD
    ]
    if not wanted or not text:
        return text, False

    first: list[str] = []
    rest: list[str] = []
    for line in text.split("\n"):
        folded = normalize_search(line)
        (first if any(word in folded for word in wanted) else rest).append(line)
    if not first:
        return text, False
    return "\n".join([*first, *rest]), True


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------


def _readable(kind: str) -> bool:
    """Whether `content-type` names something the parser can make words of.
    No header at all is read as HTML, which is what a server that says
    nothing almost always sends."""
    return kind in ("", "text/html", "application/xhtml+xml", "text/plain")


async def _read(response: httpx.Response, limit: int) -> bytes:
    """The body, up to `limit` bytes; what comes after is never fetched."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit]


def _words_of(root: Node) -> str:
    """The text under `root` in reading order, with a line break around
    every block element.

    Walked with an explicit stack rather than by mutating the tree: a
    newline inserted after a block was found to end the walk early on
    ordinary pages, and a page a thousand elements deep would end a
    recursive one.
    """
    out: list[str] = []
    stack: list[Node | str] = list(reversed(list(root.iter(include_text=True))))
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif item.tag == _TEXT:
            out.append(item.text_content or "")
        elif item.tag in _BLOCKS:
            out.append("\n")
            stack.append("\n")
            stack.extend(reversed(list(item.iter(include_text=True))))
        else:
            stack.extend(reversed(list(item.iter(include_text=True))))
    return "".join(out)


def _hidden_by_style(node: Node) -> bool:
    style = (node.attributes.get("style") or "").replace(" ", "").casefold()
    return "display:none" in style or "visibility:hidden" in style


def _lines(text: str) -> str:
    """One paragraph per line, one space between words, nothing blank."""
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)
