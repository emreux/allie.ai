"""One kept connection for every quick-facts service, and one way to fail.

Built like the weather's (`tools/weather.py::OpenMeteo`): a client made on
first use and given back at shutdown, one attempt per question, and every
failure turned into a sentence the model can pass on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

__all__ = ["FACTS_SECONDS", "USER_AGENT", "FactsError", "Fetcher"]

# Long enough for a slow line, short enough that a spoken turn is not held.
FACTS_SECONDS = 8.0

# Nominatim-style services refuse an anonymous client; the others do not mind.
USER_AGENT = "allie (+https://github.com/emreux/allie.ai)"

Params = Mapping[str, str | int | float]


class FactsError(Exception):
    """A service that could not answer, worded so the tool can pass it on."""


class Fetcher:
    """GETs over one kept connection; JSON or text back, or `FactsError`."""

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, seconds: float = FACTS_SECONDS
    ) -> None:
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds

    async def json(self, url: str, params: Params | None = None, *, service: str) -> Any:
        response = await self._get(url, params, service)
        try:
            return response.json()
        except ValueError as failure:
            raise FactsError(f"{service} answered with something that is not JSON.") from failure

    async def text(self, url: str, params: Params | None = None, *, service: str) -> str:
        return (await self._get(url, params, service)).text

    async def aclose(self) -> None:
        """Gives back the connection, at shutdown. A borrowed client is left alone."""
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str, params: Params | None, service: str) -> httpx.Response:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._seconds, follow_redirects=True, headers={"User-Agent": USER_AGENT}
            )
        try:
            response = await self._client.get(url, params=params, timeout=self._seconds)
            response.raise_for_status()
        except httpx.TimeoutException as failure:
            raise FactsError(
                f"{service} did not answer within {self._seconds:.0f} seconds."
            ) from failure
        except httpx.HTTPError as failure:
            raise FactsError(
                f"{service} could not be reached ({type(failure).__name__})."
            ) from failure
        return response
