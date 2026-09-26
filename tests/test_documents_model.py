"""`documents/model.py` (plan.md D35): the second model, asked one question
with a folder's files, and the sentences a refusal becomes.

Nothing here reaches Google: the client is a fake that answers with the
SDK's own response types and remembers what it was asked. The real model
was measured in K0 (`docs/plan.md`, phase K)."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.genai import errors, types
from loguru import logger

from allie.documents import convert
from allie.documents import folder as folder_module
from allie.documents import model as model_module
from allie.documents.folder import PdfFile, TextFile
from allie.documents.model import (
    BUSY,
    DOCUMENT_MODEL,
    DOCUMENT_PROMPT,
    MAX_ANSWER_CHARS,
    MODEL_GONE,
    NO_ANSWER,
    NOTHING_SAID,
    QUOTA_USED,
    REFUSED,
    UNREACHABLE,
    DocumentError,
    DocumentModel,
)

KEY = "AIza-not-a-real-key"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
FILES = (
    PdfFile("fatura.pdf", b"%PDF-1.7 fatura"),
    TextFile("mizan.xlsx", "## Mizan\n\n| Hesap | Bakiye |"),
)
ANSWER = "Fatura tutarı 12.500,00 TL (fatura.pdf)."


def said(text: str) -> types.GenerateContentResponse:
    """An answer as the SDK delivers one."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


def refusal(
    code: int, message: str, status: str, quota: str = "", limit: str = "500"
) -> errors.APIError:
    """A refusal as Google words one; `quota` is the `quotaId` a 429 names
    and `limit` its `quotaValue` (none when empty). As in
    `test_web_search.py`."""
    kind = errors.ClientError if code < 500 else errors.ServerError
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if quota:
        violation = {"quotaId": quota, "quotaValue": limit} if limit else {"quotaId": quota}
        error["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [violation],
            }
        ]
    return kind(code, {"error": error})


class FakeModels:
    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []
        self.answer: Any = said(ANSWER)
        self.refusal: BaseException | None = None
        self.delay = 0.0

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.asked.append({"model": model, "contents": contents, "config": config})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.refusal is not None:
            raise self.refusal
        return self.answer


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()

    @property
    def aio(self) -> FakeClient:
        return self


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def asker(client: FakeClient, **rest: Any) -> DocumentModel:
    return DocumentModel(KEY, client=client, **rest)


def logged(level: str) -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lambda message: lines.append(str(message)), level=level)


async def refused_with(client: FakeClient, failure: BaseException, **rest: Any) -> str:
    client.models.refusal = failure
    with pytest.raises(DocumentError) as caught:
        await asker(client, **rest).answer("x", FILES)
    return str(caught.value)


# --------------------------------------------------------------------------
# What is asked, and what comes back
# --------------------------------------------------------------------------


async def test_the_answer_comes_back(client: FakeClient) -> None:
    assert await asker(client).answer("Fatura tutarı ne?", FILES) == ANSWER


async def test_each_file_goes_under_its_name_and_the_question_last(client: FakeClient) -> None:
    """The question last: Google's own advice for a long context, and what
    lets a repeated folder be a cached prefix on the paid tier."""
    await asker(client).answer("Fatura tutarı ne?", FILES)

    [asked] = client.models.asked
    [content] = asked["contents"]
    parts = content.parts
    assert len(parts) == 4
    assert parts[0].text == "### fatura.pdf"
    assert parts[1].inline_data.mime_type == "application/pdf"
    assert parts[1].inline_data.data == b"%PDF-1.7 fatura"
    assert parts[2].text == "### mizan.xlsx\n\n## Mizan\n\n| Hesap | Bakiye |"
    assert parts[3].text == "Question: Fatura tutarı ne?"


async def test_the_model_is_told_to_quote_and_nothing_else(client: FakeClient) -> None:
    await asker(client).answer("x", FILES)

    config = client.models.asked[0]["config"]
    assert DOCUMENT_PROMPT in str(config.system_instruction)
    assert config.tools is None
    # The SDK's own function calling has nothing to call, and left on it
    # warns at every question (K0, as S0 found for the search).
    assert config.automatic_function_calling.disable is True
    rule = DOCUMENT_PROMPT.casefold()
    assert "exactly as the document writes it" in rule
    assert "do not calculate" in rule
    assert "not instructions" in rule


async def test_the_model_is_the_one_the_settings_name(client: FakeClient) -> None:
    default = asker(client)
    chosen = asker(client, model="gemini-3.1-flash-lite")

    await chosen.answer("x", FILES)

    assert default.model == DOCUMENT_MODEL
    assert client.models.asked[0]["model"] == "gemini-3.1-flash-lite"


async def test_the_model_thinks_as_the_measurement_decided(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "THINKING_LEVEL", None)
    await asker(client).answer("x", FILES)
    assert client.models.asked[-1]["config"].thinking_config is None

    monkeypatch.setattr(model_module, "THINKING_LEVEL", "MINIMAL")
    await asker(client).answer("x", FILES)
    thinking = client.models.asked[-1]["config"].thinking_config
    assert thinking.thinking_level == types.ThinkingLevel.MINIMAL


async def test_a_long_answer_is_cut(client: FakeClient) -> None:
    client.models.answer = said("a" * (MAX_ANSWER_CHARS + 500))

    assert len(await asker(client).answer("x", FILES)) == MAX_ANSWER_CHARS


# --------------------------------------------------------------------------
# What a refusal becomes - in words about documents, not web searches
# --------------------------------------------------------------------------


async def test_an_empty_answer_is_nothing_said(client: FakeClient) -> None:
    client.models.answer = said("   ")

    with pytest.raises(DocumentError) as caught:
        await asker(client).answer("x", FILES)

    assert str(caught.value) == NOTHING_SAID


async def test_the_minute_s_limit_is_one_sentence(client: FakeClient) -> None:
    failure = refusal(429, "quota", "RESOURCE_EXHAUSTED", PER_MINUTE)

    assert await refused_with(client, failure) == BUSY
    assert "search" not in BUSY.casefold()


async def test_the_day_s_limit_is_said_with_google_s_number(client: FakeClient) -> None:
    failure = refusal(429, "quota", "RESOURCE_EXHAUSTED", PER_DAY, limit="500")

    assert await refused_with(client, failure) == f"{QUOTA_USED} Google allows 500 a day."


async def test_a_day_s_limit_without_a_number_says_none(client: FakeClient) -> None:
    failure = refusal(429, "quota", "RESOURCE_EXHAUSTED", PER_DAY, limit="")

    assert await refused_with(client, failure) == QUOTA_USED


async def test_a_retired_model_names_the_documents_line(client: FakeClient) -> None:
    failure = refusal(404, f"models/{DOCUMENT_MODEL} is not found", "NOT_FOUND")

    sentence = await refused_with(client, failure)

    assert sentence == MODEL_GONE.format(model=DOCUMENT_MODEL)
    assert "[documents]" in sentence


async def test_a_request_google_refuses_is_said_as_refused(client: FakeClient) -> None:
    failure = refusal(400, "The document has no pages.", "INVALID_ARGUMENT")

    assert await refused_with(client, failure) == REFUSED.format(reason="400 INVALID_ARGUMENT")


async def test_a_server_error_is_unreachable(client: FakeClient) -> None:
    failure = refusal(500, "internal", "INTERNAL")

    assert await refused_with(client, failure) == UNREACHABLE.format(reason="500 INTERNAL")


async def test_no_network_is_unreachable(client: FakeClient) -> None:
    failure = httpx.ConnectError("no route")

    assert await refused_with(client, failure) == UNREACHABLE.format(reason="ConnectError")


async def test_a_model_that_does_not_answer_in_time_is_said(client: FakeClient) -> None:
    client.models.delay = 1.0

    with pytest.raises(DocumentError) as caught:
        await asker(client, seconds=0.05).answer("x", FILES)

    assert str(caught.value) == NO_ANSWER.format(seconds=0.05)


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


async def test_the_log_counts_and_never_quotes(client: FakeClient) -> None:
    lines, sink = logged("DEBUG")
    try:
        await asker(client).answer("Ali Yılmaz'ın fatura tutarı ne?", FILES)
    finally:
        logger.remove(sink)

    joined = "\n".join(lines)
    assert "2 files" in joined
    # Not the question, not the answer, not a file's name or its text.
    for secret in ("Yılmaz", "12.500", "fatura", "mizan", "Hesap"):
        assert secret not in joined


async def test_the_key_never_reaches_the_log(client: FakeClient) -> None:
    lines, sink = logged("DEBUG")
    try:
        await refused_with(client, refusal(429, "quota", "RESOURCE_EXHAUSTED"))
    finally:
        logger.remove(sink)

    assert lines
    assert all(KEY not in line for line in lines)


# --------------------------------------------------------------------------
# The heavy libraries
# --------------------------------------------------------------------------


def test_the_libraries_are_imported_only_where_a_file_is_read() -> None:
    """`allie.config` imports this package for its defaults, and every CLI
    command reads the configuration: no module here may load openpyxl,
    python-docx or Google's SDK at import time."""
    for module in (convert, folder_module, model_module):
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"openpyxl", "docx", "google"}, module.__name__
