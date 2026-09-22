"""`tools/local.py`: the owner's own tools, read from a folder outside the repository.

The folder is `%APPDATA%\\allie\\tools`; a test never touches it and writes
its files into `tmp_path` instead. What is asserted is the contract the
composition root relies on: every `Tool` in every file, in a fixed order; a
file that will not import skipped with a warning rather than a crash; nothing
that is not a `Tool` mistaken for one; and a name clash with a public tool
refused rather than resolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from allie.tools.local import load_local_tools, local_tools_dir
from allie.tools.registry import Tool, ToolRegistry
from allie.tools.system import get_current_time

TWO_TOOLS = '''
from allie.tools.registry import tool


@tool(risk="safe")
async def first_thing() -> str:
    """Does the first thing."""
    return "first done"


@tool(risk="confirm", confirm_prompt="Do the second thing?")
async def second_thing() -> str:
    """Does the second thing."""
    return "second done"
'''

ONE_TOOL = '''
from allie.tools.registry import tool


@tool(risk="safe")
async def {name}() -> str:
    """Does {name}."""
    return "{name} done"
'''

NOT_ONLY_TOOLS = '''
from allie.tools.registry import tool

LIMIT = 3


def helper() -> str:
    return "not a tool"


class Thing:
    pass


@tool(risk="safe")
async def the_tool() -> str:
    """The one tool in here."""
    return "done"


alias = the_tool
'''


def write(folder: Path, name: str, source: str) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(source, encoding="utf-8")
    return path


def names(tools: list[Tool]) -> list[str]:
    return [entry.spec.name for entry in tools]


def test_no_folder_means_no_tools(tmp_path: Path) -> None:
    assert load_local_tools(tmp_path / "tools") == []


def test_every_tool_in_a_file_in_definition_order(tmp_path: Path) -> None:
    folder = tmp_path / "tools"
    write(folder, "mine.py", TWO_TOOLS)

    tools = load_local_tools(folder)

    assert names(tools) == ["first_thing", "second_thing"]
    assert [entry.risk for entry in tools] == ["safe", "confirm"]
    assert tools[1].confirm_prompt == "Do the second thing?"


def test_files_are_read_in_name_order(tmp_path: Path) -> None:
    folder = tmp_path / "tools"
    write(folder, "b.py", ONE_TOOL.format(name="beta"))
    write(folder, "a.py", ONE_TOOL.format(name="alpha"))

    assert names(load_local_tools(folder)) == ["alpha", "beta"]


def test_a_file_named_with_an_underscore_is_left_alone(tmp_path: Path) -> None:
    """`_helpers.py` is for code the other files share; it is not run on its own."""
    folder = tmp_path / "tools"
    marker = tmp_path / "touched"
    write(folder, "_helpers.py", f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    write(folder, "real.py", ONE_TOOL.format(name="real"))

    assert names(load_local_tools(folder)) == ["real"]
    assert not marker.exists()


def test_a_file_that_will_not_import_is_skipped_with_a_warning(tmp_path: Path) -> None:
    folder = tmp_path / "tools"
    write(folder, "bad.py", 'raise RuntimeError("the secret is in the message")\n')
    write(folder, "good.py", ONE_TOOL.format(name="good"))
    lines: list[str] = []
    handle = logger.add(lines.append, format="{message}")
    try:
        tools = load_local_tools(folder)
    finally:
        logger.remove(handle)

    assert names(tools) == ["good"]
    assert [line for line in lines if "bad.py" in line and "RuntimeError" in line]
    assert all("secret" not in line for line in lines)


def test_only_tools_are_taken_and_each_once(tmp_path: Path) -> None:
    folder = tmp_path / "tools"
    write(folder, "mixed.py", NOT_ONLY_TOOLS)

    assert names(load_local_tools(folder)) == ["the_tool"]


async def test_a_loaded_tool_runs(tmp_path: Path) -> None:
    folder = tmp_path / "tools"
    write(folder, "mine.py", TWO_TOOLS)

    tools = load_local_tools(folder)

    assert await tools[0].run() == "first done"


def test_a_name_clash_with_a_public_tool_is_refused(tmp_path: Path) -> None:
    """A private file must not silently replace a public tool: the registry
    refuses two tools of one name, and startup stops with that sentence."""
    folder = tmp_path / "tools"
    write(folder, "shadow.py", ONE_TOOL.format(name="get_current_time"))

    with pytest.raises(ValueError, match="two tools named 'get_current_time'"):
        ToolRegistry([get_current_time, *load_local_tools(folder)])


def test_the_folder_sits_beside_the_settings(config_home: Path) -> None:
    assert local_tools_dir() == config_home / "tools"
