"""Tools that live on this computer only (design.md section 3.9, 13 Sep 2026).

Two kinds of tool exist from today on. The ones in this package are for
everyone: written here, registered in `__main__.py`, tested in `tests/`,
published. The other kind is the owner's own - "open my project and start
its server" - and is nobody else's business, so it never enters the
repository at all: it is a `.py` file in `%APPDATA%\\live-assistant\\tools\\`,
next to `config.toml`, and this module reads that folder at startup.

A local file is ordinary Python written against the same interpreter and
virtual environment: it imports `tool` from `assistant.tools.registry`,
declares its risk the way every tool does, and what it defines is put into
the same `ToolRegistry` as everything else. There is no second gate and no
second kind of tool - only a second place a file can be.

A file that will not import is skipped with one warning naming the file, so
that a typo in a private tool does not keep the assistant from starting. A
name clash with a public tool is not forgiven: `ToolRegistry` refuses two
tools of one name, and a private file must not silently replace a public one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from loguru import logger

from assistant.config import config_dir
from assistant.tools.registry import Tool

__all__ = ["LOCAL_TOOLS_DIR_NAME", "MODULE_PREFIX", "load_local_tools", "local_tools_dir"]

LOCAL_TOOLS_DIR_NAME = "tools"
# The package the files are imported under. `sys.modules` needs a name, and
# this one cannot collide with anything installed.
MODULE_PREFIX = "assistant_local"


def local_tools_dir() -> Path:
    """`%APPDATA%\\live-assistant\\tools` - beside `config.toml`."""
    return config_dir() / LOCAL_TOOLS_DIR_NAME


def load_local_tools(directory: Path | None = None) -> list[Tool]:
    """Every `Tool` the `*.py` files of `directory` define, in file order.

    Files are read in name order; one whose name starts with `_` is left
    alone, which makes `_helpers.py` a place for code the others share. A
    file that raises while importing is skipped with a warning naming it,
    and the rest still load.
    """
    folder = local_tools_dir() if directory is None else directory
    if not folder.is_dir():
        return []

    tools: list[Tool] = []
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tools.extend(_tools_in(path))
    return tools


def _tools_in(path: Path) -> list[Tool]:
    """Imports one file from its path and picks out the tools it defines."""
    name = f"{MODULE_PREFIX}.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        logger.warning("local tool file {} did not load: not importable", path.name)
        return []
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as `import` would: a dataclass or an
    # annotation in the file may look its own module up by name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        # Whatever the file did wrong, the assistant still starts. The class
        # name is enough; the message is the file's own business.
        del sys.modules[name]
        logger.warning("local tool file {} did not load: {}", path.name, type(error).__name__)
        return []

    found: list[Tool] = []
    for value in vars(module).values():
        if isinstance(value, Tool) and not any(value is seen for seen in found):
            found.append(value)
    return found
