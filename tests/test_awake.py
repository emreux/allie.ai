"""`StayAwake` (plan.md D61): a power request held while the phone remote
runs, and given back when it stops - also when it stops with an error.

Over a fake kernel, except for one test that asks Windows itself for a
request and gives it straight back: the structure the four calls take is
only proved by Windows accepting it."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from loguru import logger

from allie.computer.awake import REASON, StayAwake, WindowsKernel

HANDLE = 0x5A


class FakeKernel:
    """Records the four calls of a power request, in order."""

    def __init__(self, *, handle: int | None = HANDLE, sets: bool = True) -> None:
        self.handle = handle
        self.sets = sets
        self.calls: list[tuple[str, object]] = []

    def create(self, reason: str) -> int | None:
        self.calls.append(("create", reason))
        return self.handle

    def set(self, handle: int) -> bool:
        self.calls.append(("set", handle))
        return self.sets

    def clear(self, handle: int) -> bool:
        self.calls.append(("clear", handle))
        return True

    def close(self, handle: int) -> None:
        self.calls.append(("close", handle))


@pytest.fixture
def warned() -> Iterator[list[str]]:
    caught: list[str] = []
    sink = logger.add(caught.append, level="WARNING", format="{message}")
    yield caught
    logger.remove(sink)


def test_the_request_is_held_inside_and_given_back_after() -> None:
    kernel = FakeKernel()

    with StayAwake(kernel) as awake:
        assert kernel.calls == [("create", REASON), ("set", HANDLE)]
        assert awake.held

    assert kernel.calls[2:] == [("clear", HANDLE), ("close", HANDLE)]
    assert not awake.held


def test_the_request_is_given_back_when_the_body_fails() -> None:
    kernel = FakeKernel()

    with pytest.raises(RuntimeError), StayAwake(kernel):
        raise RuntimeError("the channel stopped")

    assert kernel.calls[2:] == [("clear", HANDLE), ("close", HANDLE)]


def test_the_reason_names_the_assistant_and_the_phone() -> None:
    """What `powercfg /requests` shows under SYSTEM."""
    assert REASON == "Allie: Telegram remote control"


def test_a_request_windows_will_not_make_is_a_warning_and_the_body_runs(
    warned: list[str],
) -> None:
    kernel = FakeKernel(handle=None)
    ran = False

    with StayAwake(kernel) as awake:
        ran = True
        assert not awake.held

    assert ran
    assert kernel.calls == [("create", REASON)]
    assert len(warned) == 1


def test_a_request_windows_will_not_set_is_closed_and_the_body_runs(
    warned: list[str],
) -> None:
    kernel = FakeKernel(sets=False)

    with StayAwake(kernel) as awake:
        assert not awake.held

    assert kernel.calls == [("create", REASON), ("set", HANDLE), ("close", HANDLE)]
    assert len(warned) == 1


@pytest.mark.skipif(os.name != "nt", reason="the product ships on Windows")
def test_windows_takes_the_request_and_gives_it_back(warned: list[str]) -> None:
    """The real `kernel32`: created, set, cleared and closed in a moment."""
    with StayAwake(WindowsKernel()) as awake:
        assert awake.held

    assert not awake.held
    assert warned == []
