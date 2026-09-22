"""The mark (`ui/logo.py`, task U5): Allie's icon, drawn rather than shipped.

What is claimed: every size Windows is offered comes out at that size, on
nothing, with the orb's own idle blue burning white at the centre; the
corners stay empty so the mark is a disc and not a tile; the small sizes
drop the hairline and grow into the room it leaves; the same size always
gives the same pixels; a size of nothing is a bug and not an empty picture.
Nothing here opens a screen.
"""

from __future__ import annotations

import inspect

import pytest
from PIL import Image

from allie.app import State
from allie.ui import logo
from allie.ui.logo import ICON_SIZES, MARK, SMALLEST_WITH_HAIRLINE, draw_logo
from allie.ui.orb import COLOURS


def _pixel(image: Image.Image, at: tuple[int, int]) -> tuple[int, int, int, int]:
    """One pixel as four numbers, which is what an RGBA picture holds."""
    value = image.getpixel(at)
    assert isinstance(value, tuple) and len(value) == 4
    return value


def _opaque(image: Image.Image) -> float:
    """How much of the picture is not transparent."""
    counts = image.getchannel("A").histogram()
    return sum(counts[9:]) / (image.width * image.height)


@pytest.mark.parametrize("size", ICON_SIZES)
def test_every_offered_size_is_drawn_at_that_size_on_nothing(size: int) -> None:
    image = draw_logo(size)

    assert image.size == (size, size)
    assert image.mode == "RGBA"
    for corner in ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)):
        assert _pixel(image, corner)[3] == 0


def test_the_mark_is_the_orbs_ready_blue_burning_white_at_the_centre() -> None:
    """One blue, not two: the icon reads `COLOURS[State.IDLE]` rather than
    keeping a copy that could drift from the orb's."""
    assert COLOURS[State.IDLE] == MARK

    image = draw_logo(64)
    red, green, blue, alpha = _pixel(image, (32, 32))

    assert alpha == 255
    assert blue > green > red
    assert red > MARK[0]


def test_the_small_sizes_drop_the_hairline_and_fill_the_room_it_leaves() -> None:
    """At sixteen pixels a twentieth-of-a-radius stroke is dust; what is
    left grows instead, so the mark still fills its box."""
    small = _opaque(draw_logo(SMALLEST_WITH_HAIRLINE - 8))
    large = _opaque(draw_logo(SMALLEST_WITH_HAIRLINE))

    assert 0.2 < small < 0.7
    assert 0.2 < large < 0.7


@pytest.mark.parametrize("size", ICON_SIZES)
def test_the_mark_has_ink_in_it_at_every_size(size: int) -> None:
    assert _opaque(draw_logo(size)) > 0.15


def test_the_same_size_gives_the_same_pixels() -> None:
    assert draw_logo(32).tobytes() == draw_logo(32).tobytes()


def test_a_mark_of_no_pixels_is_a_bug() -> None:
    with pytest.raises(ValueError, match="cannot be drawn"):
        draw_logo(0)


def test_the_title_bars_size_is_offered_and_so_is_the_taskbars() -> None:
    assert 16 in ICON_SIZES
    assert max(ICON_SIZES) >= 256


def test_the_module_opens_no_screen() -> None:
    """A drawing, testable because nothing here touches Tk."""
    assert "tkinter" not in inspect.getsource(logo)
