"""The mark: Allie's own icon, drawn rather than shipped (task U5).

The window wears it in its top-left corner and Windows shows it on the
taskbar button. It is the orb seen from far away and standing still - the
idle blue, a white-hot core, one heavy broken ring around it and, where
there are pixels for it, a hairline arc further out. Nobody needs to
recognise a mechanism at sixteen pixels; they need to recognise the blue
eye, which is what is left when the fine work is dropped.

**Drawn, not shipped**, like the tray's disc (`ui/tray.py`): no binary in
the repository, no file to lose, and the blue is the orb's own
`COLOURS[State.IDLE]` rather than a second copy of it that could drift.

**Supersampled.** Pillow's ellipse and arc have stepped edges; every piece
is drawn several times larger and the picture shrunk back with Lanczos, so
that the curves come out smooth at the size the window actually asks for.

No Tk here, on the orb's rule: the mark is a picture and can be tested as
one, without a screen.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from assistant.app import State
from assistant.ui.orb import COLOURS, RGB

__all__ = ["ICON_SIZES", "MARK", "SMALLEST_WITH_HAIRLINE", "draw_logo"]

# What `iconphoto` is handed: Windows picks the title bar's from the small
# end and the taskbar's and Alt-Tab's from the large.
ICON_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 256)

# The orb's ready blue, and the white its core burns towards.
MARK: RGB = COLOURS[State.IDLE]
HOT: RGB = (238, 248, 255)

# Under this, the outer hairline is dropped: at twenty-four pixels it is a
# third of a pixel wide and comes out as grey dust round the mark.
SMALLEST_WITH_HAIRLINE = 32
# With the hairline dropped there is empty room where it would have been,
# so the rest of the mark grows into it - a small icon that fills its box.
COMPACT = 1.27

# Everything below is a fraction of the mark's half-size, the orb's units.
CORE = 0.36
# The core is a glow, not a target: this many discs stepping inwards, each
# nearer the white, so that no ring of its own is left in the gradient.
CORE_STEPS = 28
CORE_HOT = 0.92
CORE_FALLOFF = 1.8
RING = 0.64
RING_WIDTH = 0.17
# The heavy ring is broken twice, opposite each other - the orb's arcs seen
# from a distance, not a closed circle.
RING_ARCS: tuple[tuple[float, float], ...] = ((-58, 118), (142, 302))
HAIRLINE = 0.90
HAIRLINE_WIDTH = 0.05
HAIRLINE_ARCS: tuple[tuple[float, float], ...] = ((168, 258), (-12, 44))
# The bright dot riding the heavy ring, as the orb's caps do.
CAP = 0.075
CAP_AT = 118.0


def _scale(size: int) -> int:
    """How many times larger the mark is drawn before it is shrunk back.
    Small marks get the most, because that is where a stepped edge shows;
    the largest would cost four million pixels at the same factor."""
    return 8 if size <= 64 else 4


def _box(centre: float, radius: float) -> tuple[float, float, float, float]:
    return (centre - radius, centre - radius, centre + radius, centre + radius)


def _blend(colour: RGB, towards: RGB, amount: float) -> RGB:
    red, green, blue = (round(c + (t - c) * amount) for c, t in zip(colour, towards, strict=True))
    return (red, green, blue)


def draw_logo(size: int) -> Image.Image:
    """The mark, `size` by `size`, on nothing (RGBA). The same size always
    gives the same pixels: it is a drawing, not a render of a live orb."""
    if size < 1:
        raise ValueError(f"a mark of {size} pixels cannot be drawn")
    scale = _scale(size)
    side = size * scale
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    centre = side / 2
    # A hair of room, so that the outermost stroke is not clipped.
    fine = size >= SMALLEST_WITH_HAIRLINE
    unit = (side / 2) * 0.94 * (1.0 if fine else COMPACT)

    if fine:
        width = max(1, round(HAIRLINE_WIDTH * unit))
        thin = (*_blend(MARK, (11, 15, 20), 0.35), 255)
        for start, end in HAIRLINE_ARCS:
            draw.arc(_box(centre, HAIRLINE * unit), start, end, fill=thin, width=width)

    heavy = max(1, round(RING_WIDTH * unit))
    for start, end in RING_ARCS:
        draw.arc(_box(centre, RING * unit), start, end, fill=(*MARK, 255), width=heavy)

    for step in range(CORE_STEPS):
        reach = 1 - step / CORE_STEPS
        light = CORE_HOT * (1 - reach) ** CORE_FALLOFF
        draw.ellipse(_box(centre, CORE * unit * reach), fill=(*_blend(MARK, HOT, light), 255))

    if fine:
        cap = CAP * unit
        radians = math.radians(CAP_AT)
        x = centre + RING * unit * math.cos(radians)
        y = centre + RING * unit * math.sin(radians)
        draw.ellipse((x - cap, y - cap, x + cap, y + cap), fill=(*_blend(MARK, HOT, 0.55), 255))

    return image.resize((size, size), Image.Resampling.LANCZOS)
