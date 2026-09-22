"""Finding out what the user meant and handing it to something that plays it.

The engine is here; the tools the model calls are in `tools/media.py`. The
split is the one the rest of the program uses: a tool is a docstring, a
signature and a sentence back to the model, and everything it has to know how
to do lives behind it.

Read `track.py` first - it is the shape everything here produces - then
`youtube.py`, which is where the identifier the model must never invent is
actually looked up.
"""
