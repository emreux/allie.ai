"""The machine itself (plan.md D40-D42, 27 Sep 2026): power, switches,
windows and files. Each piece is a `Protocol` with one Windows
implementation, so that the tools are tested over fakes and nothing in the
suite locks, sleeps, switches a radio or opens a file."""
