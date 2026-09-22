"""The database (design.md section 3.7): plain `sqlite3`, one file, no ORM.

`db.py` opens it, `migrations.py` keeps its schema up to date, `repos.py`
writes to it - and nothing else in the project knows a table's columns.

Phase 2.1d is `tool_audit` and the audit repository; 2.4 added `usage_log`,
the `args_hash` column and the second migration. `settings` comes with the
probe of 2.6, notes and reminders with phase 4; do not assume a table exists
because section 3.7 lists it.
"""

__all__: list[str] = []
