"""Stable, offline Vectornaut test suite.

Run from the repository root with:

    python -m unittest discover -s tests -t .

``tests/live/`` and ``tests/manual/`` intentionally have no ``__init__.py``,
so unittest discovery does not descend into them. See docs/TESTING.md.
"""
