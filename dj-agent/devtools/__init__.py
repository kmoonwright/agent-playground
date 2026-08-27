"""Verification and fixtures, reachable as `dj.py` flags rather than as scripts.

Nothing in a running set imports these. `dj.py` loads them for
`--selftest-mixer`, `--trace-selftest`, and `--make-test-crate`; `tests/` reuses
`click_track` and `write_track` so there is one implementation of each fixture.
"""
