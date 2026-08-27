"""The DSP and tracing selftests, run as tests so CI catches regressions.

Each returns 0 only when all of its own assertions passed, so the exit code is
the whole contract -- there is nothing to add by also matching its printed text.
"""

from __future__ import annotations

import devtools.selftest as selftest


def test_mixer_selftest_passes_all_four_assertions():
    # Short crossfade keeps the test quick; the assertions are the same.
    assert selftest.run(seconds=20.0, xfade_bars=4, write_wav=False) == 0


def test_tracing_selftest_proves_cross_thread_parenting():
    assert selftest.run_tracing() == 0
