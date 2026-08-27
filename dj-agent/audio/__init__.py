"""The DSP: the realtime mixer and the offline track analysis.

Knows nothing about agents, sources, or tracing. `mixer.py` in particular is the
one module in the project that must not import `instrumentation` — see the
callback rule at the top of that file.
"""
