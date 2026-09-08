# Measuring and improving agent reliability in production

A 200 response, clean-looking spans, and a fluent confirmation message can all coexist with a task that actually failed. Reliability work starts from that assumption.

## Five silent failure modes
1. **False completion** — the agent reports success without evidence the outcome actually happened.
2. **No-progress behavior** — the agent repeats actions without advancing the task.
3. **Partial completion** — only part of the requested change happens, leaving an inconsistent state.
4. **Constraint loss** — the agent drops a requirement, permission, or delivery instruction partway through a workflow.
5. **Harmful success** — the task technically completes, but through an unauthorized path or action.

Long trajectories compound the risk: at a simplified 95% success rate per step, an eight-step task only succeeds end-to-end about 66% of the time.

## What a reliability scorecard tracks
Task success rate with real outcome verification (not just "did it return"), consistency across repeated runs, recovery rate after an error, robustness to input/environment changes, how often false-completion and no-progress happen, failure severity and blast radius, cost and latency per completed task, and how often a human has to escalate, abandon, or repair the result. These should be segmented by workflow, customer, tool path, and consequence — a single global average hides most of what matters.

## An eight-step improvement loop
1. Define what "done" means for the task and how you'd prove it.
2. Capture full execution traces, not just final responses.
3. Measure known risks with offline evaluation.
4. Look at production traffic to find failure modes you didn't already know about.
5. Turn confirmed failures into regression-test dataset rows.
6. Investigate root cause, propose a narrowly-scoped fix.
7. Verify the fix against full agent behavior in an experiment — not just the final answer.
8. Release gradually, with monitoring and a rollback path.

The underlying idea: reliability improves when every real production failure becomes durable evidence that shapes the next change, instead of being forgotten once it's fixed.
