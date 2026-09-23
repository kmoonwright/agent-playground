#!/usr/bin/env python3
"""
Security Agent demo traces for Arize AX
========================================

Generates realistic, multi-step agent traces (OpenInference spans over
OpenTelemetry) for five agentic use cases:

  1. Agentic Vendor Risk Assessment   -> vendor-risk-assessment-agent
  2. Safe AI Adoption (shadow AI)     -> ai-discovery-agent
  3. Blast Radius Impact Analysis     -> blast-radius-agent
  4. Attack Surface Reduction         -> attack-surface-agent
  5. Agentic Access Policy Enforcement -> policy-enforcement-agent

Each scenario is generated as a multi-turn **session**: 2-4 AGENT-rooted
traces that share one `session.id`, spaced out in time like a real
back-and-forth between a security analyst and the agent (kickoff ask,
then follow-up asks that reference the prior turn's finding). Each
trace has LLM / TOOL / RETRIEVER child spans, sent to Arize with
synthetic (backdated) timestamps so the Tracing tab looks like a few
days of real production activity instead of one burst of
identical-looking, single-turn traces.

Setup
-----
    pip install -r requirements.txt

Configure via .env in this directory (ARIZE_SPACE_ID, ARIZE_API_KEY,
ARIZE_PROJECT_NAME). Optional: SESSIONS_PER_SCENARIO, TURNS_PER_SESSION_MIN,
TURNS_PER_SESSION_MAX, HOURS_OF_HISTORY.

Run:
    python generate_security_demo_traces.py

Then open Arize AX -> your Space -> the configured project -> Tracing.
Use the `use_case` and `vendor` tags/metadata to filter by scenario in the UI.
"""

import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes
from openinference.instrumentation import using_session, using_metadata, using_tags
from arize.otel import register

# ---------------------------------------------------------------------------
# 0. Config + tracer setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

SPACE_ID = os.environ.get("ARIZE_SPACE_ID", "").strip()
API_KEY = os.environ.get("ARIZE_API_KEY", "").strip()
PROJECT_NAME = os.environ.get("ARIZE_PROJECT_NAME", "security-agent").strip()
SESSIONS_PER_SCENARIO = int(os.environ.get("SESSIONS_PER_SCENARIO", 5))
TURNS_PER_SESSION_MIN = int(os.environ.get("TURNS_PER_SESSION_MIN", 2))
TURNS_PER_SESSION_MAX = int(os.environ.get("TURNS_PER_SESSION_MAX", 4))
HOURS_OF_HISTORY = int(os.environ.get("HOURS_OF_HISTORY", 72))

if not SPACE_ID or not API_KEY:
    raise SystemExit(
        f"ARIZE_SPACE_ID and ARIZE_API_KEY are required. "
        f"Put them in {ROOT / '.env'} and re-run."
    )

tracer_provider = register(
    space_id=SPACE_ID,
    api_key=API_KEY,
    project_name=PROJECT_NAME,
)
tracer = tracer_provider.get_tracer(__name__)

random.seed(7)

# ---------------------------------------------------------------------------
# 1. Fake org / vendor data — gives the demo a believable "customer"
# ---------------------------------------------------------------------------
VENDORS = [
    {"name": "Harborline Analytics", "slug": "harborline", "category": "Marketing SaaS",
     "domain": "harborline.io"},
    {"name": "Cascade DataOps", "slug": "cascade-dataops", "category": "Data Pipeline / ETL",
     "domain": "cascadedata.com"},
    {"name": "Verdant CX", "slug": "verdant-cx", "category": "AI Customer Support",
     "domain": "verdantcx.ai"},
    {"name": "Quillbind Docs", "slug": "quillbind-docs", "category": "Contract Management",
     "domain": "quillbind-docs.com"},
    {"name": "Fenwick Payroll", "slug": "fenwick-payroll", "category": "HR / Payroll",
     "domain": "fenwickpay.com"},
]

ORG_NAME = "Acme Corp"

# ---------------------------------------------------------------------------
# 2. Low-level span helpers, with fully synthetic (backdated) timestamps
# ---------------------------------------------------------------------------

def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def open_span(name, kind, start_dt, attributes=None):
    """Start a span with an explicit historical start time; caller must
    call close_span() — we don't let OTel auto-timestamp end_time."""
    cm = tracer.start_as_current_span(
        name,
        openinference_span_kind=kind,
        start_time=_ns(start_dt),
        end_on_exit=False,
    )
    span = cm.__enter__()
    if attributes:
        span.set_attributes(attributes)
    return cm, span


def close_span(cm, span, end_dt, error=None):
    if error:
        span.set_status(Status(StatusCode.ERROR, description=error))
        span.record_exception(Exception(error))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end(end_time=_ns(end_dt))
    cm.__exit__(None, None, None)


def leaf(name, kind, start_dt, duration_s, input_text=None, output_text=None,
         extra_attrs=None, error=None):
    attrs = {}
    if input_text is not None:
        attrs[SpanAttributes.INPUT_VALUE] = input_text
    if output_text is not None:
        attrs[SpanAttributes.OUTPUT_VALUE] = output_text
    if extra_attrs:
        attrs.update(extra_attrs)
    cm, span = open_span(name, kind, start_dt, attrs)
    end_dt = start_dt + timedelta(seconds=duration_s)
    close_span(cm, span, end_dt, error=error)
    return end_dt


def llm_leaf(name, start_dt, duration_s, model, prompt, completion,
             prompt_tokens, completion_tokens, error=None):
    provider = "anthropic" if "claude" in model else "openai"
    attrs = {
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.LLM_PROVIDER: provider,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: prompt_tokens + completion_tokens,
    }
    return leaf(name, "llm", start_dt, duration_s, input_text=prompt,
                output_text=completion, extra_attrs=attrs, error=error)


def tool_leaf(name, start_dt, duration_s, tool_name, input_text, output_text, error=None):
    attrs = {SpanAttributes.TOOL_NAME: tool_name}
    return leaf(name, "tool", start_dt, duration_s, input_text=input_text,
                output_text=output_text, extra_attrs=attrs, error=error)


def retriever_leaf(name, start_dt, duration_s, query, documents):
    # documents: list[(content, score)]
    output_text = "; ".join(f"[{score:.2f}] {content}" for content, score in documents)
    return leaf(name, "retriever", start_dt, duration_s, input_text=query, output_text=output_text)


# ---------------------------------------------------------------------------
# 3. Scenario builders — one per Security use case
#    Each returns (end_cursor, final_output_text, had_error)
# ---------------------------------------------------------------------------

def scenario_vendor_risk_assessment(vendor, cursor, variant):
    """Use case: Agentic Vendor Risk Assessment"""
    risk_score = random.randint(35, 92)
    cursor = llm_leaf(
        "plan_assessment", cursor, random.uniform(0.6, 1.4), model="claude-sonnet-4-5",
        prompt=f"Plan a risk assessment refresh for vendor '{vendor['name']}' "
               f"({vendor['category']}). Decide which data sources to pull.",
        completion="Plan: pull security rating, scan for recent breach disclosures, "
                   "retrieve prior assessment history, then score and summarize.",
        prompt_tokens=random.randint(180, 260), completion_tokens=random.randint(60, 110),
    )
    cursor = tool_leaf(
        "fetch_security_posture", cursor, random.uniform(0.4, 1.8),
        tool_name="security_ratings_api",
        input_text=f"GET /ratings/{vendor['domain']}",
        output_text=f"{{'domain': '{vendor['domain']}', "
                    f"'letter_grade': '{random.choice(['A','B','C','D'])}', "
                    f"'open_cves': {random.randint(0, 6)}}}",
    )
    breach_found = variant % 3 == 1
    cursor = tool_leaf(
        "scan_breach_disclosures", cursor, random.uniform(0.5, 1.6),
        tool_name="threat_intel_search",
        input_text=f"breach disclosure {vendor['name']}",
        output_text=(f"1 disclosure found: credential-stuffing attempt reported against "
                     f"{vendor['name']} 6 weeks ago." if breach_found
                     else "No new disclosures in the last 90 days."),
    )
    cursor = retriever_leaf(
        "retrieve_prior_assessments", cursor, random.uniform(0.3, 0.9),
        query=f"prior risk assessments for {vendor['name']}",
        documents=[
            (f"Q{random.randint(1, 4)} assessment: baseline access to CRM contacts, "
             f"no PII processing flagged.", 0.88),
            (f"Onboarding review: {vendor['name']} passed SOC2 Type II review.", 0.74),
        ],
    )
    had_error = variant == 3
    write_input = f"update risk_score={risk_score} for {vendor['name']}"
    if had_error:
        cursor = tool_leaf(
            "write_score_to_grc", cursor, random.uniform(0.3, 0.7),
            tool_name="grc_system_write", input_text=write_input, output_text="",
            error="GRC API returned 503 Service Unavailable",
        )
        cursor = tool_leaf(
            "write_score_to_grc_retry", cursor, random.uniform(0.3, 0.6),
            tool_name="grc_system_write", input_text=write_input + " (retry)",
            output_text="200 OK — risk score updated",
        )
    else:
        cursor = tool_leaf(
            "write_score_to_grc", cursor, random.uniform(0.3, 0.6),
            tool_name="grc_system_write", input_text=write_input,
            output_text="200 OK — risk score updated",
        )
    summary = (f"Risk score for {vendor['name']} set to {risk_score}/100. "
               f"{'Elevated due to a recent breach signal.' if breach_found else 'No material change since last cycle.'}")
    cursor = llm_leaf(
        "generate_summary", cursor, random.uniform(0.5, 1.1), model="gpt-4o",
        prompt=f"Summarize the assessment findings for {vendor['name']} for the security team.",
        completion=summary,
        prompt_tokens=random.randint(220, 320), completion_tokens=random.randint(40, 80),
    )
    return cursor, summary, had_error


def scenario_ai_discovery(vendor, cursor, variant):
    """Use case: Safe AI Adoption / shadow-AI discovery"""
    is_ai_vendor = vendor["slug"] in ("verdant-cx",) or variant % 2 == 0
    cursor = tool_leaf(
        "scan_vendor_api_traffic", cursor, random.uniform(0.5, 1.5),
        tool_name="network_traffic_scanner",
        input_text=f"scan outbound calls for integration with {vendor['name']}",
        output_text=(f"Detected outbound calls from {vendor['domain']} to api.openai.com "
                     f"and api.anthropic.com" if is_ai_vendor
                     else f"No LLM-provider traffic detected for {vendor['domain']}"),
    )
    cursor = llm_leaf(
        "classify_ai_usage", cursor, random.uniform(0.5, 1.2), model="claude-sonnet-4-5",
        prompt=f"Classify the AI usage pattern detected for {vendor['name']}.",
        completion=(f"{vendor['name']} embeds third-party LLM providers (shadow sub-processor) "
                    f"inside its product to power {vendor['category'].lower()} features."
                    if is_ai_vendor else
                    f"{vendor['name']} shows no evidence of undisclosed AI/LLM usage."),
        prompt_tokens=random.randint(150, 220), completion_tokens=random.randint(50, 90),
    )
    cursor = tool_leaf(
        "check_data_classification", cursor, random.uniform(0.4, 1.0),
        tool_name="data_classification_api",
        input_text=f"classify data fields shared with {vendor['name']}",
        output_text="Fields shared: customer_name, email, support_transcript (contains PII)"
                    if is_ai_vendor else "Fields shared: account_id, usage_metrics (no PII)",
    )
    risk_level = "High" if is_ai_vendor else "Low"
    cursor = llm_leaf(
        "assess_ai_risk", cursor, random.uniform(0.6, 1.3), model="gpt-4o",
        prompt=f"Assess AI/data risk for {vendor['name']} given detected usage and data fields.",
        completion=f"Risk level: {risk_level}. "
                   f"{'PII is being processed by an undisclosed sub-processor LLM.' if is_ai_vendor else 'No undisclosed AI processing of sensitive data.'}",
        prompt_tokens=random.randint(200, 280), completion_tokens=random.randint(40, 70),
    )
    had_error = is_ai_vendor and variant == 1
    if is_ai_vendor:
        cursor = tool_leaf(
            "flag_shadow_ai_asset", cursor, random.uniform(0.3, 0.7),
            tool_name="asset_inventory_write",
            input_text=f"register shadow-AI sub-processor for {vendor['name']}",
            output_text="" if had_error else "Asset flagged in inventory; compliance team notified",
            error="Inventory service timeout" if had_error else None,
        )
    summary = f"{vendor['name']}: AI risk = {risk_level}."
    return cursor, summary, had_error


def scenario_blast_radius(vendor, cursor, variant):
    """Use case: Blast Radius Impact Analysis"""
    downstream_count = random.randint(2, 9)
    cursor = tool_leaf(
        "query_vendor_graph", cursor, random.uniform(0.4, 1.1),
        tool_name="dependency_graph_query",
        input_text=f"MATCH (v:Vendor {{name:'{vendor['name']}'}})-[:CONNECTS_TO*1..3]->(x) RETURN x",
        output_text=f"{downstream_count} connected systems found (2nd/3rd-party hops)",
    )
    cursor = llm_leaf(
        "simulate_breach_scenario", cursor, random.uniform(0.8, 1.9), model="claude-sonnet-4-5",
        prompt=f"Simulate a credential-compromise breach at {vendor['name']} and trace impact.",
        completion=f"If {vendor['name']} is compromised, attacker could pivot into "
                   f"{downstream_count} connected systems via shared API tokens and SSO scopes.",
        prompt_tokens=random.randint(260, 380), completion_tokens=random.randint(90, 160),
    )
    cursor = tool_leaf(
        "query_downstream_integrations", cursor, random.uniform(0.4, 1.0),
        tool_name="integration_inventory_api",
        input_text=f"list integrations reachable from {vendor['name']}",
        output_text=f"Includes: internal CRM, data warehouse, {random.choice(['Slack workspace', 'SSO/Okta tenant', 'billing system'])}",
    )
    impact_score = random.randint(40, 96)
    severity = "Critical" if impact_score > 80 else "Moderate"
    cursor = llm_leaf(
        "compute_impact_score", cursor, random.uniform(0.6, 1.2), model="gpt-4o",
        prompt=f"Score the blast-radius impact for {vendor['name']} on a 0-100 scale.",
        completion=f"Impact score: {impact_score}/100 ({severity}).",
        prompt_tokens=random.randint(220, 300), completion_tokens=random.randint(30, 60),
    )
    had_error = variant == 4
    cursor = tool_leaf(
        "generate_impact_report", cursor, random.uniform(0.3, 0.8),
        tool_name="report_generator",
        input_text=f"generate blast-radius report for {vendor['name']}",
        output_text="" if had_error else f"Report published to #vendor-risk (impact={severity})",
        error="Report generator returned malformed template" if had_error else None,
    )
    summary = f"{vendor['name']} blast radius: {downstream_count} systems, impact {impact_score}/100 ({severity})."
    return cursor, summary, had_error


def scenario_attack_surface(vendor, cursor, variant):
    """Use case: Attack Surface Reduction"""
    permission_count = random.randint(3, 14)
    cursor = tool_leaf(
        "list_vendor_permissions", cursor, random.uniform(0.5, 1.3),
        tool_name="idp_permissions_query",
        input_text=f"list OAuth scopes and roles granted to {vendor['name']}",
        output_text=f"{permission_count} scopes granted, including {random.choice(['admin:write', 'contacts:read', 'files:read-write'])}",
    )
    last_used_days = random.choice([2, 14, 45, 120, 210])
    cursor = tool_leaf(
        "cross_check_usage_logs", cursor, random.uniform(0.5, 1.4),
        tool_name="access_log_query",
        input_text=f"last API activity for {vendor['name']}",
        output_text=f"Last active {last_used_days} days ago",
    )
    is_stale = last_used_days > 90
    is_over_permissioned = permission_count > 8
    flag_text = ("No issues found." if not (is_stale or is_over_permissioned)
                 else f"{vendor['name']} holds more access than its usage justifies.")
    cursor = llm_leaf(
        "identify_stale_or_over_permissioned", cursor, random.uniform(0.7, 1.5), model="claude-sonnet-4-5",
        prompt=f"Given {permission_count} scopes and last activity {last_used_days} days ago for "
               f"{vendor['name']}, flag stale or over-permissioned access.",
        completion=f"{'STALE: ' if is_stale else ''}{'OVER-PERMISSIONED: ' if is_over_permissioned else ''}{flag_text}",
        prompt_tokens=random.randint(190, 260), completion_tokens=random.randint(50, 100),
    )
    cursor = llm_leaf(
        "recommend_deprovisioning", cursor, random.uniform(0.5, 1.0), model="gpt-4o",
        prompt=f"Recommend a remediation action for {vendor['name']}.",
        completion=(f"Recommend revoking {random.randint(1, max(permission_count - 2, 1))} unused scopes"
                    if (is_stale or is_over_permissioned) else "No action needed this cycle."),
        prompt_tokens=random.randint(160, 220), completion_tokens=random.randint(30, 60),
    )
    had_error = (is_stale or is_over_permissioned) and variant == 2
    if is_stale or is_over_permissioned:
        cursor = tool_leaf(
            "create_remediation_ticket", cursor, random.uniform(0.3, 0.7),
            tool_name="ticketing_api",
            input_text=f"create remediation ticket for {vendor['name']} over-permissioning",
            output_text="" if had_error else "Ticket SEC-4821 created and assigned to vendor-mgmt queue",
            error="Ticketing API auth token expired" if had_error else None,
        )
        summary = f"{vendor['name']}: flagged for over-permissioning/staleness, remediation ticket filed."
    else:
        summary = f"{vendor['name']}: access footprint is right-sized, no action needed."
    return cursor, summary, had_error


def scenario_policy_enforcement(vendor, cursor, variant):
    """Use case: Agentic Access Policy Enforcement"""
    cursor = retriever_leaf(
        "retrieve_access_policy", cursor, random.uniform(0.3, 0.8),
        query=f"third-party access policy for {vendor['category']} vendors",
        documents=[
            (f"TPRM Policy 4.2: {vendor['category']} vendors may not hold standing write access "
             f"to production data stores.", 0.91),
            ("TPRM Policy 2.1: All vendor API tokens must rotate every 90 days.", 0.68),
        ],
    )
    cursor = tool_leaf(
        "get_current_vendor_access", cursor, random.uniform(0.4, 1.1),
        tool_name="idp_permissions_query",
        input_text=f"current access grants for {vendor['name']}",
        output_text=f"Grants: {random.choice(['prod-db:write', 'contacts:read-write', 'files:admin'])}, "
                    f"token age {random.randint(10, 260)} days",
    )
    violation = variant % 2 == 0
    cursor = llm_leaf(
        "evaluate_policy_compliance", cursor, random.uniform(0.7, 1.5), model="claude-sonnet-4-5",
        prompt=f"Evaluate {vendor['name']}'s current access grants against TPRM policy 4.2 and 2.1.",
        completion=(f"Violation: {vendor['name']} holds standing write access in breach of Policy 4.2."
                    if violation else f"{vendor['name']} is within policy on both checks."),
        prompt_tokens=random.randint(240, 320), completion_tokens=random.randint(50, 90),
    )
    had_error = violation and variant == 0
    if violation:
        action_input = f"revoke excess write access for {vendor['name']}"
        cursor = tool_leaf(
            "revoke_excess_access", cursor, random.uniform(0.4, 1.0),
            tool_name="idp_access_enforcement_api",
            input_text=action_input,
            output_text="" if had_error else "Write scope revoked; vendor downgraded to read-only",
            error="403 Forbidden — enforcement account lacks admin scope, escalating to on-call" if had_error else None,
        )
        if had_error:
            cursor = tool_leaf(
                "escalate_to_oncall", cursor, random.uniform(0.2, 0.5),
                tool_name="pagerduty_api",
                input_text=f"escalate blocked enforcement action for {vendor['name']}",
                output_text="Incident created, on-call security engineer paged",
            )
        summary = f"{vendor['name']}: policy violation found, enforcement {'escalated to on-call' if had_error else 'auto-remediated'}."
    else:
        summary = f"{vendor['name']}: compliant, no enforcement action taken."
    cursor = llm_leaf(
        "generate_enforcement_summary", cursor, random.uniform(0.4, 0.9), model="gpt-4o",
        prompt=f"Write a one-line enforcement summary for {vendor['name']}.",
        completion=summary,
        prompt_tokens=random.randint(150, 200), completion_tokens=random.randint(20, 45),
    )
    return cursor, summary, had_error


SCENARIOS = {
    "vendor_risk_assessment": {
        "agent_name": "vendor-risk-assessment-agent",
        "use_case": "agentic-vendor-risk-assessment",
        "fn": scenario_vendor_risk_assessment,
    },
    "ai_discovery": {
        "agent_name": "ai-discovery-agent",
        "use_case": "safe-ai-adoption",
        "fn": scenario_ai_discovery,
    },
    "blast_radius": {
        "agent_name": "blast-radius-agent",
        "use_case": "blast-radius-impact-analysis",
        "fn": scenario_blast_radius,
    },
    "attack_surface": {
        "agent_name": "attack-surface-agent",
        "use_case": "attack-surface-reduction",
        "fn": scenario_attack_surface,
    },
    "policy_enforcement": {
        "agent_name": "policy-enforcement-agent",
        "use_case": "agentic-access-policy-enforcement",
        "fn": scenario_policy_enforcement,
    },
}

# Opening ask for turn 1 of a session, per scenario.
KICKOFF_PROMPTS = {
    "vendor_risk_assessment": "Run a risk assessment refresh on",
    "ai_discovery": "Check whether there's any shadow AI usage from",
    "blast_radius": "Run a blast-radius impact analysis for",
    "attack_surface": "Review the attack surface / access footprint for",
    "policy_enforcement": "Check access policy compliance for",
}

# Analyst follow-up asks for turn 2+ of a session, per scenario. Cycled
# through in order so a 3-4 turn session reads like a real back-and-forth
# rather than a repeat of turn 1. `{vendor}` is filled in at generation time.
FOLLOW_UPS = {
    "vendor_risk_assessment": [
        "Can you double check that? Anything new on {vendor} since last time?",
        "Ops says they patched the issue — please re-verify and update the score.",
        "One more pass before we close this out — did the remediation hold?",
    ],
    "ai_discovery": [
        "Are you sure? {vendor} claims they don't send data to a third-party LLM — check again.",
        "Legal wants a follow-up: has {vendor} disclosed this sub-processor yet?",
        "Re-scan {vendor} now that the DPA amendment should be in place.",
    ],
    "blast_radius": [
        "What if {vendor} rotates their API tokens — does that change the blast radius?",
        "IT revoked some of the shared SSO scopes for {vendor}. Re-run the impact analysis.",
        "Double check the downstream integration list for {vendor}, it looked incomplete.",
    ],
    "attack_surface": [
        "Did the remediation ticket actually get resolved for {vendor}?",
        "Check again post-offboarding — are those scopes still live for {vendor}?",
        "Confirm the deprovisioning recommendation was applied to {vendor}.",
    ],
    "policy_enforcement": [
        "Has {vendor} pushed back on the enforcement action? Check current status.",
        "On-call says they fixed the permissions issue for {vendor} — please verify.",
        "Re-evaluate {vendor} now that the token rotation policy changed.",
    ],
}


def _sample_session_hours_ago(hours_of_history):
    """Sample how far back (in hours) a session should start, biased
    toward recent activity with a long tail back to `hours_of_history`.

    Uses an exponential distribution instead of a uniform one so the
    Tracing tab looks like steady, ongoing traffic — most sessions land
    in roughly the last quarter of the window, with a shrinking tail of
    older sessions stretching back the full `hours_of_history` — rather
    than a flat, equally-likely-any-time-in-3-days spread where any
    given recent window (e.g. "the last hour") is very sparsely
    populated.
    """
    mean_hours = max(hours_of_history / 6.0, 0.1)
    hours_ago = random.expovariate(1.0 / mean_hours)
    return min(max(hours_ago, 0.05), hours_of_history)


def build_session_turn_starts(now, turns_total, hours_of_history):
    """Pick a backdated session start time (recency-biased across the
    last `hours_of_history` hours — see `_sample_session_hours_ago`),
    then space the turns *within* that session a realistic
    conversational distance apart — tens of seconds to a few minutes.

    Keeping intra-session gaps small (instead of hours) keeps the total
    session duration short. Arize's session view renders a duration-based
    timeline; a multi-hour synthetic session can break that rendering or
    leave the session row blank, even though the individual traces are
    all valid and correctly tagged with the same session.id.
    """
    hours_ago = _sample_session_hours_ago(hours_of_history)
    starts = [now - timedelta(hours=hours_ago)]
    for _ in range(1, turns_total):
        gap = timedelta(seconds=random.uniform(20, 240))
        nxt = starts[-1] + gap
        nxt = min(nxt, now - timedelta(seconds=5))
        starts.append(nxt)
    return starts


# ---------------------------------------------------------------------------
# 4. Trace runner — wraps a scenario in its AGENT root span + session/tags
# ---------------------------------------------------------------------------

def run_trace(scenario_key, vendor, start_dt, variant, session_id, turn_idx,
              turns_total, prior_summary=None):
    cfg = SCENARIOS[scenario_key]

    with using_session(session_id=session_id), \
         using_metadata({
             "vendor": vendor["name"],
             "vendor_domain": vendor["domain"],
             "vendor_category": vendor["category"],
             "use_case": cfg["use_case"],
             "org": ORG_NAME,
             "turn.index": turn_idx,
             "turn.count": turns_total,
         }), \
         using_tags([cfg["use_case"], vendor["slug"]]):

        if turn_idx == 0:
            human_ask = f"{KICKOFF_PROMPTS[scenario_key]} {vendor['name']}."
        else:
            phrasing = FOLLOW_UPS[scenario_key][(turn_idx - 1) % len(FOLLOW_UPS[scenario_key])]
            human_ask = phrasing.format(vendor=vendor["name"])
        agent_input = f"Security analyst (turn {turn_idx + 1}/{turns_total}): {human_ask}"

        agent_cm, agent_span = open_span(
            cfg["agent_name"], "agent", start_dt,
            attributes={SpanAttributes.INPUT_VALUE: agent_input},
        )
        cursor = start_dt + timedelta(milliseconds=50)

        if turn_idx > 0 and prior_summary:
            cursor = llm_leaf(
                "recall_session_context", cursor, random.uniform(0.2, 0.5),
                model="claude-sonnet-4-5",
                prompt=f"Turn {turn_idx + 1}/{turns_total} of this session for "
                       f"{vendor['name']}. Prior finding: {prior_summary}",
                completion=f"Understood — continuing the {cfg['use_case'].replace('-', ' ')} "
                           f"session for {vendor['name']} with that context.",
                prompt_tokens=random.randint(80, 130), completion_tokens=random.randint(20, 40),
            )

        cursor, final_output, had_error = cfg["fn"](vendor, cursor, variant)
        agent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_output)
        close_span(
            agent_cm, agent_span, cursor + timedelta(milliseconds=50),
            error="one or more downstream steps failed" if had_error else None,
        )
    return cursor, final_output


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating demo sessions -> Arize project '{PROJECT_NAME}'\n")
    now = datetime.now(timezone.utc)
    trace_count = 0
    session_count = 0

    for scenario_key, cfg in SCENARIOS.items():
        for _ in range(SESSIONS_PER_SCENARIO):
            vendor = random.choice(VENDORS)
            turns_total = random.randint(TURNS_PER_SESSION_MIN, TURNS_PER_SESSION_MAX)
            turn_starts = build_session_turn_starts(now, turns_total, HOURS_OF_HISTORY)
            session_id = f"{scenario_key}-{vendor['slug']}-{uuid.uuid4().hex[:8]}"
            session_count += 1
            print(f"Session {session_count:02d}: {cfg['agent_name']:<28} "
                  f"vendor={vendor['name']:<20} turns={turns_total}")

            prior_summary = None
            for turn_idx, turn_start in enumerate(turn_starts):
                variant = random.randint(0, 4)
                _, final_output = run_trace(
                    scenario_key, vendor, turn_start, variant,
                    session_id=session_id, turn_idx=turn_idx, turns_total=turns_total,
                    prior_summary=prior_summary,
                )
                prior_summary = final_output
                trace_count += 1
                print(f"    [{trace_count:03d}] turn {turn_idx + 1}/{turns_total} "
                      f"start={turn_start.isoformat(timespec='seconds')}")

    tracer_provider.force_flush()
    print(f"\nDone — emitted {trace_count} traces across {session_count} sessions "
          f"({len(SCENARIOS)} agent use cases).")
    print(f"Open Arize AX -> Project '{PROJECT_NAME}' -> Tracing to view them.")
    print("Group by session to see the multi-turn conversations; filter by the "
          "'use_case' or 'vendor' tags to walk through each scenario.")


if __name__ == "__main__":
    main()
