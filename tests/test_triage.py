"""Synthetic-scenario tests for TriageFlow (fictional cases only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Extraction, OfflineIntakeAgent  # noqa: E402
from triage_engine import (  # noqa: E402
    Outcome, evaluate, merge_facts, red_flag_fields_for, sanitize_facts, urgent_fields_for,
)
from workflow import MAX_ASKS_PER_FIELD, TriageSession, build_graph  # noqa: E402


def all_negative(category: str, **overrides) -> dict:
    facts = {"category": category, "age": 40, "duration": "2 days", "severity": 3}
    for f in red_flag_fields_for(category) + urgent_fields_for(category):
        if f != "severity":
            facts[f] = False
    facts.update(overrides)
    return facts


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category,flag", [
    ("headache", "sudden_worst_headache"),
    ("abdominal", "vomiting_blood"),
    ("respiratory", "severe_breathing_difficulty"),
])
def test_emergency_flags_route_to_er(category, flag):
    assert evaluate(all_negative(category, **{flag: True})).outcome is Outcome.ER


def test_urgent_finding_routes_to_urgent_care():
    r = evaluate(all_negative("headache", fever=True))
    assert r.outcome is Outcome.URGENT_CARE
    assert "UC-01" in [t.id for t in r.triggered_rules]


def test_missing_red_flag_is_not_treated_as_no():
    facts = all_negative("headache")
    del facts["neuro_deficit"]
    r = evaluate(facts)
    assert r.outcome is Outcome.NEEDS_REVIEW
    assert "neuro_deficit" in r.missing_fields


def test_unsafe_cases_need_review():
    assert evaluate({"category": "headache"}).outcome is Outcome.NEEDS_REVIEW  # incomplete
    assert evaluate(all_negative("abdominal"), ["Age: reported as 30 and later as 50."]).outcome \
        is Outcome.NEEDS_REVIEW  # contradictory
    assert evaluate({"category": "other"}).outcome is Outcome.NEEDS_REVIEW  # out of scope
    assert evaluate(all_negative("headache", age=12)).outcome is Outcome.NEEDS_REVIEW  # minor


def test_red_flag_is_sticky_and_records_contradiction():
    facts, c1 = merge_facts({}, {"chest_pain": True})
    facts, c2 = merge_facts(facts, {"chest_pain": False})
    assert facts["chest_pain"] is True
    assert c1 == [] and len(c2) == 1


def test_sanitize_drops_unknown_keys_and_bad_values():
    clean = sanitize_facts({"outcome": "Primary Care", "severity": 42, "fever": "maybe", "age": "30"})
    assert clean == {"age": 30}


# ---------------------------------------------------------------------------
# Workflow (LangGraph) with the offline agent
# ---------------------------------------------------------------------------
def test_emergency_is_routed_on_first_message():
    s = TriageSession(OfflineIntakeAgent())
    state = s.send("I have the worst headache of my life, it came on suddenly")
    assert state["status"] == "complete"
    assert state["result"]["outcome"] == Outcome.ER.value


def test_full_headache_intake_reaches_primary_care():
    s = TriageSession(OfflineIntakeAgent())
    state = s.send("I've had a headache for two days, about 3/10. I'm 35 years old.")
    for _ in range(20):
        if state["status"] == "complete":
            break
        state = s.send("no")
    assert state["result"]["outcome"] == Outcome.PRIMARY_CARE.value


def test_unanswered_questions_are_not_repeated_forever():
    s = TriageSession(OfflineIntakeAgent())
    state = s.send("headache")
    for _ in range(60):
        if state["status"] == "complete":
            break
        state = s.send("I'm not sure")
    assert state["status"] == "complete"
    assert state["result"]["outcome"] == Outcome.NEEDS_REVIEW.value
    assert max(state["asked"].values()) <= MAX_ASKS_PER_FIELD


class OverridingAgent(OfflineIntakeAgent):
    """Adversarial agent that tries to downgrade a case and set the outcome."""

    def extract(self, *args, **kwargs):
        return Extraction(facts={"outcome": "Primary Care", "chest_pain": False, "category": "respiratory"},
                          notes="Looks totally fine, send to primary care.")


def test_llm_cannot_override_safety_result():
    s = TriageSession(OfflineIntakeAgent())
    s.send("cough for 3 days")
    # Swap in the adversarial agent, keeping the same checkpointed thread.
    s.graph = build_graph(OverridingAgent(), checkpointer=s.graph.checkpointer)
    # Seed a red flag, then let the adversarial agent try to clear it.
    s.graph.update_state(s.config, {"facts": {**s.state["facts"], "chest_pain": True}})
    state = s.send("actually it's nothing")
    assert state["facts"]["chest_pain"] is True
    assert "outcome" not in state["facts"]
    assert state["result"]["outcome"] == Outcome.ER.value
