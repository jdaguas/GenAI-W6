"""LangGraph workflow for TriageFlow.

One graph run happens per user message:

    START -> extract -> safety_screen --(emergency flag)--> finalize -> END
                                   \\--(no flag)--> assess --(info missing)--> ask -> END
                                                         \\--(decidable / budget spent)--> finalize -> END

* ``extract``       the LLM agent turns the message into validated facts.
* ``safety_screen`` deterministic emergency check runs on EVERY turn, so red
                    flags are caught immediately instead of at the end of intake.
* ``assess``        the rule engine says what is still missing; the workflow
                    chooses the next field to ask about (deterministically).
* ``ask``           the LLM agent phrases ONE follow-up question.
* ``finalize``      the rule engine produces the outcome. The LLM is not consulted.

State persists between turns through a LangGraph checkpointer keyed by thread id.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent import IntakeAgent
from triage_engine import FIELDS, Outcome, check_emergency, evaluate, merge_facts

MAX_QUESTIONS = 26          # hard cap; the per-field limit below is the real bound
MAX_ASKS_PER_FIELD = 2      # re-ask an unanswered field at most once


class TriageState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], operator.add]
    facts: dict[str, Any]
    contradictions: list[str]
    asked: dict[str, int]          # field -> times asked
    pending_field: Optional[str]   # field the last question targeted
    questions_asked: int
    agent_notes: list[str]
    result: Optional[dict[str, Any]]
    status: Literal["intake", "complete"]


def initial_state() -> TriageState:
    return TriageState(
        messages=[], facts={}, contradictions=[], asked={}, pending_field=None,
        questions_asked=0, agent_notes=[], result=None, status="intake",
    )


def _last(messages: list[dict[str, str]], role: str) -> Optional[str]:
    for m in reversed(messages):
        if m["role"] == role:
            return m["content"]
    return None


def next_field_to_ask(missing: list[str], asked: dict[str, int]) -> Optional[str]:
    """Pick the first missing field (engine order = safety first) not over-asked."""
    for f in missing:
        if asked.get(f, 0) < MAX_ASKS_PER_FIELD:
            return f
    return None


OUTCOME_MESSAGES = {
    Outcome.ER: "Based on the example rules, this fictional case is routed to the **Emergency Department**.",
    Outcome.URGENT_CARE: "Based on the example rules, this fictional case is routed to **Urgent Care**.",
    Outcome.PRIMARY_CARE: "Based on the example rules, this fictional case is routed to **Primary Care**.",
    Outcome.NEEDS_REVIEW: "The example rules **cannot safely select a care setting** for this case. "
                          "It is flagged for **human review**.",
}


def build_graph(agent: IntakeAgent, checkpointer: Any = None):
    def extract(state: TriageState) -> dict:
        messages = state.get("messages", [])
        user_msg = _last(messages, "user") or ""
        last_q = _last(messages, "assistant")
        ext = agent.extract(user_msg, last_q, state.get("pending_field"), state.get("facts", {}))
        facts, new_contra = merge_facts(state.get("facts", {}), ext.facts)
        notes = state.get("agent_notes", []) + ([ext.notes] if ext.notes else [])
        return {
            "facts": facts,
            "contradictions": state.get("contradictions", []) + new_contra,
            "agent_notes": notes,
        }

    def safety_screen(state: TriageState) -> dict:
        return {}  # pure routing node; the check itself lives in route_after_screen

    def route_after_screen(state: TriageState) -> str:
        return "finalize" if check_emergency(state.get("facts", {})) else "assess"

    def assess(state: TriageState) -> dict:
        return {}

    def route_after_assess(state: TriageState) -> str:
        result = evaluate(state.get("facts", {}), state.get("contradictions", []))
        if result.is_final:
            return "finalize"
        if state.get("questions_asked", 0) >= MAX_QUESTIONS:
            return "finalize"
        if next_field_to_ask(result.missing_fields, state.get("asked", {})) is None:
            return "finalize"
        return "ask"

    def ask(state: TriageState) -> dict:
        result = evaluate(state.get("facts", {}), state.get("contradictions", []))
        asked = dict(state.get("asked", {}))
        field_key = next_field_to_ask(result.missing_fields, asked)
        assert field_key is not None  # guaranteed by route_after_assess
        question = agent.phrase_question(field_key, state.get("facts", {}), state.get("messages", []))
        asked[field_key] = asked.get(field_key, 0) + 1
        return {
            "messages": [{"role": "assistant", "content": question}],
            "asked": asked,
            "pending_field": field_key,
            "questions_asked": state.get("questions_asked", 0) + 1,
        }

    def finalize(state: TriageState) -> dict:
        # Deterministic engine only. Nothing from the agent can alter this.
        result = evaluate(state.get("facts", {}), state.get("contradictions", []))
        return {
            "result": result.to_dict(),
            "status": "complete",
            "pending_field": None,
            "messages": [{"role": "assistant", "content": OUTCOME_MESSAGES[result.outcome]}],
        }

    g = StateGraph(TriageState)
    g.add_node("extract", extract)
    g.add_node("safety_screen", safety_screen)
    g.add_node("assess", assess)
    g.add_node("ask", ask)
    g.add_node("finalize", finalize)
    g.add_edge(START, "extract")
    g.add_edge("extract", "safety_screen")
    g.add_conditional_edges("safety_screen", route_after_screen, ["finalize", "assess"])
    g.add_conditional_edges("assess", route_after_assess, ["ask", "finalize"])
    g.add_edge("ask", END)
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class TriageSession:
    """Convenience wrapper: one fictional case == one LangGraph thread."""

    def __init__(self, agent: IntakeAgent):
        self.agent = agent
        self.graph = build_graph(agent)
        self.thread_id = str(uuid.uuid4())
        self._started = False

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def send(self, text: str) -> TriageState:
        if self.state.get("status") == "complete":
            raise RuntimeError("This case is complete. Start a new case.")
        update: dict[str, Any] = {"messages": [{"role": "user", "content": text}]}
        if not self._started:
            update = {**initial_state(), **update}
            self._started = True
        return self.graph.invoke(update, self.config)

    @property
    def state(self) -> TriageState:
        snap = self.graph.get_state(self.config)
        return snap.values if snap and snap.values else initial_state()


def handoff_summary(state: TriageState) -> str:
    """Plain-text summary for a (fictional) clinician handoff."""
    facts = state.get("facts", {})
    result = state.get("result") or {}
    lines = [
        "TRIAGEFLOW HANDOFF SUMMARY — EDUCATIONAL PROTOTYPE, FICTIONAL CASE, NOT MEDICAL ADVICE",
        "",
        f"Outcome (example rules): {result.get('outcome', 'not evaluated')}",
        "",
        "Reported findings:",
    ]
    for key, spec in FIELDS.items():
        if key in facts:
            v = facts[key]
            v = "yes" if v is True else "no" if v is False else v
            lines.append(f"  - {spec.label}: {v}")
    if result.get("triggered_rules"):
        lines += ["", "Rules triggered:"]
        lines += [f"  - [{r['id']}] {r['description']}" for r in result["triggered_rules"]]
    if result.get("missing_fields"):
        lines += ["", "Not established:"]
        lines += [f"  - {FIELDS[f].label}" for f in result["missing_fields"]]
    if state.get("contradictions"):
        lines += ["", "Contradictions:"] + [f"  - {c}" for c in state["contradictions"]]
    if result.get("reasons"):
        lines += ["", "Reasoning:"] + [f"  - {r}" for r in result["reasons"]]
    lines += ["", "Transcript:"]
    lines += [f"  {m['role'].upper()}: {m['content']}" for m in state.get("messages", [])]
    return "\n".join(lines)
