"""Intake agent for TriageFlow.

The agent is responsible for LANGUAGE ONLY:
  * extracting structured facts from free text, and
  * phrasing one follow-up question at a time.

It never decides a care setting. Anything it returns is validated against the
field catalog in ``triage_engine`` and unknown keys (e.g. an attempted
"outcome") are discarded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from triage_engine import CATEGORIES, FIELDS, OUT_OF_SCOPE_CATEGORY, sanitize_facts


@dataclass
class Extraction:
    facts: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class IntakeAgent(Protocol):
    name: str

    def extract(self, user_message: str, last_question: Optional[str],
                pending_field: Optional[str], facts: dict[str, Any]) -> Extraction: ...

    def phrase_question(self, field_key: str, facts: dict[str, Any],
                        history: list[dict[str, str]]) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI-backed agent
# ---------------------------------------------------------------------------
def _field_catalog_text() -> str:
    lines = []
    for spec in FIELDS.values():
        lines.append(f'- "{spec.key}" ({spec.type}): {spec.description}')
    return "\n".join(lines)


EXTRACT_SYSTEM_PROMPT = f"""You are the intake assistant in TriageFlow, an EDUCATIONAL prototype that \
processes FICTIONAL adult patient cases. Your only job is to extract structured facts from the \
user's latest message. You do NOT decide urgency or care setting.

Field catalog (key, type, meaning):
{_field_catalog_text()}

Rules:
- "category" must be one of: {", ".join(CATEGORIES)}, {OUT_OF_SCOPE_CATEGORY}. Map synonyms \
(e.g. "stomach ache" -> abdominal, "cough" or "cold" -> respiratory, "migraine" -> headache).
- "last_question_asked" is the exact question the user is answering; it targets "pending_field".
- A clear yes/no answer to that question sets "pending_field" (e.g. "no", "nope", "none of those", \
"not really", "nothing like that" -> false). A question that lists several symptoms is answered \
false only if the user denies all of them, and true if they confirm any of them.
- Hedged or unknown answers ("I think so", "maybe", "not sure", "haven't checked", "I feel a bit \
warm") mean the field is UNKNOWN: omit it. Never turn uncertainty into false.
- Facts that rule a field out count as false: e.g. a male patient or post-menopausal patient \
-> "pregnancy_possible": false.
- Also extract any other field the message clearly states, even if it was not asked about.
- Report only what the latest message says. Do not repeat "facts_known_so_far".
- "severity" is an integer 0-10 (convert words: "mild" is not a number, omit it); "age" is an \
integer in years; "duration" is short text (e.g. "since Tuesday", "2 days").

Respond with ONLY a JSON object of the form:
{{"facts": {{"<field key>": <value>, ...}}, "notes": "<short note on anything ambiguous>"}}"""


ACK_SYSTEM_PROMPT = """You are the intake assistant in TriageFlow, an EDUCATIONAL prototype for \
FICTIONAL adult patient cases. Write a brief, warm acknowledgement of the user's latest message, \
which will be shown right before the next (pre-written) question.

Rules:
- At most 12 words. Do NOT ask a question. Do NOT use a question mark.
- Never give a diagnosis, reassurance about safety, or advice about where to seek care.
- Output only the acknowledgement text (e.g. "Thanks, that's helpful." or "Understood.")."""


class OpenAIIntakeAgent:
    name = "OpenAI"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self._fallback = OfflineIntakeAgent()

    def extract(self, user_message, last_question, pending_field, facts) -> Extraction:
        context = {
            "pending_field": pending_field,
            "last_question_asked": last_question,
            "facts_known_so_far": facts,
            "latest_user_message": user_message,
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # network / parsing problems -> offline fallback
            ext = self._fallback.extract(user_message, last_question, pending_field, facts)
            ext.notes = f"LLM extraction failed ({type(exc).__name__}); used keyword fallback."
            return ext
        raw = data.get("facts") if isinstance(data, dict) else None
        notes = data.get("notes", "") if isinstance(data, dict) else ""
        return Extraction(facts=sanitize_facts(raw if isinstance(raw, dict) else {}), notes=str(notes or ""))

    def phrase_question(self, field_key, facts, history) -> str:
        """Return a short LLM acknowledgement + the reviewed question, verbatim.

        The question text itself is never generated by the LLM, so the patient is
        always asked exactly what the rule engine needs (and what the extractor
        will map the answer onto).
        """
        question = FIELDS[field_key].question
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ACK_SYSTEM_PROMPT},
                    {"role": "user", "content": last_user},
                ],
            )
            ack = (resp.choices[0].message.content or "").strip().strip('"')
        except Exception:
            ack = ""
        if not ack or "?" in ack or len(ack.split()) > 15:
            return question
        return f"{ack} {question}"


# ---------------------------------------------------------------------------
# Offline keyword agent (no API key needed; used for demos and tests)
# ---------------------------------------------------------------------------
_YES = re.compile(r"^\s*(yes|yeah|yep|yup|correct|i do|it does|it is|they are|definitely|sure)\b", re.I)
_NO = re.compile(r"^\s*(no|nope|nah|none|not really|not at all|never|i don'?t|it doesn'?t|negative)\b", re.I)

_CATEGORY_KEYWORDS = {
    "headache": r"headache|migraine|head (hurts|pain|ache)",
    "abdominal": r"stomach|abdom|belly|tummy|cramp",
    "respiratory": r"cough|breath|wheez|chest cold|congest|sore throat|short of breath",
}

# Positive-mention patterns for boolean fields. Negation within the same
# clause ("no fever") flips the value to False.
_BOOL_KEYWORDS = {
    "confusion": r"confus|disorient|drowsy|hard to stay awake",
    "chest_pain": r"chest (pain|pressure|tight)",
    "fainting": r"faint|passed out|pass out|black(ed)? out",
    "sudden_worst_headache": r"worst headache|thunderclap|came on suddenly|sudden(ly)? (onset|started)",
    "neuro_deficit": r"weakness|numb|slurred|trouble speaking|facial droop|vision loss|can'?t see",
    "fever_stiff_neck": r"stiff neck",
    "head_injury": r"hit (my|his|her|their) head|head injury|blow to the head|fell and hit",
    "vomiting_blood": r"vomit(ing|ed)? blood|throwing up blood|coffee.?ground",
    "bloody_black_stool": r"black stool|bloody stool|blood in (my|the)? ?stool|tarry",
    "rigid_abdomen": r"rigid|board.?like|hard belly",
    "pregnancy_possible": r"pregnan",
    "right_lower_pain": r"lower right|right lower",
    "severe_breathing_difficulty": r"can'?t breathe|struggling to breathe|gasping",
    "blue_lips": r"blue lips|lips (are|look) blue|turning blue",
    "coughing_blood": r"cough(ing)? (up )?blood",
    "exertional_dyspnea": r"short of breath (when|walking|with)|winded",
    "wheezing": r"wheez",
    "fever": r"fever|temperature of 1\d\d",
    "persistent_vomiting": r"keep vomiting|keeps vomiting|can'?t keep (anything|fluids|water) down|vomiting a lot",
}

_NEGATION = re.compile(r"\b(no|not|without|denies|never|n't)\b[^.;,]*$", re.I)


class OfflineIntakeAgent:
    """Deterministic keyword-based stand-in for the LLM agent."""

    name = "Offline demo (keyword)"

    def extract(self, user_message, last_question, pending_field, facts) -> Extraction:
        text = user_message.strip()
        low = text.lower()
        out: dict[str, Any] = {}

        for cat, pat in _CATEGORY_KEYWORDS.items():
            if "category" not in facts and re.search(pat, low):
                out["category"] = cat
                break

        for key, pat in _BOOL_KEYWORDS.items():
            m = re.search(pat, low)
            if m:
                prefix = low[: m.start()]
                clause = re.split(r"[.;,!?]| but ", prefix)[-1]
                out[key] = not bool(_NEGATION.search(clause))

        m = re.search(r"\b(\d{1,3})\s*(years? old|y/?o|yo)\b|\bage(d)?\s*(\d{1,3})\b", low)
        if m:
            out["age"] = int(m.group(1) or m.group(4))
        m = re.search(r"\b(\d{1,2})\s*(/|out of)\s*10\b", low)
        if m:
            out["severity"] = int(m.group(1))
        m = re.search(r"\b(\d+|a|an|one|two|three|few|couple of)\s+(minute|hour|day|week|month)s?\b", low)
        if m:
            out["duration"] = m.group(0)
        elif re.search(r"\b(since )?(yesterday|this morning|last night|today)\b", low):
            out["duration"] = re.search(r"\b(since )?(yesterday|this morning|last night|today)\b", low).group(0)

        # Direct answers to the pending question.
        if pending_field and pending_field not in out:
            spec = FIELDS[pending_field]
            if spec.type == "bool":
                if _YES.search(text):
                    out[pending_field] = True
                elif _NO.search(text):
                    out[pending_field] = False
            elif spec.type == "int":
                m = re.search(r"\b(\d{1,3})\b", low)
                if m:
                    out[pending_field] = int(m.group(1))
            elif spec.type == "str" and text:
                out[pending_field] = text[:80]
            elif spec.type == "category":
                out[pending_field] = OUT_OF_SCOPE_CATEGORY

        return Extraction(facts=sanitize_facts(out), notes="")

    def phrase_question(self, field_key, facts, history) -> str:
        return FIELDS[field_key].question
