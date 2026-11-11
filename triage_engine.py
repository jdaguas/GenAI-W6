"""Deterministic, reviewable triage engine for TriageFlow.

EDUCATIONAL PROTOTYPE ONLY. The rules below are illustrative examples for
fictional adult cases. They are NOT validated clinical criteria and must be
reviewed by qualified clinicians before any real-world use.

Design principles
-----------------
* No LLM calls here. Every outcome is produced by explicit Python rules.
* Emergency flags take precedence over everything else.
* Missing information is never treated as a negative finding: a red flag must
  be *explicitly* answered "no" before a lower-acuity outcome is allowed.
* When the engine cannot safely pick a care setting it returns NEEDS_REVIEW.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
class Outcome(str, Enum):
    ER = "Emergency Department"
    URGENT_CARE = "Urgent Care"
    PRIMARY_CARE = "Primary Care"
    NEEDS_REVIEW = "Needs Review / Insufficient Information"


CATEGORIES = ("headache", "abdominal", "respiratory")
OUT_OF_SCOPE_CATEGORY = "other"
ADULT_AGE = 18


# ---------------------------------------------------------------------------
# Field catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    key: str
    type: str  # "bool" | "int" | "str" | "category"
    label: str
    question: str  # canned fallback question (also a hint for the LLM)
    description: str  # what the field means, used in the extraction prompt


FIELDS: dict[str, FieldSpec] = {
    spec.key: spec
    for spec in [
        # --- common -------------------------------------------------------
        FieldSpec("category", "category", "Main complaint",
                  "What is the main symptom you'd like help with today?",
                  "Primary complaint group: headache, abdominal, respiratory, or other."),
        FieldSpec("age", "int", "Age",
                  "How old is the patient?",
                  "Patient age in years."),
        FieldSpec("duration", "str", "Duration",
                  "How long have the symptoms been going on?",
                  "How long symptoms have lasted, as stated (e.g. '2 days')."),
        FieldSpec("severity", "int", "Severity (0-10)",
                  "On a scale of 0 to 10, how severe is it right now?",
                  "Self-reported severity on a 0-10 scale."),
        # --- shared red flags --------------------------------------------
        FieldSpec("confusion", "bool", "New confusion",
                  "Has there been any new confusion, unusual drowsiness, or trouble staying awake?",
                  "New confusion, disorientation, or decreased alertness."),
        FieldSpec("chest_pain", "bool", "Chest pain",
                  "Is there any chest pain or chest pressure?",
                  "Any chest pain, pressure, or tightness."),
        FieldSpec("fainting", "bool", "Fainting",
                  "Has the patient fainted or nearly fainted?",
                  "Fainting, passing out, or near-fainting."),
        # --- headache -----------------------------------------------------
        FieldSpec("sudden_worst_headache", "bool", "Sudden / worst-ever headache",
                  "Did the headache come on suddenly, reaching full intensity within about a minute, or is it the worst headache ever?",
                  "Thunderclap onset (peak within ~1 minute) or 'worst headache of my life'."),
        FieldSpec("neuro_deficit", "bool", "Neurological symptoms",
                  "Is there any weakness, numbness, trouble speaking, facial drooping, or loss of vision?",
                  "Focal neurological symptoms: weakness, numbness, speech trouble, facial droop, vision loss."),
        FieldSpec("fever_stiff_neck", "bool", "Fever with stiff neck",
                  "Is there a fever together with a stiff neck?",
                  "Fever combined with neck stiffness."),
        FieldSpec("head_injury", "bool", "Recent head injury",
                  "Was there any recent injury or blow to the head?",
                  "Recent head trauma."),
        # --- abdominal ----------------------------------------------------
        FieldSpec("vomiting_blood", "bool", "Vomiting blood",
                  "Has there been any vomiting of blood or material that looks like coffee grounds?",
                  "Vomiting blood or coffee-ground material."),
        FieldSpec("bloody_black_stool", "bool", "Bloody or black stool",
                  "Has there been any bloody or black, tar-like stool?",
                  "Blood in stool or black/tarry stool."),
        FieldSpec("rigid_abdomen", "bool", "Rigid abdomen",
                  "Is the belly hard or rigid, or does the pain get much worse with any movement?",
                  "Rigid/board-like abdomen or pain sharply worse with any movement."),
        FieldSpec("pregnancy_possible", "bool", "Possible pregnancy",
                  "Is there any chance the patient could be pregnant?",
                  "Patient is or could be pregnant."),
        FieldSpec("right_lower_pain", "bool", "Right lower abdominal pain",
                  "Is the pain mainly in the lower right part of the belly?",
                  "Pain concentrated in the right lower abdomen."),
        # --- respiratory --------------------------------------------------
        FieldSpec("severe_breathing_difficulty", "bool", "Severe breathing difficulty",
                  "Is the patient struggling to breathe while resting, or unable to speak in full sentences?",
                  "Severe shortness of breath at rest or unable to speak full sentences."),
        FieldSpec("blue_lips", "bool", "Blue or gray lips",
                  "Are the lips or face turning blue or gray?",
                  "Bluish or gray lips/face (cyanosis)."),
        FieldSpec("coughing_blood", "bool", "Coughing blood",
                  "Has the patient been coughing up blood?",
                  "Coughing up blood."),
        FieldSpec("exertional_dyspnea", "bool", "Short of breath on exertion",
                  "Does the patient get short of breath with light activity, like walking across a room?",
                  "Shortness of breath with mild exertion."),
        FieldSpec("wheezing", "bool", "Wheezing",
                  "Is there any wheezing or whistling sound when breathing?",
                  "Audible wheezing."),
        # --- shared urgent ------------------------------------------------
        FieldSpec("fever", "bool", "Fever",
                  "Is there a fever of 100.4°F (38°C) or higher?",
                  "Measured or felt fever >= 100.4°F / 38°C."),
        FieldSpec("persistent_vomiting", "bool", "Persistent vomiting",
                  "Is there repeated vomiting, or trouble keeping fluids down?",
                  "Repeated vomiting or inability to keep fluids down."),
    ]
}


# ---------------------------------------------------------------------------
# Rules (data, so they can be reviewed in one place)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    outcome: Outcome
    description: str
    fields: tuple[str, ...]
    predicate: Callable[[dict[str, Any]], bool]
    categories: tuple[str, ...] = CATEGORIES  # complaint groups that must screen it


def _is_true(key: str) -> Callable[[dict[str, Any]], bool]:
    return lambda facts: facts.get(key) is True


def _severity_at_least(n: int) -> Callable[[dict[str, Any]], bool]:
    return lambda facts: isinstance(facts.get("severity"), int) and facts["severity"] >= n


EMERGENCY_RULES: tuple[Rule, ...] = (
    Rule("ER-01", Outcome.ER, "New confusion or decreased alertness", ("confusion",), _is_true("confusion")),
    Rule("ER-02", Outcome.ER, "Chest pain or pressure", ("chest_pain",), _is_true("chest_pain"),
         ("abdominal", "respiratory")),
    Rule("ER-03", Outcome.ER, "Fainting or near-fainting", ("fainting",), _is_true("fainting"),
         ("abdominal",)),
    Rule("ER-H1", Outcome.ER, "Sudden-onset / worst-ever headache", ("sudden_worst_headache",),
         _is_true("sudden_worst_headache"), ("headache",)),
    Rule("ER-H2", Outcome.ER, "Focal neurological symptoms", ("neuro_deficit",),
         _is_true("neuro_deficit"), ("headache",)),
    Rule("ER-H3", Outcome.ER, "Fever with stiff neck", ("fever_stiff_neck",),
         _is_true("fever_stiff_neck"), ("headache",)),
    Rule("ER-H4", Outcome.ER, "Headache after head injury", ("head_injury",),
         _is_true("head_injury"), ("headache",)),
    Rule("ER-A1", Outcome.ER, "Vomiting blood", ("vomiting_blood",), _is_true("vomiting_blood"), ("abdominal",)),
    Rule("ER-A2", Outcome.ER, "Bloody or black stool", ("bloody_black_stool",),
         _is_true("bloody_black_stool"), ("abdominal",)),
    Rule("ER-A3", Outcome.ER, "Rigid abdomen / pain worse with movement", ("rigid_abdomen",),
         _is_true("rigid_abdomen"), ("abdominal",)),
    Rule("ER-R1", Outcome.ER, "Severe breathing difficulty at rest", ("severe_breathing_difficulty",),
         _is_true("severe_breathing_difficulty"), ("respiratory",)),
    Rule("ER-R2", Outcome.ER, "Blue or gray lips/face", ("blue_lips",), _is_true("blue_lips"), ("respiratory",)),
    Rule("ER-R3", Outcome.ER, "Coughing up blood", ("coughing_blood",), _is_true("coughing_blood"),
         ("respiratory",)),
)

URGENT_RULES: tuple[Rule, ...] = (
    Rule("UC-01", Outcome.URGENT_CARE, "Fever >= 100.4°F / 38°C", ("fever",), _is_true("fever")),
    Rule("UC-02", Outcome.URGENT_CARE, "Severity 7/10 or higher", ("severity",), _severity_at_least(7)),
    Rule("UC-H1", Outcome.URGENT_CARE, "Headache with persistent vomiting", ("persistent_vomiting",),
         _is_true("persistent_vomiting"), ("headache",)),
    Rule("UC-A1", Outcome.URGENT_CARE, "Abdominal pain with persistent vomiting", ("persistent_vomiting",),
         _is_true("persistent_vomiting"), ("abdominal",)),
    Rule("UC-A2", Outcome.URGENT_CARE, "Abdominal pain with possible pregnancy", ("pregnancy_possible",),
         _is_true("pregnancy_possible"), ("abdominal",)),
    Rule("UC-A3", Outcome.URGENT_CARE, "Right lower abdominal pain", ("right_lower_pain",),
         _is_true("right_lower_pain"), ("abdominal",)),
    Rule("UC-R1", Outcome.URGENT_CARE, "Short of breath with light activity", ("exertional_dyspnea",),
         _is_true("exertional_dyspnea"), ("respiratory",)),
    Rule("UC-R2", Outcome.URGENT_CARE, "Wheezing", ("wheezing",), _is_true("wheezing"), ("respiratory",)),
)

REQUIRED_COMMON = ("age", "duration", "severity")

# Boolean fields where a "yes" is sticky (never silently downgraded).
SAFETY_FIELDS: frozenset[str] = frozenset(
    f for r in EMERGENCY_RULES + URGENT_RULES for f in r.fields if FIELDS[f].type == "bool"
)
RED_FLAG_FIELDS: frozenset[str] = frozenset(f for r in EMERGENCY_RULES for f in r.fields)


def _fields_for(rules: tuple[Rule, ...], category: str) -> list[str]:
    seen: list[str] = []
    for r in rules:
        if category in r.categories:
            for f in r.fields:
                if f not in seen:
                    seen.append(f)
    return seen


def red_flag_fields_for(category: str) -> list[str]:
    return _fields_for(EMERGENCY_RULES, category)


def urgent_fields_for(category: str) -> list[str]:
    return _fields_for(URGENT_RULES, category)


# ---------------------------------------------------------------------------
# Fact validation and merging
# ---------------------------------------------------------------------------
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def coerce_value(key: str, value: Any) -> Any:
    """Validate a value for a known field. Returns None if invalid/unknown."""
    spec = FIELDS.get(key)
    if spec is None or value is None:
        return None
    if spec.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        return None
    if spec.type == "int":
        if isinstance(value, bool):
            return None
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
        if key == "severity" and not 0 <= n <= 10:
            return None
        if key == "age" and not 0 <= n <= 120:
            return None
        return n
    if spec.type == "category":
        v = str(value).strip().lower()
        return v if v in CATEGORIES or v == OUT_OF_SCOPE_CATEGORY else None
    if spec.type == "str":
        v = str(value).strip()
        return v or None
    return None


def sanitize_facts(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unknown keys and invalid values (e.g. anything an LLM invents)."""
    clean: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        v = coerce_value(key, value)
        if v is not None:
            clean[key] = v
    return clean


def merge_facts(
    current: dict[str, Any], new: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Merge newly extracted facts into the current fact set.

    Safety rule: once a safety finding is recorded as present (True), a later
    "no" does NOT clear it. The conflict is recorded as a contradiction so the
    case is escalated / reviewed rather than silently downgraded.
    Returns (merged_facts, new_contradictions).
    """
    merged = dict(current)
    contradictions: list[str] = []
    for key, value in sanitize_facts(new).items():
        old = merged.get(key)
        if key in SAFETY_FIELDS and old is True and value is False:
            contradictions.append(
                f"{FIELDS[key].label}: previously reported present, later reported absent (kept as present)."
            )
            continue
        if key == "age" and old is not None and old != value:
            contradictions.append(f"Age: reported as {old} and later as {value}.")
        merged[key] = value
    return merged, contradictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class TriggeredRule:
    id: str
    description: str
    evidence: dict[str, Any]


@dataclass
class TriageResult:
    outcome: Outcome
    triggered_rules: list[TriggeredRule] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        """True when more questions could not change the outcome."""
        return self.outcome is not Outcome.NEEDS_REVIEW or not self.missing_fields

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


def _fire(rules: tuple[Rule, ...], facts: dict[str, Any], category: Optional[str]) -> list[TriggeredRule]:
    fired = []
    for r in rules:
        # Emergency rules fire regardless of complaint group: a disclosed red flag
        # always counts. Urgent rules only apply to their complaint groups.
        if r.outcome is not Outcome.ER and category not in r.categories:
            continue
        if r.predicate(facts):
            fired.append(TriggeredRule(r.id, r.description, {f: facts.get(f) for f in r.fields}))
    return fired


def check_emergency(facts: dict[str, Any]) -> list[TriggeredRule]:
    """Fast safety screen: any disclosed emergency warning sign."""
    return _fire(EMERGENCY_RULES, facts, facts.get("category"))


def evaluate(facts: dict[str, Any], contradictions: list[str] | None = None) -> TriageResult:
    """Apply rules in strict precedence order and return a TriageResult."""
    contradictions = contradictions or []
    category = facts.get("category")

    # 1. Emergency flags win, always.
    er = check_emergency(facts)
    if er:
        return TriageResult(Outcome.ER, er, reasons=["One or more predefined emergency warning signs were reported."])

    # 2. Scope checks.
    if category is None:
        return TriageResult(Outcome.NEEDS_REVIEW, missing_fields=["category"],
                            reasons=["The main complaint has not been identified yet."])
    if category == OUT_OF_SCOPE_CATEGORY:
        return TriageResult(Outcome.NEEDS_REVIEW,
                            reasons=["This complaint is outside the prototype's supported categories "
                                     "(headache, abdominal pain, respiratory symptoms)."])
    age = facts.get("age")
    if isinstance(age, int) and age < ADULT_AGE:
        return TriageResult(Outcome.NEEDS_REVIEW,
                            reasons=["The prototype only covers fictional adult cases (age 18+)."])

    # 3. Contradictions cannot be resolved by the rules -> human review.
    if contradictions:
        return TriageResult(Outcome.NEEDS_REVIEW,
                            reasons=["Contradictory information was reported:"] + contradictions)

    # 4. Every red flag for the complaint must be explicitly screened (answered no).
    missing = [f for f in red_flag_fields_for(category) if facts.get(f) is None]
    missing += [f for f in REQUIRED_COMMON if facts.get(f) is None]
    if missing:
        return TriageResult(Outcome.NEEDS_REVIEW, missing_fields=missing,
                            reasons=["Required safety-screening information is missing; "
                                     "absence of a reported warning sign is not treated as a 'no'."])

    # 5. Urgent findings.
    urgent = _fire(URGENT_RULES, facts, category)
    if urgent:
        return TriageResult(Outcome.URGENT_CARE, urgent,
                            reasons=["No emergency warning signs reported; urgent-care criteria met."])

    missing_urgent = [f for f in urgent_fields_for(category) if facts.get(f) is None]
    if missing_urgent:
        return TriageResult(Outcome.NEEDS_REVIEW, missing_fields=missing_urgent,
                            reasons=["Urgent-care criteria have not all been screened."])

    # 6. All screens explicitly negative.
    return TriageResult(Outcome.PRIMARY_CARE,
                        reasons=["All example emergency and urgent-care screens were explicitly answered "
                                 "negative for this complaint group."])
