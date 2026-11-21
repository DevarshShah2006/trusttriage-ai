"""The triage decision: combine rules, the ML model and (optionally) an LLM, then decide
whether the ticket can be handled automatically or must go to a person.

Design principle: a wrong label on an urgent ticket is the expensive mistake, so every
disagreement, missing signal or unsafe input sends the ticket to human review, and
priority disagreements are always resolved upwards.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import pandas as pd

from .llm import LLMClassifier, LLMUnavailable
from .preprocess import prepare
from .rules import PRIORITY_RANK, decide, match

REVIEW_THRESHOLD = 0.70        # below this combined confidence -> human review
ML_DISAGREE_P = 0.50           # ML must be at least this sure to count as a disagreement
DOWNPLAY = ("no rush", "minor issue", "tiny problem", "small thing", "when you can",
            "whenever you get a chance", "probably nothing", "no worries if busy", "sorry to bother")


@dataclass
class Decision:
    id: str
    category: str
    priority: str
    needs_review: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rule_intent: str = ""
    explanation: str = ""
    language: str = "en"
    latency_ms: float = 0.0
    suggested_category: str = ""
    suggested_priority: str = ""


def triage_one(tid: str, subject: str, body: str, ml_row: dict | None,
               llm: LLMClassifier | None = None) -> Decision:
    t0 = time.perf_counter()
    prep = prepare(subject, body)
    reasons, notes = [], []

    if "prompt_injection" in prep.flags:
        reasons.append("prompt injection attempt (instructions to the AI were removed)")
    if "log_attached" in prep.flags:
        notes.append(f"{prep.log_lines} log lines stripped before classification")
    if "angry_tone" in prep.flags:
        notes.append("angry tone: priority set from content, not tone")
    if any(d in prep.text.lower() for d in DOWNPLAY):
        notes.append("customer downplays the issue: priority set from content, not tone")

    if "no_usable_content" in prep.flags:
        return Decision(tid, "", "", "yes", 0.0, ["no usable content, ask customer for details"],
                        notes, language=prep.language,
                        latency_ms=(time.perf_counter() - t0) * 1000)

    rule = decide(match(prep.segments))

    # ---- combine rule + ML -------------------------------------------------------------
    if rule:
        cat, pri = rule["category"], rule["priority"]
        conf = 0.6
        if len(rule["categories"]) > 1:
            reasons.append("several requests in different categories: " + ", ".join(rule["categories"]))
            conf -= 0.2
    else:
        cat, pri, conf = "", "", 0.0
        reasons.append("no known intent matched")

    if ml_row is None:
        notes.append("ML model unavailable for this ticket, rules only")
        if pri in ("High", "Urgent"):
            reasons.append("high-impact ticket with no second opinion (model unavailable)")
    else:
        mc, mcp, mp, mpp = (ml_row["ml_category"], ml_row["ml_category_p"],
                            ml_row["ml_priority"], ml_row["ml_priority_p"])
        if not rule:
            cat, pri, conf = mc, mp, 0.4 * min(mcp, mpp)
        else:
            if mc == cat:
                conf += 0.4 * mcp
            elif mcp >= ML_DISAGREE_P:
                reasons.append(f"rules say {cat}, ML says {mc} ({mcp:.0%})")
                conf -= 0.2
            if mp != pri and mpp >= ML_DISAGREE_P:
                if PRIORITY_RANK[mp] > PRIORITY_RANK[pri]:
                    reasons.append(f"ML thinks priority may be higher: {mp} ({mpp:.0%})")
                    pri = mp                    # escalate-only: never lower a priority
                else:
                    notes.append(f"ML suggested lower priority {mp}; kept {pri}")

    # ---- optional LLM vote ---------------------------------------------------------------
    if llm is not None and llm.available:
        try:
            out = llm.classify(prep.text)
            if out["category"] != cat:
                reasons.append(f"LLM says category {out['category']}")
            if PRIORITY_RANK[out["priority"]] > PRIORITY_RANK.get(pri, 0):
                reasons.append(f"LLM says priority {out['priority']}")
                pri = out["priority"]
        except LLMUnavailable as e:
            notes.append(f"LLM unavailable ({e}); decided by local models")

    if prep.language != "en" and conf < 0.9:
        notes.append(f"non-English ticket ({prep.language})")

    conf = max(0.0, min(1.0, conf))
    if conf < REVIEW_THRESHOLD and not reasons:
        reasons.append(f"low confidence ({conf:.2f})")

    needs = "yes" if reasons else "no"
    return Decision(
        tid, cat, pri, needs, round(conf, 3), reasons, notes,
        rule_intent=rule["intent"] if rule else "",
        explanation=rule["why"] if rule else "",
        language=prep.language,
        latency_ms=(time.perf_counter() - t0) * 1000,
        suggested_category=cat, suggested_priority=pri,
    )


def triage_frame(df: pd.DataFrame, ml_preds: pd.DataFrame | None, llm: LLMClassifier | None = None,
                 chaos: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Triage a whole batch. `chaos` simulates the ML service failing for that share of tickets."""
    rng = random.Random(seed)
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        ml_row = None if ml_preds is None else ml_preds.iloc[i].to_dict()
        if chaos and rng.random() < chaos:
            ml_row = None
        d = triage_one(r.id, r.subject, r.body, ml_row, llm)
        rows.append({
            "id": d.id, "category": d.category, "priority": d.priority,
            "needs_review": d.needs_review, "confidence": d.confidence,
            "suggested_category": d.suggested_category, "suggested_priority": d.suggested_priority,
            "review_reasons": " | ".join(d.reasons), "notes": " | ".join(d.notes),
            "rule_intent": d.rule_intent, "explanation": d.explanation,
            "language": d.language, "latency_ms": round(d.latency_ms, 2),
            "received_at": r.received_at, "customer": r.customer, "subject": r.subject,
        })
    return pd.DataFrame(rows)
