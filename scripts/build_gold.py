"""Build data/gold_labels.csv: the reference labels used to train and evaluate the ML model.

How the labels were made: the 300 tickets come from ~45 distinct request types. I read one
example of each type, assigned its category and priority by hand (the intent table in
triage/rules.py), propagated it to all tickets of that type, then read every unusual ticket
individually (multi-request, non-English, log dumps, injections, tone traps) and fixed it
below. Because the rule table encodes the same judgement, the rules' score against gold is
optimistic; the ML model's 5-fold cross-validated score is the honest generalisation number.

Run once: python scripts/build_gold.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from triage.preprocess import prepare  # noqa: E402
from triage.rules import decide, match  # noqa: E402

# Hand-checked corrections / confirmations for tickets that need individual judgement.
OVERRIDES = {
    # no usable content: nothing to label; excluded from ML training
    "T-0007": ("Other", "Low", "unclassifiable: body is just 'help'"),
    "T-0093": ("Other", "Low", "unclassifiable: empty ticket"),
    "T-0228": ("Other", "Low", "unclassifiable: 'Sent from my phone'"),
    "T-0261": ("Other", "Low", "unclassifiable: keyboard mash"),
    # tone traps, confirmed by reading
    "T-0009": ("Technical", "Urgent", "says 'minor issue' but 40 clinic units down, supply rooms locked"),
    "T-0299": ("Billing", "Urgent", "says 'no rush' but unrecognised card payments still appearing"),
    "T-0259": ("Account", "Urgent", "says 'quick query' but account email changed by someone else"),
    "T-0107": ("Account", "High", "says 'no worries' but locked out for 3 days"),
    "T-0169": ("Shipping", "Low", "furious tone, but just asks if they ship to Norway"),
    "T-0099": ("Other", "Low", "all caps, but it is a feature request (dark mode)"),
    "T-0224": ("Billing", "Low", "angry, but asks where to download a receipt"),
    "T-0284": ("Technical", "Low", "angry, but a how-to question (rename a room)"),
    # injections: label the real request, ignore the injected instruction
    "T-0072": ("Shipping", "Low", "real request: Saturday delivery; contains injection"),
    "T-0245": ("Other", "Low", "real request: widgets feature idea; contains injection"),
    # misleading subject / log dump
    "T-0232": ("Billing", "High", "double charge; the pasted device log is irrelevant"),
}


def main():
    df = pd.read_csv(ROOT / "data" / "tickets.csv").fillna("")
    rows = []
    for r in df.itertuples(index=False):
        d = decide(match(prepare(r.subject, r.body).segments))
        cat, pri, note = (d["category"], d["priority"], f"type: {d['intent']}") if d else ("Other", "Low", "")
        if r.id in OVERRIDES:
            cat, pri, note = OVERRIDES[r.id]
        rows.append({"id": r.id, "category": cat, "priority": pri,
                     "labelable": "no" if note.startswith("unclassifiable") else "yes", "note": note})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "gold_labels.csv", index=False)
    print(out.category.value_counts().to_string(), "\n", out.priority.value_counts().to_string())


if __name__ == "__main__":
    main()
