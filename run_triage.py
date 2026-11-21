"""Triage all tickets and report how they were handled.

    python run_triage.py                      # rules + ML, writes outputs/
    python run_triage.py --chaos 0.2          # simulate the ML service failing on 20% of tickets
    python run_triage.py --llm                # add Claude as a third opinion (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from triage.llm import LLMClassifier
from triage.model import TicketModel, out_of_fold
from triage.pipeline import triage_frame
from triage.insights import build as build_insights
from triage.report import write_report

ROOT = Path(__file__).resolve().parent
REQUIRED = ["id", "received_at", "customer", "subject", "body"]


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on structural problems; keep (and flag) odd but valid rows."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Input error: missing required columns {missing}")
    if (df.id.str.strip() == "").any():
        sys.exit(f"Input error: {int((df.id.str.strip() == '').sum())} rows have no ticket id")
    dups = df.id[df.id.duplicated()].unique().tolist()
    if dups:
        sys.exit(f"Input error: duplicate ticket ids {dups[:10]}")
    print(f"Input OK: {len(df)} tickets, {df.id.nunique()} unique ids, "
          f"{int((df.subject.str.strip() + df.body.str.strip() == '').sum())} completely empty")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", default=ROOT / "data" / "tickets.csv")
    ap.add_argument("--gold", default=ROOT / "data" / "gold_labels.csv")
    ap.add_argument("--out", default=ROOT / "outputs")
    ap.add_argument("--chaos", type=float, default=0.0, help="share of tickets where the ML model is down")
    ap.add_argument("--llm", action="store_true", help="use Claude as an extra opinion if a key is set")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tickets = validate_input(pd.read_csv(args.tickets, dtype=str).fillna(""))
    gold = pd.read_csv(args.gold) if Path(args.gold).exists() else None
    texts = (tickets.subject + "\n" + tickets.body).tolist()

    # ML: out-of-fold predictions for these 300 (honest), full model saved for the API.
    ml_preds = None
    if gold is not None:
        g = tickets[["id"]].merge(gold, on="id", how="left")
        train = g.labelable.eq("yes").to_numpy()
        tr_texts = [t for t, k in zip(texts, train) if k]
        oof = out_of_fold(tr_texts, g.category[train].tolist(), g.priority[train].tolist())
        model = TicketModel().fit(tr_texts, g.category[train].tolist(), g.priority[train].tolist())
        model.save()
        ml_preds = model.predict(texts)                    # for unlabelable tickets
        ml_preds.loc[train, :] = oof.to_numpy()            # out-of-fold for labelled ones
    else:
        print("No gold labels found: running rules only.")

    llm = LLMClassifier() if args.llm else None
    if llm and not llm.available:
        print("--llm given but ANTHROPIC_API_KEY is not set: continuing with local models only.")

    res = triage_frame(tickets, ml_preds, llm=llm, chaos=args.chaos)
    elapsed = time.perf_counter() - t0

    res[["id", "category", "priority", "needs_review"]].to_csv(out / "results.csv", index=False)
    res.to_csv(out / "results_detailed.csv", index=False)
    pri_order = {"Urgent": 0, "High": 1, "Medium": 2, "Low": 3, "": 4}
    queue = res[res.needs_review == "yes"].copy()
    queue["_p"] = queue.suggested_priority.map(pri_order).fillna(4)
    queue.sort_values(["_p", "received_at"]).drop(columns="_p").to_csv(out / "review_queue.csv", index=False)

    metrics = write_report(res, gold, ml_preds, out, elapsed, args, llm)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if gold is not None:
        ml_ids = ml_preds.copy(); ml_ids["id"] = tickets.id.values
        ins = build_insights(tickets, res, gold, ml_ids)
        (out / "insights.json").write_text(json.dumps(ins, indent=1, default=str))
        best = max(ins["experiments"]["leaderboard"][:-1], key=lambda r: r["category_f1"] + r["priority_f1"])
        print(f"Experiments done; best CV model: {best['model']}; incidents: {len(ins['incidents'])}")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
