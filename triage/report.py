"""End-of-run report: how the tickets were handled, how good the decisions were, and charts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:  # charts are optional
    plt = None
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

CATS = ["Billing", "Technical", "Account", "Shipping", "Other"]
PRIS = ["Urgent", "High", "Medium", "Low"]
RANK = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}


def _pct(a, b):
    return f"{(a / b * 100 if b else 0):.1f}%"


def _reason_key(r: str) -> str:
    r = r.lower()
    for key, label in [("injection", "prompt injection"), ("several requests", "multiple requests"),
                       ("no usable", "no usable content"), ("no known intent", "no intent matched"),
                       ("ml thinks priority", "possible under-prioritisation"),
                       ("rules say", "rules vs ML disagree"), ("no second opinion", "model unavailable, high impact"),
                       ("llm", "LLM disagrees"), ("low confidence", "low confidence")]:
        if key in r:
            return label
    return r[:40]


def write_report(res: pd.DataFrame, gold, ml_preds, out: Path, elapsed: float, args, llm) -> dict:
    n = len(res)
    auto = res[res.needs_review == "no"]
    rev = res[res.needs_review == "yes"]
    reasons = Counter(_reason_key(r) for rr in rev.review_reasons for r in rr.split(" | ") if r)
    m: dict = {"tickets": n, "auto": len(auto), "review": len(rev), "seconds": round(elapsed, 2),
               "review_reasons": dict(reasons)}

    L = ["# Triage run report", "",
         f"Processed **{n} tickets** in **{elapsed:.1f}s** "
         f"(median {res.latency_ms.median():.1f} ms / ticket for the decision step).", "",
         "## How the tickets were handled", "",
         "| Outcome | Tickets | Share |", "|---|---:|---:|",
         f"| Handled automatically | {len(auto)} | {_pct(len(auto), n)} |",
         f"| Sent to human review | {len(rev)} | {_pct(len(rev), n)} |", "",
         "### Automatically handled, by category and priority", "",
         "| Category | " + " | ".join(PRIS) + " | Total |", "|---" * (len(PRIS) + 2) + "|"]
    for c in CATS:
        row = [int(((auto.category == c) & (auto.priority == p)).sum()) for p in PRIS]
        L.append(f"| {c} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    L += ["", "### Why tickets went to review", "", "| Reason | Tickets |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in reasons.most_common()]

    urgent = res[res.suggested_priority == "Urgent"].sort_values("received_at")
    L += ["", f"### Urgent tickets ({len(urgent)}), page the on-call team", "",
          "| id | received | category | why | review? |", "|---|---|---|---|---|"]
    L += [f"| {r.id} | {r.received_at} | {r.suggested_category} | {r.explanation} | {r.needs_review} |"
          for r in urgent.itertuples()]

    flags = {
        "prompt injections neutralised": int(res.review_reasons.str.contains("injection").sum()),
        "log dumps stripped": int(res.notes.str.contains("log lines stripped").sum()),
        "non-English tickets": int((res.language != "en").sum()),
        "tone traps (angry/downplayed) priced by content": int(res.notes.str.contains("not tone").sum()),
        "ML model unavailable (chaos/real)": int(res.notes.str.contains("ML model unavailable").sum()),
    }
    m["robustness"] = flags
    L += ["", "### Robustness", "", "| Check | Tickets |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in flags.items()]
    if llm is not None:
        L += ["", f"LLM calls: {llm.stats}"]

    if gold is not None:
        g = res.merge(gold.rename(columns={"category": "g_cat", "priority": "g_pri"}), on="id")
        g = g[g.labelable == "yes"]
        ga = g[g.needs_review == "no"]
        cat_acc = (ga.category == ga.g_cat).mean()
        pri_acc = (ga.priority == ga.g_pri).mean()
        both = ((ga.category == ga.g_cat) & (ga.priority == ga.g_pri)).mean()
        under = ga[ga.priority.map(RANK) < ga.g_pri.map(RANK)]
        gu = g[g.g_pri == "Urgent"]
        urgent_caught = ((gu.suggested_priority == "Urgent") | (gu.needs_review == "yes")).mean()
        mlp = ml_preds.copy(); mlp["id"] = res.id
        gm = mlp.merge(gold, on="id"); gm = gm[gm.labelable == "yes"]
        ml_cat = (gm.ml_category == gm.category).mean(); ml_pri = (gm.ml_priority == gm.priority).mean()
        ml_cat_f1 = f1_score(gm.category, gm.ml_category, average="macro")
        ml_pri_f1 = f1_score(gm.priority, gm.ml_priority, average="macro")
        m["quality"] = dict(auto_category_acc=round(cat_acc, 4), auto_priority_acc=round(pri_acc, 4),
                            auto_both_acc=round(both, 4), auto_under_prioritised=len(under),
                            urgent_recall=round(urgent_caught, 4), ml_cv_category_acc=round(ml_cat, 4),
                            ml_cv_priority_acc=round(ml_pri, 4), ml_cv_category_f1=round(ml_cat_f1, 4),
                            ml_cv_priority_f1=round(ml_pri_f1, 4))
        L += ["", "## Quality against hand-checked labels", "",
              "| Measure | Value |", "|---|---:|",
              f"| Category correct on auto-handled tickets | {cat_acc:.1%} |",
              f"| Priority correct on auto-handled tickets | {pri_acc:.1%} |",
              f"| Both correct on auto-handled tickets | {both:.1%} |",
              f"| Auto-handled tickets given a LOWER priority than they deserve | {len(under)} |",
              f"| Urgent tickets caught (labelled Urgent or sent to review) | {urgent_caught:.1%} |",
              f"| ML alone, 5-fold CV, category accuracy / macro-F1 | {ml_cat:.1%} / {ml_cat_f1:.2f} |",
              f"| ML alone, 5-fold CV, priority accuracy / macro-F1 | {ml_pri:.1%} / {ml_pri_f1:.2f} |", "",
              "_The rules were written from the same reading of the data as the labels, so their score is "
              "optimistic; the ML cross-validation numbers are out-of-sample. A real deployment should "
              "measure against labels from support agents (see design note)._"]
        _charts(res, gm, out)
        L += ["", "![handled](charts/overview.png)", "", "![confusion](charts/ml_confusion.png)"]
    else:
        _charts(res, None, out)
        L += ["", "![handled](charts/overview.png)"]

    L += ["", "## Files", "", "- `results.csv`: id, category, priority, needs_review (the required output)",
          "- `review_queue.csv`: tickets for people, most urgent first, with reasons and suggested labels",
          "- `results_detailed.csv`: every decision with confidence, reasons, notes and explanation"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return m


def _charts(res, gm, out: Path):
    if plt is None:
        return
    cdir = out / "charts"; cdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    auto = res[res.needs_review == "no"]
    ax[0].pie([len(auto), len(res) - len(auto)], labels=["Automatic", "Human review"],
              autopct="%1.0f%%", colors=["#4C72B0", "#DD8452"], startangle=90)
    ax[0].set_title("How tickets were handled")
    cc = res.suggested_category.replace("", "Unclear").value_counts()
    ax[1].barh(cc.index, cc.values, color="#55A868"); ax[1].invert_yaxis(); ax[1].set_title("Category")
    pc = res.suggested_priority.replace("", "Unclear").value_counts().reindex(PRIS + ["Unclear"]).dropna()
    ax[2].bar(pc.index, pc.values, color=["#C44E52", "#DD8452", "#CCB974", "#64B5CD", "#999"][:len(pc)])
    ax[2].set_title("Priority")
    fig.tight_layout(); fig.savefig(cdir / "overview.png", dpi=110); plt.close(fig)
    if gm is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for a, (y, p, labels, t) in zip(ax, [(gm.category, gm.ml_category, CATS, "ML category (CV)"),
                                             (gm.priority, gm.ml_priority, PRIS, "ML priority (CV)")]):
            cm = confusion_matrix(y, p, labels=labels)
            a.imshow(cm, cmap="Blues")
            a.set_xticks(range(len(labels)), labels, rotation=35); a.set_yticks(range(len(labels)), labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    a.text(j, i, cm[i, j], ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
            a.set_xlabel("predicted"); a.set_ylabel("gold"); a.set_title(t)
        fig.tight_layout(); fig.savefig(cdir / "ml_confusion.png", dpi=110); plt.close(fig)
