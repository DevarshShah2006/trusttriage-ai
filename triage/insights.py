"""Model experiments, calibration, incident clusters and related tickets.

Everything here is computed from the real 300 tickets and the hand-checked labels;
nothing is simulated. Output: outputs/insights.json, read by the dashboard.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .model import _pipe
from .preprocess import prepare
from .rules import decide, match

PRIS = ["Low", "Medium", "High", "Urgent"]
SEVERE = {"Urgent", "High"}


def _feats():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


CANDIDATES = {
    "Logistic Regression (production)": _pipe,
    "Linear SVM": lambda: Pipeline([("f", _feats()), ("c", LinearSVC(C=1.0, class_weight="balanced"))]),
    "Complement Naive Bayes": lambda: Pipeline([("f", _feats()), ("c", ComplementNB(alpha=0.3))]),
    "Logistic Regression (word only)": lambda: Pipeline([
        ("f", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("c", LogisticRegression(max_iter=2000, C=8.0, class_weight="balanced"))]),
}


def _calibration(proba: np.ndarray, classes, y) -> dict:
    """Multi-class Brier score and expected calibration error (10 bins)."""
    y = np.asarray(y)
    onehot = (np.asarray(classes)[None, :] == y[:, None]).astype(float)
    brier = float(((proba - onehot) ** 2).sum(1).mean())
    conf = proba.max(1)
    correct = (np.asarray(classes)[proba.argmax(1)] == y).astype(float)
    bins = np.linspace(0, 1, 11)
    ece, curve = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            curve.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                          "confidence": round(float(conf[m].mean()), 3),
                          "accuracy": round(float(correct[m].mean()), 3)})
    return {"brier": round(brier, 4), "ece": round(float(ece), 4), "curve": curve}


def experiments(texts, cats, pris, bodies) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, calib = [], {}
    for name, make in CANDIDATES.items():
        t0 = time.perf_counter()
        pc = cross_val_predict(make(), texts, cats, cv=cv)
        pp = cross_val_predict(make(), texts, pris, cv=cv)
        secs = time.perf_counter() - t0
        urgent = np.asarray(pris) == "Urgent"
        under = np.array([PRIS.index(p) < PRIS.index(g) for p, g in zip(pp, pris)])
        rows.append({
            "model": name,
            "category_acc": round(accuracy_score(cats, pc), 4),
            "category_f1": round(f1_score(cats, pc, average="macro"), 4),
            "priority_acc": round(accuracy_score(pris, pp), 4),
            "priority_f1": round(f1_score(pris, pp, average="macro"), 4),
            "urgent_recall": round(float((pp[urgent] == "Urgent").mean()), 4),
            "under_prioritised": int(under.sum()),
            "cv_seconds": round(secs, 2),
        })
        if name.startswith("Logistic Regression (production)"):
            for task, y in (("category", cats), ("priority", pris)):
                proba = cross_val_predict(make(), texts, y, cv=cv, method="predict_proba")
                calib[task] = _calibration(proba, sorted(set(y)), y)
    # Rules alone (no training; optimistic because labels share the same judgement)
    rp = [decide(match(prepare("", b).segments)) for b in bodies]
    rc = [d["category"] if d else "Other" for d in rp]
    rpr = [d["priority"] if d else "Low" for d in rp]
    rows.append({"model": "Rules only (no ML)", "category_acc": round(accuracy_score(cats, rc), 4),
                 "category_f1": round(f1_score(cats, rc, average="macro"), 4),
                 "priority_acc": round(accuracy_score(pris, rpr), 4),
                 "priority_f1": round(f1_score(pris, rpr, average="macro"), 4),
                 "urgent_recall": round(float(np.mean([p == "Urgent" for p, g in zip(rpr, pris) if g == "Urgent"])), 4),
                 "under_prioritised": int(sum(PRIS.index(p) < PRIS.index(g) for p, g in zip(rpr, pris))),
                 "cv_seconds": 0.0, "note": "optimistic: written from the same reading as the labels"})
    return {"leaderboard": rows, "calibration": calib}


def per_class(gold: pd.DataFrame, res: pd.DataFrame, ml: pd.DataFrame) -> dict:
    g = gold[gold.labelable == "yes"].set_index("id")
    m = ml.set_index("id").loc[g.index]
    out = {}
    for task, col in (("category", "ml_category"), ("priority", "ml_priority")):
        rep = classification_report(g[task], m[col], output_dict=True, zero_division=0)
        out[task] = [{"label": k, "precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                      "f1": round(v["f1-score"], 3), "support": int(v["support"])}
                     for k, v in rep.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")]
    r = res.set_index("id")
    rr = r.loc[g.index]
    out["agreement"] = {
        "rules_ml_category_agree": round(float((rr.rule_category == rr.ml_category).mean()), 4),
        "rules_ml_priority_agree": round(float((rr.rule_priority == rr.ml_priority).mean()), 4),
        "review_rate": round(float((r.needs_review == "yes").mean()), 4),
        "automation_coverage": round(float((r.needs_review == "no").mean()), 4),
    }
    return out


def incidents(res: pd.DataFrame) -> list[dict]:
    """Clusters of the same severe problem reported by several customers."""
    df = res.copy()
    df["ts"] = pd.to_datetime(df.received_at, errors="coerce")
    out = []
    for intent, g in df[df.suggested_priority.isin(SEVERE) & (df.rule_intent != "")].groupby("rule_intent"):
        if len(g) < 3:
            continue
        g = g.sort_values("ts")
        out.append({
            "intent": intent, "category": g.suggested_category.mode()[0],
            "priority": "Urgent" if (g.suggested_priority == "Urgent").any() else "High",
            "tickets": len(g), "customers": int(g.customer.nunique()),
            "first_seen": str(g.ts.min())[:16], "last_seen": str(g.ts.max())[:16],
            "span_hours": round((g.ts.max() - g.ts.min()).total_seconds() / 3600, 1),
            "explanation": g.explanation.iloc[0], "ticket_ids": g.id.tolist(),
            "status": "Active" if g.suggested_priority.eq("Urgent").any() else "Monitoring",
        })
    return sorted(out, key=lambda x: (x["priority"] != "Urgent", -x["tickets"]))


def related(ids, bodies, k: int = 3, min_sim: float = 0.45) -> dict:
    """Most similar other tickets (TF-IDF cosine) - duplicate / same-issue detection."""
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(bodies)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    out = {}
    for i, tid in enumerate(ids):
        idx = np.argsort(-S[i])[:k]
        out[tid] = [{"id": ids[j], "similarity": round(float(S[i, j]), 3)} for j in idx if S[i, j] >= min_sim]
    return out


def hourly(res: pd.DataFrame) -> list[dict]:
    ts = pd.to_datetime(res.received_at, errors="coerce").dt.floor("h")
    df = pd.DataFrame({"h": ts, "urgent": res.suggested_priority.eq("Urgent"),
                       "review": res.needs_review.eq("yes")})
    g = df.groupby("h").agg(total=("urgent", "size"), urgent=("urgent", "sum"), review=("review", "sum"))
    return [{"hour": str(h)[:13] + ":00", "total": int(r.total), "urgent": int(r.urgent), "review": int(r.review)}
            for h, r in g.iterrows()]


def tone(notes: str, intent: str) -> str:
    if "angry tone" in notes:
        return "Frustrated"
    if "downplays" in notes:
        return "Downplaying"
    if intent == "praise":
        return "Positive"
    return "Neutral"


def build(tickets: pd.DataFrame, res: pd.DataFrame, gold: pd.DataFrame, ml: pd.DataFrame) -> dict:
    lab = tickets.merge(gold, on="id")
    lab = lab[lab.labelable == "yes"]
    texts = (lab.subject + "\n" + lab.body).tolist()
    cleaned = [prepare(s, b).body for s, b in zip(tickets.subject, tickets.body)]
    return {
        "experiments": experiments(texts, lab.category.tolist(), lab.priority.tolist(), lab.body.tolist()),
        "per_class": per_class(gold, res, ml),
        "incidents": incidents(res),
        "related": related(tickets.id.tolist(), [c or "empty" for c in cleaned]),
        "hourly": hourly(res),
        "tone": {i: tone(n, it) for i, n, it in zip(res.id, res.notes, res.rule_intent)},
    }
