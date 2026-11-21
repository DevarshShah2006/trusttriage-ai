"""TrustTriage AI: REST API + web dashboard.

    python run_triage.py          # produce outputs/ first
    uvicorn api:app --port 8000   # then open http://localhost:8000
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from triage.db import latest_reviews, load_reviews, save_review
from triage.model import MODEL_PATH, TicketModel
from triage.pipeline import triage_one

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
app = FastAPI(title="TrustTriage AI", version="1.0")
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


def _load_model():
    try:
        return TicketModel.load()
    except Exception:
        # Saved model missing or built with another sklearn version: retrain from gold labels (~1 s).
        try:
            t = pd.read_csv(ROOT / "data" / "tickets.csv", dtype=str).fillna("")
            g = t.merge(pd.read_csv(ROOT / "data" / "gold_labels.csv"), on="id")
            g = g[g.labelable == "yes"]
            texts = (g.subject + "\n" + g.body).tolist()
            return TicketModel().fit(texts, g.category.tolist(), g.priority.tolist())
        except Exception:
            return None  # rules-only mode; high-impact tickets go to review


_model = _load_model()


def _results() -> pd.DataFrame:
    p = OUT / "results_detailed.csv"
    if not p.exists():
        raise HTTPException(404, "Run `python run_triage.py` first.")
    res = pd.read_csv(p, dtype=str).fillna("")
    body = pd.read_csv(ROOT / "data" / "tickets.csv", dtype=str).fillna("")[["id", "body"]]
    return res.merge(body, on="id", how="left")


class Ticket(BaseModel):
    id: str = "LIVE"
    subject: str = ""
    body: str = ""


class Review(BaseModel):
    id: str
    category: str
    priority: str
    reviewer: str = "agent"
    comment: str = ""


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "ml_model_loaded": _model is not None}


@app.get("/api/summary")
def summary():
    return json.loads((OUT / "metrics.json").read_text())


@app.get("/api/tickets")
def tickets():
    res = _results()
    rev = latest_reviews()
    res["reviewed"] = res.id.isin(set(rev.id) if len(rev) else set())
    return res.to_dict(orient="records")


@app.post("/triage")
@app.post("/api/triage")
def triage(t: Ticket):
    ml = None
    if _model is not None:
        try:
            ml = _model.predict([f"{t.subject}\n{t.body}"]).iloc[0].to_dict()
        except Exception:
            ml = None  # model failure -> rules only; high-impact tickets go to review
    d = triage_one(t.id, t.subject, t.body, ml).__dict__
    if ml:
        d["ml"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ml.items()}
    return d


@app.post("/api/review")
def review(r: Review):
    res = _results().set_index("id")
    if r.id not in res.index:
        raise HTTPException(404, "unknown ticket")
    s = res.loc[r.id]
    action = "approved" if (r.category, r.priority) == (s.suggested_category, s.suggested_priority) else "corrected"
    save_review(r.id, r.category, r.priority, action, r.reviewer, s.suggested_category,
                s.suggested_priority, r.comment)
    return {"ok": True, "action": action}


@app.get("/api/reviews")
def reviews():
    return load_reviews().to_dict(orient="records")


@app.get("/api/export")
def export():
    """results.csv with human decisions applied on top of the model's."""
    final = _results()[["id", "category", "priority", "needs_review"]].set_index("id")
    lr = latest_reviews()
    if len(lr):
        lr = lr.set_index("id")
        final.loc[lr.index, ["category", "priority"]] = lr[["category", "priority"]].values
        final.loc[lr.index, "needs_review"] = "no"
    buf = io.StringIO(); final.reset_index().to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=results_reviewed.csv"})
