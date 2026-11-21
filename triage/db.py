"""SQLite store for human review decisions: an audit trail and future training data."""
from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import os

# Vercel's filesystem is read-only except /tmp (data there lasts only while the instance is warm).
DB_PATH = (Path("/tmp/reviews.db") if os.getenv("VERCEL")
           else Path(__file__).resolve().parent.parent / "outputs" / "reviews.db")



@contextmanager
def _db(path: Path = DB_PATH):
    """Open, commit and always close (so the file is never left locked)."""
    with closing(_conn(path)) as c, c:
        yield c


def _conn(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id TEXT, category TEXT, priority TEXT, action TEXT, reviewer TEXT,
        model_category TEXT, model_priority TEXT, comment TEXT, reviewed_at TEXT)""")
    return c


def save_review(tid, category, priority, action, reviewer, model_cat, model_pri, comment="", path=DB_PATH):
    with _db(path) as c:
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?)",
                  (tid, category, priority, action, reviewer, model_cat, model_pri, comment,
                   datetime.now(timezone.utc).isoformat(timespec="seconds")))


def load_reviews(path=DB_PATH) -> pd.DataFrame:
    with _db(path) as c:
        return pd.read_sql("SELECT * FROM reviews ORDER BY reviewed_at", c)


def latest_reviews(path=DB_PATH) -> pd.DataFrame:
    df = load_reviews(path)
    return df.groupby("id").tail(1) if len(df) else df
