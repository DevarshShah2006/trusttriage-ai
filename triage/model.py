"""Statistical second opinion: TF-IDF (word + character n-grams) + Logistic Regression.

Character n-grams make it robust to typos and give it some signal on non-English text.
It is trained on the hand-checked gold labels and evaluated with 5-fold cross-validation,
so the numbers in the report are out-of-sample, not memorised.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "model.joblib"


def _pipe() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=8.0, class_weight="balanced")
    return Pipeline([("tfidf", feats), ("clf", clf)])


class TicketModel:
    def __init__(self):
        self.cat = _pipe()
        self.pri = _pipe()

    def fit(self, texts: list[str], cats: list[str], pris: list[str]) -> "TicketModel":
        self.cat.fit(texts, cats)
        self.pri.fit(texts, pris)
        return self

    def predict(self, texts: list[str]) -> pd.DataFrame:
        pc, pp = self.cat.predict_proba(texts), self.pri.predict_proba(texts)
        return _frame(pc, self.cat.classes_, pp, self.pri.classes_)

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "TicketModel":
        return joblib.load(path)


def _frame(pc, cat_classes, pp, pri_classes) -> pd.DataFrame:
    return pd.DataFrame({
        "ml_category": np.asarray(cat_classes)[pc.argmax(1)],
        "ml_category_p": pc.max(1),
        "ml_priority": np.asarray(pri_classes)[pp.argmax(1)],
        "ml_priority_p": pp.max(1),
    })


def out_of_fold(texts: list[str], cats: list[str], pris: list[str], seed: int = 42) -> pd.DataFrame:
    """Predictions for every training ticket from a model that never saw that ticket."""
    out = {}
    for name, y in (("cat", cats), ("pri", pris)):
        min_class = pd.Series(y).value_counts().min()
        cv = StratifiedKFold(n_splits=max(2, min(5, min_class)), shuffle=True, random_state=seed)
        pipe = _pipe()
        proba = cross_val_predict(pipe, texts, y, cv=cv, method="predict_proba")
        out[name] = (proba, sorted(set(y)))
    return _frame(out["cat"][0], out["cat"][1], out["pri"][0], out["pri"][1])
