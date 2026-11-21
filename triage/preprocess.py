"""Cleaning and safety checks that run before any model sees a ticket.

Everything here is deterministic and cheap, so it runs even when every model is down.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Machine log lines pasted into tickets (e.g. "2026-11-12T06:00:12Z [ERROR] ota.update: ...").
LOG_LINE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?\s+\[(DEBUG|INFO|WARN|ERROR)\].*$", re.M)

# Text that tries to instruct the AI instead of describing a problem (prompt injection).
INJECTION = re.compile(
    r"(ignore (all )?(previous|prior) instructions|system note to the ai|\[assistant:|"
    r"the rules have changed|output priority\s*=|classify this ticket as|do not send it for review)",
    re.I,
)
# Remove the injected instruction block so it cannot steer the classifier.
INJECTION_BLOCK = re.compile(
    r"(\[assistant:[^\]]*\]|-{3,}\s*\n?\s*IMPORTANT SYSTEM NOTE.*$|IMPORTANT SYSTEM NOTE.*$)", re.I | re.S
)

MULTI_SPLIT = re.compile(r"\bAlso,? a separate thing:\s*", re.I)

# Low-content tickets nobody (human or model) can classify without asking the customer.
NO_CONTENT = {"", "help", "sent from my phone", "test", "asdf"}


@dataclass
class Prepared:
    text: str                      # cleaned text used for classification (subject + body)
    body: str                      # cleaned body only
    segments: list[str]            # body split into separate requests
    language: str                  # 'en' or a best-guess other language code
    flags: list[str] = field(default_factory=list)
    log_lines: int = 0


def detect_language(text: str) -> str:
    t = text.lower()
    if re.search(r"[ऀ-ॿ]", t):
        return "hi"
    if re.search(r"[぀-ヿ一-鿿]", t):
        return "ja"
    markers = {
        "de": r"\b(ich|bitte|seit|der|die|und|könnten|keines)\b",
        "fr": r"\b(bonjour|je|j'ai|comment|puis|merci|commande)\b",
        "es": r"\b(hola|puedo|tarjeta|gracias|cobrado|ningún)\b",
        "pt": r"\b(olá|gostaria|seria|obrigad[oa]|português|minha)\b",
        "it": r"\b(buongiorno|ordine|spedito|fermo|il mio)\b",
    }
    scores = {lang: len(re.findall(p, t)) for lang, p in markers.items()}
    lang, score = max(scores.items(), key=lambda kv: kv[1])
    english = len(re.findall(r"\b(the|and|my|is|to|i|you|it|of|for)\b", t))
    return lang if score >= 2 and score > english else "en"


def is_gibberish(body: str) -> bool:
    words = re.findall(r"[^\W\d_]+", body.lower())
    if not words:
        return True
    real = [w for w in words if w not in {"asdf", "test", "qwerty", "lorem", "ipsum"}]
    return len(real) == 0


def prepare(subject: str, body: str) -> Prepared:
    subject = unicodedata.normalize("NFKC", str(subject or "")).strip()
    body = unicodedata.normalize("NFKC", str(body or "")).strip()
    flags: list[str] = []

    log_lines = len(LOG_LINE.findall(body))
    if log_lines:
        body = LOG_LINE.sub("", body)
        flags.append("log_attached")

    if INJECTION.search(body) or INJECTION.search(subject):
        flags.append("prompt_injection")
        body = INJECTION_BLOCK.sub("", body)

    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    segments = [s.strip() for s in MULTI_SPLIT.split(body) if s.strip()]
    if len(segments) > 1:
        flags.append("multiple_requests")

    if body.lower().strip(" .!?") in NO_CONTENT or is_gibberish(body) or len(body) < 12:
        flags.append("no_usable_content")

    lang = detect_language(subject + " " + body)
    if lang != "en":
        flags.append(f"non_english:{lang}")

    letters = [c for c in subject + body if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5 or "!!!" in subject + body:
        flags.append("angry_tone")

    return Prepared(text=f"{subject}\n{body}", body=body, segments=segments or [body],
                    language=lang, flags=flags, log_lines=log_lines)
