"""Intent knowledge base: explainable, multilingual rules written by reading the tickets.

Each intent carries a category, a priority, and a 'severity reason' so every decision
can be explained to the support team. Rules look at the BODY, not the subject: in this
data the subject line is often misleading (e.g. "Minor issue" for a 40-device outage).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PRIORITY_RANK = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}
RANK_PRIORITY = {v: k for k, v in PRIORITY_RANK.items()}


@dataclass(frozen=True)
class Intent:
    name: str
    category: str
    priority: str
    pattern: re.Pattern
    why: str


def _p(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts), re.I)


# Ordered roughly from most to least severe. A ticket can match several intents.
INTENTS: list[Intent] = [
    # ---------------- Urgent: money leaving, account taken over, business down ----------------
    Intent("unauthorised_charges", "Billing", "Urgent",
           _p(r"charges?.{0,90}(did not|didn't) make", r"(do not|don't) remember making",
              r"someone is using my (saved )?card", r"orders? being placed.{0,30}not mine",
              r"पैसे कटे.*नहीं किए", r"खाता ब्लॉक"),
           "possible fraud, money is leaving the customer's card right now"),
    Intent("account_takeover", "Account", "Urgent",
           _p(r"email.{0,40}(was|been) changed.{0,80}(not me|did not do|didn't do|was not me)",
              r"devices being switched on and off", r"account (was|has been) hacked"),
           "account takeover, an attacker may control the customer's devices"),
    Intent("business_outage", "Technical", "Urgent",
           _p(r"business account shows as unreachable", r"affecting all our sites",
              r"whole office depends", r"all \d+ of our .{0,30}(offline|down)",
              r"none of the \d+ units", r"staff cannot unlock"),
           "outage for a whole business site, many users blocked"),
    # ---------------- High: customer fully blocked, double payment, legal risk ----------------
    Intent("total_outage_consumer", "Technical", "High",
           _p(r"closes the moment i open", r"cannot control any of my devices",
              r"(hub|app) will not start", r"none of my devices can be controlled",
              r"nothing in my home can be controlled", r"stürzt .{0,30}ab", r"keines meiner geräte"),
           "customer cannot use the product at all"),
    Intent("cannot_login", "Account", "High",
           _p(r"cannot log in", r"can no longer log in", r"no access to anything",
              r"temporarily locked", r"havent been able to log in", r"haven't been able to log in",
              r"no puedo iniciar sesi[oó]n", r"password is not accepted"),
           "customer is locked out of their account"),
    Intent("lost_2fa", "Account", "High",
           _p(r"authenticator app", r"two-factor", r"2fa", r"login codes"),
           "customer is about to be locked out (lost two-factor device)"),
    Intent("double_charge", "Billing", "High",
           _p(r"charged twice", r"two identical payments", r"refund the duplicate",
              r"cobrado dos veces", r"dos cargos iguales"),
           "customer was charged twice"),
    Intent("legal_threat", "Other", "High",
           _p(r"solicitor", r"ombudsman", r"breach of our service agreement", r"our legal team",
              r"third complaint"),
           "legal or regulatory escalation, deadline-sensitive"),
    # ---------------- Medium: something is broken or money is wrong, but not blocking ----------------
    Intent("renewal_dispute", "Billing", "Medium",
           _p(r"renewal charge", r"renewed automatically", r"auto-renew", r"did not mean to renew"),
           "unexpected renewal charge"),
    Intent("discount_missing", "Billing", "Medium",
           _p(r"code \w+ at checkout", r"apply the discount", r"sale price was not applied",
              r"charged the full"),
           "discount not applied"),
    Intent("order_not_arrived", "Shipping", "Medium",
           _p(r"still has not arrived", r"says delivered, but nothing", r"nothing was left at my address",
              r"n'ai toujours rien reçu", r"not arrived"),
           "order missing"),
    Intent("tracking_stuck", "Shipping", "Medium",
           _p(r"tracking (page|number)", r"label created", r"etichetta creata", r"tracking del mio ordine"),
           "tracking not updating"),
    Intent("wrong_or_damaged_item", "Shipping", "Medium",
           _p(r"box contained a different product", r"eu plug version", r"cracked casing",
              r"wet and torn", r"does not turn on", r"will not power on", r"दूसरा प्रोडक्ट", r"सही सामान"),
           "wrong or damaged item delivered"),
    Intent("change_delivery_address", "Shipping", "Medium",
           _p(r"old address", r"change it to my new address", r"deliver.{0,40}to my office",
              r"redirect.{0,30}order"),
           "address change needed before dispatch (time-sensitive)"),
    Intent("sync_issue", "Technical", "Medium",
           _p(r"do not show up on my", r"different settings in the app", r"not syncing"),
           "devices out of sync"),
    Intent("feature_broken", "Technical", "Medium",
           _p(r"export failed", r"error 5012", r"export my device history",
              r"schedule feature stopped", r"saved schedules no longer run",
              r"drops off the network", r"live view fails"),
           "a feature is failing with errors"),
    Intent("password_reset_problem", "Account", "Medium",
           _p(r"reset email took", r"link had expired", r"reset link", r"email settings"),
           "password reset email delayed"),
    Intent("data_privacy_request", "Account", "Medium",
           _p(r"data protection law", r"personal data you hold", r"what data your devices collect"),
           "data-protection request with a legal response deadline"),
    # ---------------- Low: questions, admin, feedback ----------------
    Intent("slow_but_working", "Technical", "Low",
           _p(r"ten seconds to respond", r"still works, just slowly", r"take a long time to reach", r"takes much longer than it used to",
              r"do arrive eventually", r"反応", r"10秒"),
           "performance is degraded but the product still works"),
    Intent("invoice_receipt", "Billing", "Low",
           _p(r"copy of the invoice", r"pdf receipt", r"download last month's receipt", r"rechnung"),
           "document request"),
    Intent("payment_method", "Billing", "Low",
           _p(r"bank transfer", r"update my card", r"new card", r"change the card used"),
           "payment method question"),
    Intent("shipping_info", "Shipping", "Low",
           _p(r"express shipping", r"ship to \w+", r"deliver on saturdays", r"standard delivery usually take"),
           "shipping information question"),
    Intent("close_account", "Account", "Low",
           _p(r"(delete|close) my account"),
           "account closure request"),
    Intent("change_email", "Account", "Low",
           _p(r"(would like to )?change the email address on my account",
              r"alterar o endere[cç]o de e-mail", r"update the name on my account", r"surname has changed"),
           "account details change"),
    Intent("how_to", "Technical", "Low",
           _p(r"is there a way to", r"rename a room", r"second user", r"deuxième utilisateur",
              r"how do i (set|add)"),
           "how-to question"),
    Intent("feature_request", "Other", "Low",
           _p(r"would be (great|nice) (if|to have)", r"dark mode", r"widgets", r"have you considered",
              r"seria ótimo"),
           "product feedback"),
    Intent("praise", "Other", "Low",
           _p(r"no problem to report", r"has been excellent", r"pass on my thanks", r"really helpful"),
           "positive feedback"),
    Intent("careers", "Other", "Low",
           _p(r"are you hiring", r"careers page", r"job", r"internship", r"send my cv"),
           "job enquiry"),
    Intent("partnership", "Other", "Low",
           _p(r"becoming a reseller", r"installation business", r"partnership", r"someone from sales", r"property developer"),
           "business enquiry"),
]


@dataclass
class RuleHit:
    intent: Intent
    segment: int


def match(segments: list[str]) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for i, seg in enumerate(segments):
        for intent in INTENTS:
            if intent.pattern.search(seg):
                hits.append(RuleHit(intent, i))
    return hits


def decide(hits: list[RuleHit]) -> dict | None:
    """Pick the primary intent: the most severe one, ties broken by earliest segment."""
    if not hits:
        return None
    hits = sorted(hits, key=lambda h: (-PRIORITY_RANK[h.intent.priority], h.segment))
    top = hits[0].intent
    categories = {h.intent.category for h in hits}
    return {
        "category": top.category,
        "priority": top.priority,
        "intent": top.name,
        "why": top.why,
        "all_intents": sorted({h.intent.name for h in hits}),
        "categories": sorted(categories),
    }
