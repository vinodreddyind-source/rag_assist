"""
Query analyzer: acronym expansion + product routing.
Mirrors GUIDEWIRE_SYNONYMS / DEPT_ROUTING in your interview doc. In this
build it's a plain dict for clarity, but your interview doc already has the
right answer for "where's this stored in production" -- a versioned
config/DB table, not a hardcoded dict. Keep that distinction straight.
"""

import re

SYNONYMS = {
    "BI": "bodily injury",
    "PD": "property damage",
    "UIM": "underinsured motorist",
    "FNOL": "first notice of loss",
    "SIU": "special investigations unit",
    "CC": "ClaimCenter",
    "BC": "BillingCenter",
    "PC": "PolicyCenter",
    "LOB": "line of business",
    "EFT": "electronic funds transfer",
    "NCD": "no claims discount",
}

DEPT_ROUTING = {
    "claim": "ClaimCenter", "adjuster": "ClaimCenter", "fnol": "ClaimCenter",
    "premium": "BillingCenter", "invoice": "BillingCenter", "payment": "BillingCenter",
    "policy": "PolicyCenter", "underwrite": "PolicyCenter", "renewal": "PolicyCenter",
}


def expand_acronyms(query: str) -> str:
    expanded = query
    for acro, full in SYNONYMS.items():
        # word-boundary match, case-sensitive-ish (acronyms are usually upper/mixed case in text)
        pattern = r'\b' + re.escape(acro) + r'\b'
        expanded = re.sub(pattern, f"{full} ({acro})", expanded, flags=re.IGNORECASE)
    return expanded


def route_product(query: str) -> str | None:
    lower = query.lower()
    for keyword, product in DEPT_ROUTING.items():
        if keyword in lower:
            return product
    return None  # cross-product / ambiguous -- search all indexes


if __name__ == "__main__":
    tests = [
        "how does CC handle FNOL for a BI claim?",
        "what happens to NCD after an at-fault claim?",
    ]
    for q in tests:
        print(f"ORIGINAL:  {q}")
        print(f"EXPANDED:  {expand_acronyms(q)}")
        print(f"ROUTED TO: {route_product(q)}\n")
