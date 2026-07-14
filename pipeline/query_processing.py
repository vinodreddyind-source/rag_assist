"""
Query analyzer: acronym expansion + product routing.
Mirrors GUIDEWIRE_SYNONYMS / DEPT_ROUTING in your interview doc. In this
build it's a plain dict for clarity, but your interview doc already has the
right answer for "where's this stored in production" -- a versioned
config/DB table, not a hardcoded dict. Keep that distinction straight.

ROUTING FIX (after a real production bug): the original version returned
the FIRST matching keyword, checked in dict-insertion order. "claim" is a
weak, ambiguous signal -- it shows up incidentally in billing/policy
questions ("at-fault claim") as often as in genuine ClaimCenter questions.
Checking it first meant it silently won against much stronger, unambiguous
signals like "ncd" or "premium". Fixed by SCORING every product's matched
keywords, weighted by specificity, and returning the highest score --
"claim" now contributes weight 1, while domain-specific terms like "ncd",
"premium", "fnol", "underwrite" contribute weight 2. This is still a
heuristic, not a real classifier -- see the module-level note below on why
production needs an LLM classifier instead of any keyword scheme, however
weighted.
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

# (keyword, weight) per product. Weight 1 = ambiguous/generic (can plausibly
# appear in questions about other products too). Weight 2 = specific/rare
# (strong signal for exactly this product). This is still just a heuristic --
# a real fix would be an LLM classifier (as your Guidewire doc specifies:
# GPT-4.1-mini for intent/routing), which actually understands the
# question's subject rather than counting weighted substring hits. Keep
# this distinction ready if asked "why not just add more keywords" --
# more keywords make the SAME category of mistake, just less often.
DEPT_KEYWORDS = {
    "ClaimCenter": [("claim", 1), ("adjuster", 2), ("fnol", 2)],
    "BillingCenter": [("premium", 2), ("invoice", 2), ("payment", 2), ("ncd", 2),
                       ("no claims discount", 2), ("eft", 2)],
    "PolicyCenter": [("policy", 1), ("underwrite", 2), ("renewal", 2)],
}


def expand_acronyms(query: str) -> str:
    expanded = query
    for acro, full in SYNONYMS.items():
        # word-boundary match, case-sensitive-ish (acronyms are usually upper/mixed case in text)
        pattern = r'\b' + re.escape(acro) + r'\b'
        expanded = re.sub(pattern, f"{full} ({acro})", expanded, flags=re.IGNORECASE)
    return expanded


def route_product(query: str) -> str | None:
    """Scores each product by its matched keywords' total weight and
    returns the highest scorer. Returns None (search all products) if
    nothing matches, or if the top score is tied across products --
    ties mean the query is genuinely ambiguous/cross-product, and guessing
    would be worse than searching broadly."""
    lower = query.lower()
    scores: dict[str, int] = {}
    for product, keywords in DEPT_KEYWORDS.items():
        score = sum(weight for kw, weight in keywords if kw in lower)
        if score > 0:
            scores[product] = score

    if not scores:
        return None
    max_score = max(scores.values())
    top_products = [p for p, s in scores.items() if s == max_score]
    if len(top_products) > 1:
        return None  # genuine tie -- ambiguous/cross-product, don't guess
    return top_products[0]


if __name__ == "__main__":
    tests = [
        ("how does CC handle FNOL for a BI claim?", "ClaimCenter"),
        ("what happens to NCD after an at-fault claim?", "BillingCenter"),
        ("how does ClaimCenter (CC) handle first notice of loss (FNOL) for a bodily injury (BI) claim?", "ClaimCenter"),
        ("what happens to no claims discount (NCD) after an at-fault claim?", "BillingCenter"),
    ]
    for q, expected in tests:
        result = route_product(q)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {q}\n       -> routed: {result} (expected: {expected})\n")
