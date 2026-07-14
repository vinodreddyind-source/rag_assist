"""
Generates a synthetic corpus of Guidewire-style internal documentation
(ClaimCenter / BillingCenter / PolicyCenter) with realistic header hierarchy,
insurance jargon/acronyms, and cross-product references.

No external APIs used -- pure templated generation, deterministic with a seed
so re-runs are reproducible. This is what stands in for the real 8,000-page
Guidewire corpus your interview doc describes.
"""

import json
import random
import os

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs")

# ---------------------------------------------------------------------------
# Domain content blocks per product. Each is a dict of section -> paragraphs.
# Acronyms are deliberately embedded so abbreviation-expansion / hybrid search
# has something real to demonstrate (mirrors GUIDEWIRE_SYNONYMS in your doc).
# ---------------------------------------------------------------------------

CLAIMCENTER_SECTIONS = {
    "First Notice of Loss (FNOL) Intake": [
        "When an FNOL is submitted through ClaimContactScreen, the adjuster must "
        "confirm the loss date, policy number, and line of business (LOB) before "
        "the claim can be created in ClaimCenter. For auto claims involving bodily "
        "injury (BI), the system requires an additional injury-details step before "
        "the claim can move to the Coverage Review stage.",
        "FNOL calls routed through the SIU (special investigations unit) flag are "
        "held for manual review if the reported loss amount exceeds $50,000 or if "
        "the same policy number has more than two prior FNOL entries in 12 months.",
    ],
    "Deductible Calculation": [
        "The deductible applied to a claim is read from the policy's effective "
        "coverage terms at the loss date, not the terms in effect when the claim "
        "is processed. This distinction matters most when a policy renewal occurs "
        "between the loss date and the claim processing date -- the pre-renewal "
        "deductible still applies to a loss that occurred before renewal.",
        "For property damage (PD) claims with multiple covered items, ClaimCenter "
        "applies a single per-occurrence deductible rather than a per-item "
        "deductible, unless the policy's PolicyCenter terms explicitly define "
        "split deductibles for that line of business.",
    ],
    "Claim Reserve Adjustment": [
        "Reserve amounts in ClaimCenter should be revised whenever new exposure "
        "information is entered -- for example, a getExposureById() call returning "
        "an updated repair estimate should trigger a reserve review, not just a "
        "note in the claim file.",
    ],
    "Renewal Impact on Open Claims": [
        "When a policy renews in PolicyCenter while a claim on that policy is "
        "still open in ClaimCenter, the claim retains the coverage terms and "
        "deductible from the policy term active at the loss date. Adjusters "
        "sometimes mistakenly apply the newly renewed term's deductible -- this "
        "is the single most common cross-product error reported by the claims team.",
    ],
}

BILLINGCENTER_SECTIONS = {
    "Premium Invoice Generation": [
        "BillingCenter generates the premium invoice based on the payment plan "
        "selected in PolicyCenter at bind time. Electronic funds transfer (EFT) "
        "customers are billed on the policy's designated draft date, while "
        "invoice-pay customers receive a paper or email invoice 21 days before "
        "the due date.",
    ],
    "No Claims Discount (NCD) Application": [
        "The no claims discount (NCD), sometimes called a no claims bonus, is "
        "recalculated at each renewal based on the claim-free period in "
        "ClaimCenter. A single at-fault claim resets the NCD tier; a not-at-fault "
        "claim does not affect NCD tier progression under standard policy terms.",
    ],
    "Payment Failure Handling": [
        "If an EFT payment fails, BillingCenter retries the draft once after 3 "
        "business days before flagging the account for manual collections "
        "review. Invoice-pay accounts that miss a due date move directly to a "
        "10-day grace period before a cancellation notice is generated.",
    ],
    "Mid-Term Policy Changes and Billing": [
        "A mid-term endorsement processed in PolicyCenter -- for example adding "
        "a vehicle or increasing coverage limits -- triggers a pro-rated premium "
        "adjustment in BillingCenter for the remaining term, not a full re-bill "
        "of the annual premium.",
    ],
}

POLICYCENTER_SECTIONS = {
    "Policy Renewal Processing": [
        "PolicyCenter initiates the renewal workflow 60 days before the current "
        "term's expiration date. Underwriting review is only triggered "
        "automatically if the account has an open claim with reserves above "
        "$25,000 or if risk-scoring flags a change in the underlying exposure.",
    ],
    "Underwriting Referral Rules": [
        "A submission is referred to underwriting review when the line of "
        "business (LOB) risk score exceeds the automated-approval threshold, or "
        "when the applicant has a claim history that includes an SIU-flagged "
        "claim within the past 3 years.",
    ],
    "Mid-Term Endorsements": [
        "Endorsements that change coverage limits or add a driver require "
        "re-running the automated underwriting rules before the endorsement can "
        "bind, even if the original policy was already approved.",
    ],
    "Cross-Product: Renewal While Claim Is Open": [
        "When PolicyCenter processes a renewal for a policy with an open claim in "
        "ClaimCenter, the renewal itself does not alter the terms applied to the "
        "existing claim. The new term's coverage and deductible apply only to "
        "losses occurring on or after the new effective date. This is the same "
        "rule referenced in ClaimCenter's 'Renewal Impact on Open Claims' section, "
        "and is the most frequent cross-product question raised by claims "
        "adjusters during renewal season.",
    ],
}

PRODUCTS = {
    "ClaimCenter": CLAIMCENTER_SECTIONS,
    "BillingCenter": BILLINGCENTER_SECTIONS,
    "PolicyCenter": POLICYCENTER_SECTIONS,
}

DOC_TYPES = ["user_guide", "api_reference", "release_notes"]
VERSIONS = ["10.x", "11.x"]


def make_doc(product: str, section_title: str, paragraphs: list[str], doc_idx: int) -> tuple[str, dict]:
    doc_type = random.choice(DOC_TYPES)
    version = random.choice(VERSIONS)
    dept = {"ClaimCenter": "claims", "BillingCenter": "billing", "PolicyCenter": "underwriting"}[product]

    # A few sub-headings per section so the parent-child chunker has real
    # hierarchy to split on, not just one flat paragraph block.
    md = [f"# {product} User Guide", f"## {section_title}"]
    for i, para in enumerate(paragraphs):
        md.append(f"### Detail {i+1}")
        md.append(para)
        md.append("")

    content = "\n".join(md)
    metadata = {
        "doc_id": f"{product.lower()}_{doc_idx:04d}",
        "product": product,
        "section": section_title,
        "doc_type": doc_type,
        "version": version,
        "department": dept,
    }
    return content, metadata


def generate_corpus():
    all_metadata = []
    doc_idx = 0
    for product, sections in PRODUCTS.items():
        product_dir = os.path.join(OUT_DIR, product)
        os.makedirs(product_dir, exist_ok=True)
        for section_title, paragraphs in sections.items():
            doc_idx += 1
            content, meta = make_doc(product, section_title, paragraphs, doc_idx)
            fname = f"{meta['doc_id']}.md"
            with open(os.path.join(product_dir, fname), "w") as f:
                f.write(content)
            meta["file"] = os.path.join(product, fname)
            all_metadata.append(meta)

    with open(os.path.join(OUT_DIR, "..", "corpus_metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Generated {len(all_metadata)} synthetic documents across {len(PRODUCTS)} products.")
    return all_metadata


# ---------------------------------------------------------------------------
# Golden Q&A set for RAGAS -- includes plain queries, abbreviation-heavy
# queries, and cross-product queries so the eval set actually exercises the
# behaviours your interview doc claims (91% abbreviation hit rate, 8%
# cross-product queries, etc.)
# ---------------------------------------------------------------------------

GOLDEN_QA = [
    {
        "question": "How is the deductible determined when a claim spans a policy renewal?",
        "expected_source": "policycenter_0004.md",
        "expected_answer": "The pre-renewal deductible applies -- the claim retains the coverage terms active at the loss date, not the newly renewed term.",
        "type": "cross_product",
    },
    {
        "question": "How does CC handle FNOL for a BI claim?",
        "expected_source": "claimcenter_0001.md",
        "expected_answer": "ClaimCenter requires confirming loss date, policy number, and line of business, plus an additional injury-details step for bodily injury claims before Coverage Review.",
        "type": "abbreviation",
    },
    {
        "question": "What happens to the NCD if a customer has an at-fault claim?",
        "expected_source": "billingcenter_0002.md",
        "expected_answer": "A single at-fault claim resets the no claims discount tier; a not-at-fault claim does not affect NCD tier progression.",
        "type": "abbreviation",
    },
    {
        "question": "When does PolicyCenter automatically trigger underwriting review at renewal?",
        "expected_source": "policycenter_0001.md",
        "expected_answer": "Underwriting review triggers automatically if the account has an open claim with reserves above $25,000, or if risk-scoring flags a change in exposure.",
        "type": "plain",
    },
    {
        "question": "What happens if an EFT payment fails?",
        "expected_source": "billingcenter_0003.md",
        "expected_answer": "BillingCenter retries the draft once after 3 business days before flagging the account for manual collections review.",
        "type": "abbreviation",
    },
    {
        "question": "Is the per-occurrence or per-item deductible used for a PD claim with multiple covered items?",
        "expected_source": "claimcenter_0002.md",
        "expected_answer": "A single per-occurrence deductible applies, unless the policy explicitly defines split per-item deductibles for that line of business.",
        "type": "abbreviation",
    },
    {
        "question": "Does a mid-term endorsement trigger a full annual re-bill?",
        "expected_source": "billingcenter_0004.md",
        "expected_answer": "No -- it triggers a pro-rated premium adjustment for the remaining term, not a full re-bill of the annual premium.",
        "type": "plain",
    },
    {
        "question": "What claim history flag forces an underwriting referral?",
        "expected_source": "policycenter_0002.md",
        "expected_answer": "A claim history that includes an SIU-flagged claim within the past 3 years forces a referral to underwriting review.",
        "type": "abbreviation",
    },
]


def generate_golden_set():
    out_path = os.path.join(OUT_DIR, "..", "golden_qa.json")
    with open(out_path, "w") as f:
        json.dump(GOLDEN_QA, f, indent=2)
    print(f"Wrote {len(GOLDEN_QA)} golden QA pairs to {out_path}")


if __name__ == "__main__":
    generate_corpus()
    generate_golden_set()
