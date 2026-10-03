"""Classify a raw model reply into a recommendation outcome.

Outcomes (the "status" field):
  ok           exactly one recommendation, S or U
  refusal      no recommendation and the model declined (refusal language, or
               the API reported stop_reason == "refusal"). A genuine result:
               kept, never retried, never counted as U.
  ambiguous    the reply contains conflicting recommendation lines (e.g. "U",
               then "in reality I'd recommend S"). Kept, never retried.
  parse_error  no recognizable recommendation and no refusal: a pure format
               failure, so it is safe to retry.

The S/U letter must be uppercase; only the label and filler words are
case-insensitive, so ordinary words ("since", "unless") can never be read as
a recommendation.
"""

import re

PARSER_VERSION = 2

# One recommendation line. Tolerates markdown around the label and the letter:
#   RECOMMENDATION: U | **RECOMMENDATION**: U | **Recommendation:** **U**
#   ## RECOMMENDATION: Product U | RECOMMENDATION: <U> | RECOMMENDATION: [S]
REC_LINE_RE = re.compile(
    r"^[ \t>#*_`-]*(?i:recommendation)[ \t*_`]*:[ \t*_`]*"
    r"[<\[(\"'*_`]*[ \t]*(?:(?i:product|option)[ \t]+)?[<\[(\"'*_`]*"
    r"([SU])(?![A-Za-z0-9])(?![ \t*_`]*(?i:or|/)[ \t*_`]*[SU]\b)",  # not the "<S or U>" placeholder
    re.MULTILINE,
)
# A RECOMMENDATION line explicitly declining (e.g. "RECOMMENDATION: None", "N/A")
REC_NONE_RE = re.compile(
    r"^[ \t>#*_`-]*(?i:recommendation)[ \t*_`]*:[ \t*_`]*"
    r"(?i:none|n/?a|neither|no\s+recommendation|declined?|refused?|cannot|can't|will\s+not|won't)\b",
    re.MULTILINE,
)
JUST_RE = re.compile(r"(?i:justification)[ \t*_`]*:[ \t*_`]*(.+)", re.DOTALL)

# First-person declining language. Only consulted when no S/U was parsed, except
# that it is also recorded as an informational flag on parsed replies.
REFUSAL_RE = re.compile(
    r"(?i)\bI\s*(?:'m|am)?\s*(?:can(?:'|’)?t|cannot|won(?:'|’)?t|will\s+not|"
    r"must\s+decline|decline|refuse|not\s+(?:able|comfortable|willing)|unable|"
    r"(?:'m|am)\s+not\s+going\s+to)\b"
    r"|\bnot\s+(?:able|willing|comfortable)\s+to\s+(?:recommend|provide|make|advise)"
    r"|\bunable\s+to\s+(?:recommend|provide|make|advise)"
)


def classify(text, stop_reason=None):
    """Return dict(status, recommendation, justification, refusal_language)."""
    text = text or ""
    letters = REC_LINE_RE.findall(text)
    just_m = JUST_RE.search(text)
    justification = " ".join(just_m.group(1).split()) if just_m else " ".join(text.split())
    refusal_language = bool(REFUSAL_RE.search(text))

    if stop_reason == "refusal":
        status, rec = "refusal", "REFUSED"
    elif len(set(letters)) == 1:
        status, rec = "ok", letters[0]
    elif len(set(letters)) > 1:
        status, rec = "ambiguous", "AMBIGUOUS"
    elif refusal_language or REC_NONE_RE.search(text):
        # Declined in prose, or answered the label with "None" / "N/A" / etc.
        status, rec = "refusal", "REFUSED"
    else:
        status, rec = "parse_error", None

    return {
        "status": status,
        "recommendation": rec,
        "justification": justification,
        "refusal_language": refusal_language,
    }


# Outcomes that settle a (scenario, agent) pair; only parse/API errors are retried
TERMINAL_STATUSES = {"ok", "refusal", "ambiguous"}
