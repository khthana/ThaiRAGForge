"""Standalone-phrase containment — the project's ONE copy of the rule.

This test was settled for this corpus by
``tools/eval/audit_gold_anchor_ambiguity.py`` (2026-08-09) and lived there until
``LexicalContainmentRetriever`` needed it: the core package must not import from
``tools/`` (ADR-0001), and two copies of a matching rule would eventually
disagree the way two copies of RRF would (see ``retrievers/hybrid.fuse_rrf``).
The audit script now imports it from here, so the retriever and every published
anchor-ambiguity figure are decided by the same code.

The rule has three parts, each of which was measured rather than chosen:

* **case-insensitive** — course and programme names appear in mixed case;
* **whitespace collapsed** — OCR'd minutes wrap a long name across a line, and
  matching raw text calls 3 genuine mentions absent (e.g. ``ENGLISH FOR
  ARCHITECTURAL PRESENTATION`` gains a naming document and loses its only
  apparently-silent one). Collapsing is the conservative direction: an inflated
  "does not name it" count would invent a failure mechanism that isn't there;
* **Latin-alphanumeric boundary** — so ``CALCULUS 2`` does not match inside
  ``CALCULUS 21``. The boundary is deliberately *Latin* alphanumeric, because
  Thai script has no word delimiter and requiring a boundary there would reject
  every Thai name in running text.

Contract, and it is load-bearing for cost: **the caller collapses the haystack
once**, not once per needle — 33 names x 2,853 documents is the difference
between seconds and minutes. ``collapse_ws`` is exported for that. The needle is
collapsed here, since it is short and callers pass it straight from a
dictionary.
"""
from __future__ import annotations

import re
import string

_WS = re.compile(r"\s+")
_ALNUM = set(string.ascii_letters + string.digits)


def collapse_ws(text: str) -> str:
    """Collapse every whitespace run to a single space.

    Apply to a haystack ONCE before calling `contains_phrase` on it repeatedly.
    """
    return _WS.sub(" ", text)


def contains_phrase(haystack: str, needle: str) -> bool:
    """Case-insensitive containment of `needle` as a standalone phrase.

    `haystack` must already be whitespace-collapsed (see `collapse_ws`). The
    needle is collapsed here.

    One deliberate difference from the copy this replaced, verified to be the
    ONLY one (16,084,108 haystack x needle pairs, 0 other disagreements): an
    **empty needle returns False**. The old code returned True for it whenever
    the haystack held a non-Latin-alphanumeric position -- i.e. always, for Thai
    text -- which would mark every chunk as containing the entity. No caller can
    reach it (needles come from entity dictionaries), so no published figure
    moves; it is guarded because "everything contains nothing" is the wrong
    answer for a *standalone phrase* test, not because anything was failing.
    """
    needle = _WS.sub(" ", needle)
    if not needle:
        return False
    for m in re.finditer(re.escape(needle), haystack, re.IGNORECASE):
        before = haystack[m.start() - 1] if m.start() > 0 else ""
        after = haystack[m.end()] if m.end() < len(haystack) else ""
        if before in _ALNUM or after in _ALNUM:
            continue
        return True
    return False
