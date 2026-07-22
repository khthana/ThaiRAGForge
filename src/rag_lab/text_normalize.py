"""Thai text normalization for RQ3's normalization ablation
(docs/research-framework-gap-analysis.md Sec.6, Tier 3 item 7).

Two enumerated transforms, applied in this order -- nothing else:

1. Thai digit -> Arabic digit (๑๒๓ -> 123), via
   ``pythainlp.util.thai_digit_to_arabic_digit``. pythainlp's own
   ``normalize()`` (below) has no opinion on digit direction, so this is a
   separate, explicit step.
2. ``pythainlp.util.normalize()``: removes zero-width spaces, collapses
   duplicate spaces/blank lines to one, removes spaces before tone marks,
   collapses duplicate vowels/tone marks, reorders tone marks to standard
   spelling, and drops dangling leading combining marks. See that function's
   docstring for the exact rule list.

Known side effect: step 2 collapses "\\n\\n" (paragraph breaks) down to a
single "\\n". This is inert for the two chunkers this ablation targets --
`semantic` splits on sentences (pythainlp `sent_tokenize`) and `fixed_size` /
`fixed_size_wordaware` split on raw character/word count -- neither reads
blank-line structure. It would matter for `RecursiveChunker`, whose
`"\\n\\n"`-first separator preference depends on doubled newlines surviving;
that combination is not exercised by this ablation. It also never touches
`## Page N` markers: `chunkers.pages.segment_by_page` matches those line by
line regardless of surrounding blank-line spacing.

Applied symmetrically: the `normalized` loader (rag_lab.loaders.normalized)
runs this over the corpus at build time, and the RQ3 eval script runs the
identical function over each gold query at retrieval time. Normalizing only
one side would make exact-match retrieval (BM25, and therefore `hybrid`)
artificially worse after "normalization" purely from losing lexical overlap
between a digit form used in the corpus and a different one still used in
the query -- an asymmetry artifact, not a normalization effect.
"""
from __future__ import annotations

from pythainlp.util import normalize, thai_digit_to_arabic_digit


def normalize_thai_text(text: str) -> str:
    return normalize(thai_digit_to_arabic_digit(text))
