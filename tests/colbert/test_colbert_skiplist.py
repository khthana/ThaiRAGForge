"""Pin the punctuation skiplist to the symbol's own id, both directions.

`mask_punctuation` was wrong for a week in a way no test in this repo could see:
built from ``encode(sym)[0]`` it masked the SentencePiece word-boundary marker
``▁`` and kept every punctuation mark, which is the inverse of the flag's name.
Both rules return a plausible number of plausible vectors, so it took an
external comparison against pylate to surface -- ours dropped 2 and 3 tokens on
two hand-written Thai documents where pylate dropped 0 and 2.

The stub reproduces the property that makes the two rules disagree (encoding a
standalone symbol prepends the boundary marker) without a tokenizer download, so
this test states the rule rather than merely recording today's vocabulary.
"""
from __future__ import annotations

import string

from rag_lab.colbert import ColbertEncoder

SPACE = 6      # '▁' -- what `encode(sym)[0]` returns for every symbol
UNK = 3


class StubSentencePieceTokenizer:
    """`convert_tokens_to_ids` knows bare symbols; `encode` prepends `▁`."""

    unk_token_id = UNK

    def __init__(self, known: dict[str, int]):
        self._known = known

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._known.get(token, UNK)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [SPACE, self._known.get(text, UNK)]


def test_it_masks_the_punctuation_token_and_not_the_space_marker():
    tok = StubSentencePieceTokenizer({".": 5, ",": 4, "-": 9})
    skip = ColbertEncoder._build_skiplist(tok)
    assert skip == {5, 4, 9}
    assert SPACE not in skip, "the `encode(sym)[0]` route masks whitespace instead"


def test_a_symbol_missing_from_the_vocabulary_is_not_masked_as_unk():
    """`convert_tokens_to_ids` answers unk for an absent symbol, and masking unk
    would delete every out-of-vocabulary token in the corpus."""
    tok = StubSentencePieceTokenizer({".": 5})
    assert ColbertEncoder._build_skiplist(tok) == {5}


def test_every_ascii_symbol_is_offered_to_the_tokenizer():
    seen: list[str] = []

    class Recording(StubSentencePieceTokenizer):
        def convert_tokens_to_ids(self, token: str) -> int:
            seen.append(token)
            return super().convert_tokens_to_ids(token)

    ColbertEncoder._build_skiplist(Recording({}))
    assert seen == list(string.punctuation)
