"""Pins the two facts the 2026-08-10 RQ4 truncation finding rests on.

Both were established by measurement against ollama 0.32.6 and both are the
kind of thing that goes quietly wrong later: a truncation signature that stops
matching, or a prompt layout change that reverses which documents a cut
destroys. See docs/rq4-prompt-truncation.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

from rq4_generate import build_prompt, truncated_to  # noqa: E402


def test_truncation_signature_reproduces_both_measured_points():
    """One 14,721-token prompt reported 2050 at num_ctx=4096 and 4098 at 8192.

    The guard in rq4_generate detects truncation by comparing
    `prompt_eval_count` against exactly this value, so if the arithmetic drifts
    the guard silently stops firing -- which is the failure it exists to
    prevent, one level up.
    """
    assert truncated_to(4096) == 2050
    assert truncated_to(8192) == 4098


def test_documents_precede_instructions_and_keep_rank_order():
    """Why front-truncation is the worst possible cut here.

    `build_prompt` lays documents out best-first and puts the rules last. So a
    prompt that overflows loses its *highest-ranked* documents while its rules
    survive intact -- the answer still comes back well-formed and the damage is
    invisible in the output. That asymmetry is the whole finding; if the layout
    is ever flipped back, this test should fail loudly rather than let the
    published explanation quietly become wrong.
    """
    ctx = {
        "query": "คำถามทดสอบ",
        "blocks": [
            {"label": 1, "resolution_id": "r1", "text": "BEST-RANKED"},
            {"label": 2, "resolution_id": "r2", "text": "SECOND"},
            {"label": 3, "resolution_id": "r3", "text": "WORST-RANKED"},
        ],
    }
    prompt = build_prompt(ctx, "cite_all_guarded")

    best = prompt.index("BEST-RANKED")
    worst = prompt.index("WORST-RANKED")
    rules = prompt.index("กติกา:")

    assert best < worst, "documents must be laid out best-first"
    assert worst < rules, "instructions must come after every document"


def test_closed_book_prompt_cannot_overflow():
    """The no-context arm carries no documents, so it is structurally immune.

    It is the control the blast-radius table leans on: 0 of its 106 prompts
    were truncated at num_ctx=8192, which is what makes 'the damage tracks
    context length' a measurement rather than an assumption.
    """
    prompt = build_prompt({"query": "คำถาม", "blocks": []}, "cite_all_guarded")
    assert "ไม่มีเอกสารประกอบ" in prompt
    assert len(prompt) < 2000


def test_the_screen_can_only_over_estimate_tokens():
    """The screen must never call a prompt safe that isn't.

    `preflight` clears a prompt without a forward pass when its token *upper
    bound* fits num_ctx, so the bound must hold for any input, not merely for
    the samples measured so far. It is now UTF-8 byte length: a byte-level BPE
    token consumes at least one byte, so tokens <= bytes always.

    The previous bound, `int(n_chars / 1.046) + 1`, was an empirical extreme
    (docs/rq4-prompt-truncation.md section 5) and the data violates it -- 15 of
    228 measured prompts realize fewer chars/token, minimum 1.0098. That is the
    case pinned first here, because under the old rule it screened *out*.
    """
    from rq4_generate import token_upper_bound

    # bm25_semantic/q001: 11,208 chars -> 11,099 real tokens, i.e. 1.0098
    # chars/token. The old bound read 10,715 and would have called it safe at
    # num_ctx=11_000; the byte bound (Thai prose, ~3 bytes/char) cannot.
    thai = "ก" * 11_208
    assert 11_208 / 1.046 < 11_099          # the old bound really did under-read
    assert token_upper_bound(thai) >= 11_099

    # English course tables at ~3.15 chars/token: the bound over-estimates
    # heavily (1 byte/char, so ~3x the real count). Wasted probing, never a
    # missed truncation -- the direction a screen must err in.
    assert token_upper_bound("a" * 15_915) >= 5_051
    assert token_upper_bound("") == 0


def test_the_longest_prompt_in_chars_is_not_the_screen():
    """The regression that motivated the rewrite.

    entity_boost's longest prompt by characters is 15,689 chars / 4,860 tokens
    (English-heavy), while its true worst is 14,721 tokens. A screen that
    measured only the longest-by-characters prompt would clear num_ctx=8192 on a
    4,860-token reading and then truncate ~half the run. The upper bound must
    flag it as a candidate at 8192 regardless of how it tokenizes -- and must
    keep flagging a Thai prompt of the same length, whose real count is ~3x more.
    """
    from rq4_generate import token_upper_bound

    assert token_upper_bound("a" * 15_689) > 8_192     # probed at 8192, as it must be
    assert token_upper_bound("ก" * 15_689) > 8_192     # and the Thai case a fortiori
