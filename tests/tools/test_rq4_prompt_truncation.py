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
