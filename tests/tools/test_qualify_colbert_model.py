"""Pin the rotary repair and the C7 check that gates it, both directions.

`jinaai/jina-colbert-v2` loads remote code written for transformers 4.43 under
5.12, which materialises non-persistent buffers from the meta device instead of
re-running the `__init__` that built them -- so all 24 layers' `inv_freq` arrive
as uninitialised memory and the rotation is the identity. Two properties make
this worth a test rather than a comment:

* the garbage differs across loads (zeros, 2.6e-29, 1.6e-30 all observed in one
  session), so a threshold on the *values* is not the rule -- `inv_freq` is a
  deterministic function of `(dim, base)` and the rule is exact equality with the
  checkpoint's own `_compute_inv_freq`;
* `_repair_rotary` must self-retire: the day transformers restores the buffer
  correctly it has to return 0 and change nothing, or it becomes an unnoticed
  modification of a model rather than a restoration of it.

No download and no GPU -- the stub carries the same buffer name, the same
formula and the same cache attributes as the real `RotaryEmbedding`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from rag_lab.colbert import ColbertEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
from qualify_colbert_model import audit_rotary  # noqa: E402

# the values actually observed on a corrupt load, kept verbatim: both are finite
# and non-zero, which is why the "finite and not identically zero" buffer rule
# passed on a fully position-blind model.
OBSERVED_GARBAGE = (2.563874979166064e-29, 1.0299543712787405e-42)


class StubRotary(nn.Module):
    def __init__(self, dim: int = 64, base: float = 10000.0, corrupt: bool = False):
        super().__init__()
        self.dim, self._base = dim, base
        iv = self._compute_inv_freq()
        if corrupt:
            iv = torch.zeros_like(iv)
            iv[1], iv[7] = OBSERVED_GARBAGE
        self.register_buffer("inv_freq", iv, persistent=False)
        self._seq_len_cached = 300
        self._cos_cached = torch.ones(300, dim // 2)
        self._sin_cached = torch.zeros(300, dim // 2)
        self._cos_k_cached = self._sin_k_cached = None

    @property
    def base(self) -> float:
        return self._base

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))


class StubLayer(nn.Module):
    def __init__(self, corrupt: bool):
        super().__init__()
        self.rotary_emb = StubRotary(corrupt=corrupt)


class StubModel(nn.Module):
    def __init__(self, n: int = 3, corrupt: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([StubLayer(corrupt) for _ in range(n)])


def test_c7_catches_corruption_and_names_every_layer():
    ok, ev = audit_rotary(StubModel(corrupt=True))
    assert not ok
    assert "3 of 3 layers wrong" in ev


def test_repair_fixes_it_and_reports_how_many():
    model = StubModel(corrupt=True)
    assert ColbertEncoder._repair_rotary(model) == 3
    ok, _ = audit_rotary(model)
    assert ok


def test_repair_invalidates_the_cos_sin_cache():
    """A corrected `inv_freq` alone is ignored: `_update_cos_sin_cache` rebuilds
    only when the sequence length grows, so every length already seen keeps the
    identity rotation."""
    model = StubModel(corrupt=True)
    ColbertEncoder._repair_rotary(model)
    r = model.layers[0].rotary_emb
    assert r._seq_len_cached == 0
    assert r._cos_cached is None and r._sin_cached is None


def test_repair_restores_float32_and_leaves_it_non_persistent():
    model = StubModel(corrupt=True)
    ColbertEncoder._repair_rotary(model)
    r = model.layers[0].rotary_emb
    assert r.inv_freq.dtype is torch.float32
    assert "inv_freq" in dict(r.named_buffers())
    assert "inv_freq" not in dict(r.state_dict())


def test_repair_is_a_no_op_on_a_healthy_model():
    """It has to self-retire the day transformers loads the buffer correctly."""
    model = StubModel(corrupt=False)
    ok, _ = audit_rotary(model)
    assert ok
    before = model.layers[0].rotary_emb.inv_freq.clone()
    assert ColbertEncoder._repair_rotary(model) == 0
    assert torch.equal(before, model.layers[0].rotary_emb.inv_freq)


def test_c7_fails_when_there_is_no_rotary_at_all():
    """A checkpoint that is supposed to be rotary and has none is not a pass."""
    ok, ev = audit_rotary(nn.Linear(4, 4))
    assert not ok
    assert "no rotary_emb" in ev
