"""Late-interaction (ColBERT) encoding for `jinaai/jina-colbert-v2`.

This is deliberately NOT a `BaseEmbedder`. That interface returns one row per
text and the whole `Index` is built on it: `embeddings.npy` is strictly
row-aligned with `chunks.parquet`, and that alignment is what makes
`resolution_id` attribution safe (audit check I1). ColBERT produces a *variable
number* of vectors per chunk, so it needs its own artifact shape and its own
alignment check, and pretending otherwise would quietly break the one invariant
this project's silent-corruption bugs keep landing on.

FIVE PLACES THIS CAN BE WRONG WHILE STILL RETURNING PLAUSIBLE NUMBERS
---------------------------------------------------------------------
Each is settled here against something measured, not against a recollection of
ColBERT's source, because a subtly wrong encoder still ranks documents in a
believable order -- the `gte-multilingual-reranker-base` failure shape.

1. **Marker insertion.** Original ColBERT tokenizes ``". " + text`` and then
   overwrites ``ids[:, 1]`` with the marker id, because on BERT WordPiece ``". "``
   is exactly one token. On this model's SentencePiece vocabulary it is **two**
   (``▁`` + ``.``), so that route leaves a stray ``.`` at position 2 in every
   query. Measured, not assumed. We prepend the marker's literal string, which
   is what the model card's own pylate snippet does, and the two routes are
   verified to disagree in `tools/eval/qualify_colbert_model.py` (C1).
2. **Query augmentation.** A query is padded to `query_maxlen` with ``<mask>``
   and **truncated** past it. All `query_maxlen` vectors are kept -- the mask
   vectors are the point, not padding.
3. **Attending to those masks.** `attend_to_mask_tokens` is true in this
   checkpoint's `artifact.metadata`, so the attention mask is forced to 1 on the
   expansion tokens. With it left at 0 the model still runs and still ranks.
4. **The projection head.** `config.json`'s `auto_map` sends AutoModel to a plain
   `XLMRobertaModel`, so ``linear.weight`` (1024 -> 128) is **not** loaded -- the
   loader prints it as UNEXPECTED and carries on. It is pulled from the
   safetensors by hand here.
5. **L2 normalisation.** Projected vectors come out with norm ~15, so cosine
   MaxSim without normalising is a length-weighted dot product. It is the
   caller's job, and it is done here.

A SIXTH, WHICH IS THE CHECKPOINT'S AND NOT OURS
-----------------------------------------------
Under transformers 5.x this model loads **position-blind** and has to be
repaired at load time (`_repair_rotary`). `RotaryEmbedding.inv_freq` is a
*non-persistent* buffer, so it is absent from the safetensors and is supposed to
be rebuilt by `__init__`; the 5.x loader materialises it from the meta device
instead and never re-runs that code, so all 24 layers come up holding
uninitialised memory. Measured here: 30 of 32 entries exactly 0 and 2 denormal
(2.6e-29, 1.0e-42), which makes `cos = 1`, `sin = 0`, i.e. the rotation is the
identity. The model still loads, still runs, and still ranks a Thai example
correctly -- it simply cannot tell one word order from another (a document and
its token-reversal came out identical to 1.3e-05 relative).

The repair recomputes `inv_freq` with the model's **own** `_compute_inv_freq`
and invalidates the cos/sin cache. That is restoration, not modification:
`inv_freq` is a deterministic function of `(dim, base)` written in this
checkpoint's own code, carries no trained information, and is exactly what
`__init__` would have produced.

Two things not to conclude from this. A buffer being *non-zero* does not make it
correct -- an earlier probe found all 24 `inv_freq` non-zero and read that as
"the gte zeroing did not happen", when in fact the contents were garbage and the
corruption is **nondeterministic across loads** (zeros on one load, denormals on
the next). And a *behavioural* check is not sufficient either: uninitialised
memory that happened to be large would give pseudo-random-but-stable rotations,
under which the model looks position-sensitive while being just as wrong. The
decisive check is the exact one -- C7 in `tools/eval/qualify_colbert_model.py`
compares every layer against `_compute_inv_freq` -- with G2 kept as the
behavioural backstop.

Punctuation masking carries a quirk worth knowing: ColBERT builds its skiplist
from the *first* token of each ASCII punctuation symbol, and on SentencePiece
that first token is usually the bare ``▁`` space marker. So `mask_punctuation`
here mostly drops whitespace tokens and keeps ``.`` -- faithful to the reference
implementation (pylate does the same), just not what the name suggests.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

MODEL_NAME = "jinaai/jina-colbert-v2"


@dataclass(frozen=True)
class ColbertConfig:
    """Defaults are the checkpoint's own `artifact.metadata`, not our preferences.

    `doc_maxlen`/`query_maxlen` are ColBERT conventions rather than model limits
    (this checkpoint is rotary; its card claims 8192 tokens), so raising them is
    possible -- but it is an *unmeasured* deviation, and this project treats
    those as worse than a measured handicap. `data/results/colbert_length_profile.md`
    prices both: at 300 the corpus loses the tail of 1.1-7.4% of chunks, at 512
    of 0.0-3.3% for +3.7% storage; `query_maxlen` 32 truncates 8% of Gold
    queries by at most 5 tokens, 48 truncates none.
    """

    model_name: str = MODEL_NAME
    dim: int = 128              # full Matryoshka width; 96/64 also trained
    query_maxlen: int = 32
    doc_maxlen: int = 300
    mask_punctuation: bool = True
    attend_to_mask_tokens: bool = True
    query_prefix: str = "[QueryMarker]"
    document_prefix: str = "[DocumentMarker]"
    repair_rotary: bool = True  # off only to demonstrate the bug; see `_repair_rotary`


class ColbertEncoder:
    """Loads lazily; holds the model until `release()`."""

    def __init__(
        self,
        config: ColbertConfig | None = None,
        device: str | None = None,
        dtype: str = "float16",
        batch_size: int = 32,
    ) -> None:
        self.config = config or ColbertConfig()
        self._device = device
        self._dtype_name = dtype
        self._batch_size = batch_size
        self._model = None
        self._tok = None
        self._linear = None
        self._skiplist: set[int] = set()
        self.rotary_repaired: int | None = None   # layers whose inv_freq was wrong at load

    # ------------------------------------------------------------------ load
    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoModel, AutoTokenizer

        name = self.config.model_name
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, self._dtype_name)

        self._tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=dtype)
        model.eval().to(self._device)
        self._model = model
        if self.config.repair_rotary:
            self.rotary_repaired = self._repair_rotary(model)

        # (4) the projection head AutoModel does not load
        path = hf_hub_download(name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor("linear.weight")
        if tuple(w.shape) != (self.config.dim, model.config.hidden_size):
            raise ValueError(
                f"linear.weight is {tuple(w.shape)}, expected "
                f"({self.config.dim}, {model.config.hidden_size})"
            )
        self._linear = w.to(self._device, dtype)

        if self.config.mask_punctuation:
            for sym in string.punctuation:
                ids = self._tok.encode(sym, add_special_tokens=False)
                if ids:
                    self._skiplist.add(ids[0])
        return model

    @staticmethod
    def _repair_rotary(model) -> int:
        """Rebuild every layer's `inv_freq` from the model's own formula.

        See the module docstring: transformers 5.x materialises this
        non-persistent buffer as uninitialised memory, which silently makes the
        rotation the identity. Returns how many layers were actually wrong, so
        the caller can report a real count rather than assert a fix -- the number
        varies across loads, and a future transformers that restores the buffer
        correctly should make this return 0 without anything else changing.

        The cos/sin cache is invalidated too: `_update_cos_sin_cache` rebuilds
        only when the sequence length grows, so a corrected `inv_freq` alone
        would be ignored for every length already seen.

        The buffer is restored in **float32**, which is what `__init__` registers
        and what `_update_cos_sin_cache` reads directly; at any other dtype that
        method recomputes `inv_freq` itself on every call, so a corrupt fp16
        buffer would already have self-healed. This one does not, and that is
        measured -- the observed garbage (1.0e-42) is a float32 denormal, a value
        fp16 cannot even represent.
        """
        import torch

        repaired = 0
        for module in model.modules():
            re_ = getattr(module, "rotary_emb", None)
            if re_ is None or not hasattr(re_, "_compute_inv_freq"):
                continue
            good = re_._compute_inv_freq(device=re_.inv_freq.device)
            if not torch.equal(re_.inv_freq.float(), good):
                repaired += 1
            re_.inv_freq = good
            re_._seq_len_cached = 0
            re_._cos_cached = re_._sin_cached = None
            re_._cos_k_cached = re_._sin_k_cached = None
        return repaired

    def release(self) -> None:
        if self._model is None:
            return
        import torch

        self._model = self._linear = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- internals
    def _forward(self, input_ids, attention_mask):
        """hidden -> projected -> L2-normalised. (4) and (5)."""
        import torch

        with torch.no_grad():
            h = self._model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            v = h @ self._linear.T
            return torch.nn.functional.normalize(v.float(), p=2, dim=-1)

    # ---------------------------------------------------------------- queries
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """(n, query_maxlen, dim) float32. Every position is kept, including the
        `<mask>` expansion tokens -- that is ColBERT query augmentation, not
        padding, so there is nothing to strip."""
        import torch

        self._load()
        cfg, tok = self.config, self._tok
        out = np.empty((len(texts), cfg.query_maxlen, cfg.dim), dtype=np.float32)
        for i in range(0, len(texts), self._batch_size):
            batch = [f"{cfg.query_prefix} {t}" for t in texts[i : i + self._batch_size]]
            enc = tok(batch, padding="max_length", truncation=True,
                      max_length=cfg.query_maxlen, return_tensors="pt")
            ids, am = enc["input_ids"], enc["attention_mask"]
            ids = ids.masked_fill(ids == tok.pad_token_id, tok.mask_token_id)  # (2)
            if cfg.attend_to_mask_tokens:                                       # (3)
                am = torch.ones_like(am)
            v = self._forward(ids.to(self._device), am.to(self._device))
            out[i : i + len(batch)] = v.cpu().numpy()
        return out

    # -------------------------------------------------------------- documents
    def encode_documents(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Packed token vectors plus one length per text.

        Returns ``(vecs, lengths)`` with ``vecs.shape == (lengths.sum(), dim)``
        in fp16 and ``lengths.sum()`` matching exactly -- that pair *is* the
        alignment invariant, the ColBERT analogue of I1's chunks<->rows check,
        and anything consuming this must verify it rather than trust it.

        Masked positions (padding, and punctuation when `mask_punctuation`) are
        dropped rather than zeroed. That is equivalent to ColBERT's ``-9999``
        trick under a max, and it does not cost storage for vectors no MaxSim
        can ever select.
        """
        self._load()
        cfg, tok = self.config, self._tok
        chunks: list[np.ndarray] = []
        lengths = np.empty(len(texts), dtype=np.int64)
        for i in range(0, len(texts), self._batch_size):
            batch = [f"{cfg.document_prefix} {t}" for t in texts[i : i + self._batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=cfg.doc_maxlen, return_tensors="pt")
            ids, am = enc["input_ids"], enc["attention_mask"]
            v = self._forward(ids.to(self._device), am.to(self._device)).cpu().numpy()
            id_rows = ids.numpy()
            for j in range(len(batch)):
                keep = (am[j].numpy() == 1)
                if self._skiplist:
                    keep &= ~np.isin(id_rows[j], list(self._skiplist))
                if not keep.any():  # a document of nothing but punctuation
                    keep[0] = True
                chunks.append(v[j][keep].astype(np.float16))
                lengths[i + j] = int(keep.sum())
        vecs = (np.concatenate(chunks, axis=0) if chunks
                else np.empty((0, cfg.dim), dtype=np.float16))
        if vecs.shape[0] != int(lengths.sum()):
            raise AssertionError(
                f"packed {vecs.shape[0]} vectors for {int(lengths.sum())} claimed tokens")
        return vecs, lengths
