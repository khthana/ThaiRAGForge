from __future__ import annotations

from typing import Any

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_text,
    strip_document_header,
    strip_mapping_tables,
)
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution

# HF token-classification checkpoints fine-tuned on the thainer corpus that we
# load ourselves (instead of pythainlp's engine="wangchanberta"/"phayathaibert",
# which hardcode their own checkpoints and always run on CPU).
_HF_NER_MODELS = {
    "wangchanberta-thainer": "Porameht/wangchanberta-thainer-corpus-v2-2",
    "phayathaibert-thainer": "Pavarissy/phayathaibert-thainer",
}

_ner_tagger = None
_ner_tagger_engine: str | None = None


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _group_entities(tagged: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Collapse IOB word tags (B-PERSON, I-PERSON, O, ...) into entity spans."""
    entities: list[dict[str, str]] = []
    text = ""
    tag: str | None = None

    def flush() -> None:
        nonlocal text, tag
        if tag is not None and text.strip():
            entities.append({"text": text.strip(), "tag": tag})
        text, tag = "", None

    for word, label in tagged:
        if label.startswith("B-"):
            flush()
            tag = label[2:]
            text = word
        elif label.startswith("I-") and tag == label[2:]:
            text += word
        else:
            flush()
    flush()
    return entities


class _PyThaiNLPTagger:
    """Wraps a stock pythainlp `NER` engine (e.g. "thainer", the CRF default)."""

    def __init__(self, engine: str) -> None:
        from pythainlp.tag import NER

        self._ner = NER(engine)

    def entities(self, text: str) -> list[dict[str, str]]:
        return _group_entities(self._ner.tag(text))


class _WangchanBERTaTagger:
    """GPU-capable re-implementation of pythainlp's wangchanberta NER inference
    (word_tokenize -> is_split_into_words -> IOB tags), pointed at an arbitrary
    wangchanberta-based thainer checkpoint. The upstream
    `pythainlp.wangchanberta.core.NamedEntityRecognition` class always runs on
    CPU, with no way to move the model or its inputs to CUDA.

    Meeting resolutions routinely exceed the model's 512-position cap, so the
    word sequence is recursively bisected until each half's *actual* tokenized
    length fits (RoBERTa-style position ids start at index 2, so the usable
    budget is `max_position_embeddings - 2`, not the raw config value -- and
    per-word piece counts estimated in isolation drift from the real count
    once words are joined, so pre-computing a fixed-size chunk plan isn't
    reliable; checking the real tokenized length and only splitting on actual
    overflow is)."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self._device = _device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForTokenClassification.from_pretrained(model_name)
            .to(self._device)
            .eval()
        )
        self._limit = self._model.config.max_position_embeddings - 2

    def _tag_words(self, words: list[str]) -> list[tuple[str, str]]:
        inputs = self._tokenizer(words, is_split_into_words=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self._limit:
            if len(words) == 1:
                # a single word alone tokenizes past the limit (pathological
                # OCR garbage run); hard-truncate rather than recurse forever
                inputs = self._tokenizer(
                    words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=self._limit,
                )
            else:
                mid = len(words) // 2
                return self._tag_words(words[:mid]) + self._tag_words(words[mid:])

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        predictions = self._torch.argmax(logits, dim=2)[0].tolist()
        labels = [self._model.config.id2label[p] for p in predictions]

        tagged: list[tuple[str, str]] = []
        for token_id, label in zip(inputs["input_ids"][0].tolist(), labels):
            decoded = self._tokenizer.decode(token_id)
            if decoded.isspace() and label.startswith("B-"):
                label = "O"
            if decoded in ("", "<s>", "</s>"):
                continue
            if decoded == "<_>":
                decoded = " "
            tagged.append((decoded, label))
        return tagged

    def _tag(self, text: str) -> list[tuple[str, str]]:
        from pythainlp.tokenize import word_tokenize

        words = word_tokenize(text.replace(" ", "<_>"))
        if not words:
            return []
        return self._tag_words(words)

    def entities(self, text: str) -> list[dict[str, str]]:
        return _group_entities(self._tag(text))


class _PhayaThaiBERTTagger:
    """GPU-capable re-implementation of pythainlp's phayathaibert NER inference
    (a plain HF token-classification pipeline over raw text), pointed at an
    arbitrary phayathaibert-based thainer checkpoint.

    Note: unlike the CRF/wangchanberta engines above, this pipeline already
    returns *grouped* entity spans, not IOB word tags -- so it does not go
    through `_group_entities`.

    Uses aggregation_strategy="first" rather than pythainlp's own default of
    "simple": on this corpus "simple" fragments multi-word spans into a
    separate entity per SentencePiece subword the instant a token's leading
    "_" word-boundary marker confuses its merge heuristic (e.g. the
    institution's own full name split into 8 one-word ORGANIZATION entities);
    "first" collapses back to whole spans and was confirmed clean on real
    corpus documents.

    Meeting resolutions routinely exceed the model's 512-position cap, so text
    is split into token-budget-bounded chunks along real token boundaries
    (via the fast tokenizer's offset mapping) before each chunk is run through
    the pipeline separately."""

    def __init__(self, model_name: str) -> None:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            TokenClassificationPipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipeline = TokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=0 if _device() == "cuda" else -1,
        )
        self._max_len = model.config.max_position_embeddings

    def _chunks(self, text: str) -> list[str]:
        budget = self._max_len - 8  # room for special tokens plus a margin
        offsets = self._pipeline.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if len(offsets) <= budget:
            return [text]
        chunks = []
        for i in range(0, len(offsets), budget):
            group = [span for span in offsets[i : i + budget] if span[1] > span[0]]
            if group:
                chunks.append(text[group[0][0] : group[-1][1]])
        return chunks

    def entities(self, text: str) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for chunk in self._chunks(text):
            for span in self._pipeline(chunk):
                span_text = chunk[span["start"] : span["end"]].strip()
                if "\n" in span_text:
                    # "first"'s word-boundary fallback (triggered because this
                    # tokenizer doesn't expose real word ids) occasionally
                    # over-merges past a paragraph break into an unrelated
                    # section header/separator; a real entity in this corpus
                    # never spans a blank line, so keep only the first line
                    span_text = span_text.split("\n", 1)[0].strip()
                if span_text:
                    results.append({"text": span_text, "tag": span["entity_group"]})
        return results


def _tagger(engine: str):
    """Load the NER tagger lazily (and once per engine) -- the model is heavy."""
    global _ner_tagger, _ner_tagger_engine
    if _ner_tagger is None or _ner_tagger_engine != engine:
        if engine == "wangchanberta-thainer":
            _ner_tagger = _WangchanBERTaTagger(_HF_NER_MODELS[engine])
        elif engine == "phayathaibert-thainer":
            _ner_tagger = _PhayaThaiBERTTagger(_HF_NER_MODELS[engine])
        else:
            _ner_tagger = _PyThaiNLPTagger(engine)
        _ner_tagger_engine = engine
    return _ner_tagger


@loader_registry.register("ner")
class NERLoader(BaseLoader):
    """Runs Thai NER and stores extracted entities in metadata['entities'] for
    entity-aware filtering/analysis.

    `engine` selects the tagger: pythainlp engines ("thainer", "thainer-v2",
    ...) run as-is; "wangchanberta-thainer" and "phayathaibert-thainer" load
    the corresponding HF checkpoint directly on GPU when available."""

    def __init__(self, engine: str = "thainer") -> None:
        self.engine = engine

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        entities = _tagger(self.engine).entities(text)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "entities": entities,
        }
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=text,
            year=year,
            session=session,
            title=title,
            metadata=metadata,
        )
