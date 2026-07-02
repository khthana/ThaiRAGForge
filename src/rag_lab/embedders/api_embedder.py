from __future__ import annotations

import os

import numpy as np

from rag_lab.embedders.base import BaseEmbedder
from rag_lab.registries import embedder_registry


class MissingAPIKeyError(RuntimeError):
    """Raised when an APIEmbedder is constructed without its provider key set."""


@embedder_registry.register("api")
class APIEmbedder(BaseEmbedder):
    """Interface for a hosted embedding API (OpenAI/Voyage/Cohere).

    Inert until a key is supplied: construction raises immediately so a batch
    run isolates the combo as an error (run_experiment) instead of crashing
    later mid-batch. Provider request/batching is a stub — filled in once a
    real key and provider contract are available (HITL); do not build that
    dispatch against a guess of the provider's wire format.
    """

    def __init__(
        self, provider: str, model: str, api_key_env: str | None = None
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key_env = api_key_env or f"{provider.upper()}_API_KEY"
        if not os.environ.get(self._api_key_env):
            raise MissingAPIKeyError(
                f"APIEmbedder(provider={provider!r}) requires the "
                f"{self._api_key_env} environment variable to be set."
            )

    @property
    def model_id(self) -> str:
        return f"api:{self._provider}:{self._model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            f"APIEmbedder(provider={self._provider!r}) has a key but no request "
            "dispatch implemented yet."
        )
