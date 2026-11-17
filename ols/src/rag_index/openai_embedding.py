"""Embedding model backed by an OpenAI-compatible ``/v1/embeddings`` endpoint.

vLLM (pooling runner) and TEI both serve arbitrary model names over the OpenAI
protocol. ``llama_index.embeddings.openai.OpenAIEmbedding`` cannot be used for
them because it validates the model name against a fixed enum of OpenAI models,
so this module talks to the endpoint with the ``openai`` client directly.

The module imports llama-index at import time, so callers must import it lazily
(see ``ols.src.rag_index.embed_model``). The default in-process HuggingFace path
never loads it.
"""

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from openai import AsyncOpenAI, OpenAI


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Embedding model served by an OpenAI-compatible endpoint.

    The served model name is sent verbatim, so any model exposed by vLLM or TEI
    can be used as long as it matches the weights the RAG index was built with.
    """

    _api_base: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        timeout_s: float,
    ) -> None:
        """Initialize the remote client.

        Args:
            model_name: Model name as served by the endpoint.
            api_base: Base URL of the OpenAI-compatible API (``.../v1``).
            api_key: Bearer token sent to the endpoint.
            timeout_s: Per-request timeout in seconds.
        """
        super().__init__(model_name=model_name)
        self._api_base = api_base
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout_s)
        self._async_client = AsyncOpenAI(
            base_url=api_base, api_key=api_key, timeout=timeout_s
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        """Embed a query synchronously."""
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        """Embed a text synchronously."""
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one request."""
        return self._embed(list(texts))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Embed a query asynchronously."""
        return (await self._aembed([query]))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Embed a text asynchronously."""
        return (await self._aembed([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one asynchronous request."""
        return await self._aembed(list(texts))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return _ordered_embeddings(response.data)

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.embeddings.create(
            model=self.model_name, input=texts
        )
        return _ordered_embeddings(response.data)


def _ordered_embeddings(data: Any) -> list[list[float]]:
    """Return embeddings in input order; the API may reorder response items."""
    return [list(item.embedding) for item in sorted(data, key=lambda item: item.index)]
