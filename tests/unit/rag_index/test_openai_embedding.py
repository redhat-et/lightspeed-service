"""Unit tests for the OpenAI-compatible embedding model wrapper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ols.src.rag_index.openai_embedding import OpenAICompatibleEmbedding

_CLIENT_PATCH = "ols.src.rag_index.openai_embedding.OpenAI"
_ASYNC_CLIENT_PATCH = "ols.src.rag_index.openai_embedding.AsyncOpenAI"


def _response(vectors):
    """Build an OpenAI embeddings response, with items deliberately reordered."""
    data = [
        SimpleNamespace(index=index, embedding=vector)
        for index, vector in reversed(list(enumerate(vectors)))
    ]
    return SimpleNamespace(data=data)


def test_constructor_accepts_model_name_unknown_to_openai_client():
    """A served model name is passed through without enum validation."""
    embedding = OpenAICompatibleEmbedding(
        model_name="granite-embedding-small-english-r2",
        api_base="http://embedding-host:8000/v1",
        api_key="secret",
        timeout_s=7.5,
    )

    assert embedding.model_name == "granite-embedding-small-english-r2"
    assert str(embedding._client.base_url) == "http://embedding-host:8000/v1/"
    assert embedding._client.api_key == "secret"


def test_clients_receive_endpoint_credentials_and_timeout():
    """Both sync and async clients are built from the configured endpoint."""
    with (
        patch(_CLIENT_PATCH) as mock_sync,
        patch(_ASYNC_CLIENT_PATCH) as mock_async,
    ):
        OpenAICompatibleEmbedding(
            model_name="granite-embedding-small-english-r2",
            api_base="http://embedding-host:8000/v1",
            api_key="secret",
            timeout_s=7.5,
        )

    mock_sync.assert_called_once_with(
        base_url="http://embedding-host:8000/v1", api_key="secret", timeout=7.5
    )
    mock_async.assert_called_once_with(
        base_url="http://embedding-host:8000/v1", api_key="secret", timeout=7.5
    )


def test_get_text_embedding_returns_embedding_in_response_order():
    """Embeddings are returned in the order the texts were sent."""
    with patch(_CLIENT_PATCH) as mock_sync:
        mock_sync.return_value.embeddings.create.return_value = _response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        embedding = OpenAICompatibleEmbedding(
            model_name="granite-embedding-small-english-r2",
            api_base="http://embedding-host:8000/v1",
            api_key="secret",
            timeout_s=7.5,
        )
        result = embedding.get_text_embedding_batch(["first", "second"])

    mock_sync.return_value.embeddings.create.assert_called_once_with(
        model="granite-embedding-small-english-r2", input=["first", "second"]
    )
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_get_query_embedding_sends_single_input():
    """Query embedding uses the same single-request path as text embedding."""
    with patch(_CLIENT_PATCH) as mock_sync:
        mock_sync.return_value.embeddings.create.return_value = _response([[1.0, 2.0]])
        embedding = OpenAICompatibleEmbedding(
            model_name="granite-embedding-small-english-r2",
            api_base="http://embedding-host:8000/v1",
            api_key="secret",
            timeout_s=7.5,
        )
        result = embedding.get_query_embedding("what is an HPA?")

    mock_sync.return_value.embeddings.create.assert_called_once_with(
        model="granite-embedding-small-english-r2", input=["what is an HPA?"]
    )
    assert result == [1.0, 2.0]


@pytest.mark.asyncio
async def test_aget_text_embeddings_uses_async_client():
    """Async embedding goes through the async client."""
    with patch(_ASYNC_CLIENT_PATCH) as mock_async:
        mock_async.return_value.embeddings.create = AsyncMock(
            return_value=_response([[0.5, 0.6], [0.7, 0.8]])
        )
        embedding = OpenAICompatibleEmbedding(
            model_name="granite-embedding-small-english-r2",
            api_base="http://embedding-host:8000/v1",
            api_key="secret",
            timeout_s=7.5,
        )
        result = await embedding.aget_text_embedding_batch(["first", "second"])

    mock_async.return_value.embeddings.create.assert_awaited_once_with(
        model="granite-embedding-small-english-r2", input=["first", "second"]
    )
    assert result == [[0.5, 0.6], [0.7, 0.8]]


@pytest.mark.asyncio
async def test_aget_query_embedding_uses_async_client():
    """Async query embedding goes through the async client."""
    with patch(_ASYNC_CLIENT_PATCH) as mock_async:
        mock_async.return_value.embeddings.create = AsyncMock(
            return_value=_response([[0.9, 1.0]])
        )
        embedding = OpenAICompatibleEmbedding(
            model_name="granite-embedding-small-english-r2",
            api_base="http://embedding-host:8000/v1",
            api_key="secret",
            timeout_s=7.5,
        )
        result = await embedding.aget_query_embedding("what is an HPA?")

    assert result == [0.9, 1.0]
