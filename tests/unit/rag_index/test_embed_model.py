"""Unit tests for the RAG embedding model factory."""

from unittest.mock import patch

import pytest

from ols import constants
from ols.app.models.config import EmbedModelConfig
from ols.src.rag_index.embed_model import get_embed_model
from ols.src.rag_index.openai_embedding import OpenAICompatibleEmbedding

_HF_PATCH = "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
_OPENAI_PATCH = "ols.src.rag_index.openai_embedding.OpenAICompatibleEmbedding"


def test_get_embed_model_uses_local_default_when_not_configured():
    """Without an ``embed_model`` section each RAG path keeps its bundled model."""
    with patch(_HF_PATCH) as mock_hf:
        model = get_embed_model(None, constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)

    mock_hf.assert_called_once_with(model_name=constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)
    assert model is mock_hf.return_value


def test_get_embed_model_keeps_per_path_local_default():
    """The Solr path falls back to its own bundled model, not the BYOK one."""
    with patch(_HF_PATCH) as mock_hf:
        get_embed_model(None, constants.SOLR_HYBRID_EMBEDDING_MODEL_ID)

    mock_hf.assert_called_once_with(model_name=constants.SOLR_HYBRID_EMBEDDING_MODEL_ID)


def test_get_embed_model_local_model_override():
    """``embed_model.model`` overrides the bundled model name for local weights."""
    embed_config = EmbedModelConfig({"type": "huggingface", "model": "custom-embedder"})

    with patch(_HF_PATCH) as mock_hf:
        get_embed_model(embed_config, constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)

    mock_hf.assert_called_once_with(model_name="custom-embedder")


@pytest.mark.parametrize(
    "embed_type",
    [constants.EMBED_MODEL_TYPE_VLLM, constants.EMBED_MODEL_TYPE_TEI],
)
def test_get_embed_model_remote_backend(embed_type):
    """VLLM and TEI share the OpenAI-compatible client."""
    embed_config = EmbedModelConfig(
        {
            "type": embed_type,
            "url": "http://embedding-host:8000/v1/",
            "model": "ibm-granite/granite-embedding-small-english-r2",
            "api_key": "token-from-file",
            "timeout_s": 12.5,
        }
    )

    with patch(_OPENAI_PATCH) as mock_openai:
        model = get_embed_model(embed_config, constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)

    mock_openai.assert_called_once_with(
        model_name="ibm-granite/granite-embedding-small-english-r2",
        api_base="http://embedding-host:8000/v1",
        api_key="token-from-file",
        timeout_s=12.5,
    )
    assert model is mock_openai.return_value


def test_get_embed_model_remote_backend_without_credentials():
    """Unauthenticated endpoints get the OpenAI client placeholder key."""
    embed_config = EmbedModelConfig(
        {
            "type": constants.EMBED_MODEL_TYPE_VLLM,
            "url": "http://embedding-host:8000/v1",
            "model": "granite-embedding-small-english-r2",
        }
    )

    with patch(_OPENAI_PATCH) as mock_openai:
        get_embed_model(embed_config, constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)

    assert (
        mock_openai.call_args.kwargs["api_key"]
        == constants.EMBED_MODEL_PLACEHOLDER_API_KEY
    )


def test_get_embed_model_remote_backend_accepts_served_model_name():
    """A model name unknown to the OpenAI client is passed through verbatim."""
    embed_config = EmbedModelConfig(
        {
            "type": constants.EMBED_MODEL_TYPE_VLLM,
            "url": "http://embedding-host:8000/v1",
            "model": "granite-embedding-small-english-r2",
            "api_key": "token-from-file",
        }
    )

    model = get_embed_model(embed_config, constants.EMBEDDINGS_MODEL_BYOK_SUBDIR)

    assert isinstance(model, OpenAICompatibleEmbedding)
    assert model.model_name == "granite-embedding-small-english-r2"
