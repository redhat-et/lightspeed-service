"""Embedding model factory for RAG.

All RAG paths -- local BYOK indexes, Solr hybrid search and tool/skill filtering --
embed queries through this module, so a single configuration block switches every
path between bundled HuggingFace weights and a remote OpenAI-compatible endpoint.
"""

import logging
from typing import Any, Optional

from ols import constants
from ols.app.models.config import EmbedModelConfig

logger = logging.getLogger(__name__)


def get_embed_model(
    embed_model_config: Optional[EmbedModelConfig],
    local_model_name: str,
) -> Any:
    """Build the embedding model used to encode RAG queries.

    Args:
        embed_model_config: Optional ``ols_config.embed_model`` section.
        local_model_name: Bundled HuggingFace model to use when no remote backend
            is configured. Each RAG path keeps its own default.

    Returns:
        A llama-index ``BaseEmbedding`` instance.
    """
    if (
        embed_model_config is None
        or embed_model_config.type == constants.EMBED_MODEL_TYPE_HUGGINGFACE
    ):
        configured_name = embed_model_config.model if embed_model_config else None
        return _build_huggingface_embedding(configured_name or local_model_name)
    return _build_remote_embedding(embed_model_config)


def _build_huggingface_embedding(model_name: str) -> Any:
    """Load bundled weights in-process."""
    from llama_index.embeddings.huggingface import (  # pylint: disable=import-outside-toplevel
        HuggingFaceEmbedding,
    )

    logger.debug("Loading in-process embedding model: %s", model_name)
    return HuggingFaceEmbedding(model_name=model_name)


def _build_remote_embedding(embed_model_config: EmbedModelConfig) -> Any:
    """Build a client for a remote OpenAI-compatible embedding endpoint."""
    # llama_index's OpenAIEmbedding rejects model names it does not know, but
    # vLLM and TEI serve arbitrary names, so use a plain openai client wrapper.
    from ols.src.rag_index.openai_embedding import (  # pylint: disable=C0415
        OpenAICompatibleEmbedding,
    )

    logger.info(
        "Using remote %s embedding endpoint %s with model %s",
        embed_model_config.type,
        embed_model_config.url,
        embed_model_config.model,
    )
    return OpenAICompatibleEmbedding(
        model_name=str(embed_model_config.model),
        api_base=str(embed_model_config.url).rstrip("/"),
        api_key=embed_model_config.api_key or constants.EMBED_MODEL_PLACEHOLDER_API_KEY,
        timeout_s=embed_model_config.timeout_s,
    )
