"""Reference Graphiti client factory backed by the unified runtime config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

CORE_ROOT = Path(__file__).resolve().parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


class RuntimeEmbedder(EmbedderClient):
    """Adapt the configured runtime callable to Graphiti's async protocol."""
    def __init__(self, callback, dimension: int):
        self.callback = callback
        self.dimension = max(1, int(dimension))

    async def create(self, input_data):
        if not self.callback:
            raise RuntimeError("Graphiti embedding adapter is unavailable")
        value = self.callback(input_data if isinstance(input_data, str) else " ".join(map(str, input_data)))
        if not value:
            raise RuntimeError("Graphiti embedding adapter returned an empty vector")
        return [float(item) for item in value]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch embedding over the runtime callable (one call per item).

        The vendored pipeline (node name embeddings, edge fact embeddings and
        resolution query embeddings) requires ``create_batch``; the base class
        default raises ``NotImplementedError`` which previously made every
        closed-session extraction fail at the embed step.
        """
        if not self.callback:
            raise RuntimeError("Graphiti embedding adapter is unavailable")
        return [await self.create(item) for item in input_data_list]


class RuntimeCrossEncoder(CrossEncoderClient):
    def __init__(self, callback):
        self.callback = callback

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not self.callback:
            raise RuntimeError("Graphiti reranker adapter is unavailable")
        return [(passage, float(self.callback(query, passage))) for passage in passages]


def create_graphiti_client(config: Dict[str, Any], section: str = "chat") -> Graphiti:
    database = config.get("database", {})
    chat = config.get(section, {}) or {}
    if not str(chat.get("model") or "").strip():
        raise ValueError(f"Graphiti {section}.model must be configured")
    llm_config = LLMConfig(
        # The SDK requires a non-empty credential at construction time even
        # when Graphiti is deliberately disabled (for local/offline tests).
        # Use an inert sentinel; service.graphiti_enabled gates all network use.
        api_key=chat.get("api_key") or "graphiti-disabled",
        model=chat.get("model") or "",
        base_url=chat.get("base_url") or None,
        temperature=float(chat.get("temperature", 0.7)),
        max_tokens=int(chat.get("max_tokens", 4096)),
    )
    # This client's constructor otherwise overwrites config.max_tokens with
    # its own 16384 default, making the WebUI output-budget setting ineffective.
    llm_client = OpenAIGenericClient(
        config=llm_config, max_tokens=llm_config.max_tokens, structured_output_mode="json_object",
    )
    from memos_system.graphiti_memory.runtime import load_model_adapters
    embedding = config.get("embedding", {})
    configured_embedder, configured_reranker, _ = load_model_adapters(config)
    embedder = RuntimeEmbedder(configured_embedder, int(embedding.get("dimension") or 1024))
    driver = Neo4jDriver(
        str(database.get("neo4j_uri") or "bolt://localhost:7687"),
        str(database.get("neo4j_user") or "neo4j"),
        str(database.get("neo4j_password") or ""),
        database=str(database.get("neo4j_database") or "neo4j"),
    )
    return Graphiti(
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=RuntimeCrossEncoder(configured_reranker),
        graph_driver=driver,
    )
