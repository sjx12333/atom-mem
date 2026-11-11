from __future__ import annotations

from pathlib import Path

from memos_system.graphiti_memory.graph_query import GraphQueryEngine
from memos_system.graphiti_memory.topology import SharedTopology
from memos_system.graphiti_memory.warm_store import WarmStore

ROLE = "role-test"


def _graph(tmp_path: Path):
    topology = SharedTopology(tmp_path / "topology.json")
    for node_id, name in (
        ("n_alice", "Alice"),
        ("n_bob", "Bob"),
        ("n_carol", "Carol"),
        ("n_rome", "Rome"),
    ):
        topology.upsert_node(node_id, name, kind="node")
        topology.grant_node(ROLE, node_id)
    topology.upsert_relation("n_alice", "n_bob", name="KNOWS")
    topology.upsert_relation("n_bob", "n_carol", name="KNOWS")
    topology.upsert_relation("n_carol", "n_rome", name="VISITED")
    topology.create_fact_node(ROLE, "n_alice", "Alice likes tea")
    topology.create_fact_node(ROLE, "n_carol", "Carol visited Rome")
    return topology


def test_ppr_is_a_tool_operator_and_reaches_two_hops(tmp_path: Path):
    engine = GraphQueryEngine(_graph(tmp_path), ROLE)
    result = engine.execute(
        query="Alice 通过朋友和谁有间接关联？",
        operator="ppr",
        anchors=["Alice"],
        top_k=5,
    )
    assert result["ok"] is True
    assert result["operator"] == "ppr"
    names = [item["name"] for item in result["evidence"]["ppr_nodes"]]
    assert "Carol" in names
    assert result["diagnostics"]["ppr_only_on_tool_call"] is True


def test_automatic_operator_does_not_use_ppr(tmp_path: Path):
    engine = GraphQueryEngine(_graph(tmp_path), ROLE)
    result = engine.execute(query="Alice likes what?", anchors=["Alice"], top_k=5)
    assert result["operator"] != "ppr"
    assert result["evidence"]["ppr_nodes"] == []


def test_warm_layer_raw_windows_and_rag_shape(tmp_path: Path):
    def embed(text: str):
        return [float(len(text) % 7 + 1), float(sum(map(ord, text)) % 11 + 1), 1.0]

    class Reranker:
        def rerank_many(self, query, documents):
            return [1.0 if "大橘" in doc else 0.1 for doc in documents]

    store = WarmStore(tmp_path, "role-warm", embedder=embed, reranker=Reranker())
    store.append(
        [
            {
                "event_id": "e1",
                "batch_id": "b1",
                "conversation_id": "c1",
                "round_no": 1,
                "role": "user",
                "content": "我养了一只橘猫叫大橘",
                "created_at": "2026-11-01T10:00:00",
            },
            {
                "event_id": "e2",
                "batch_id": "b1",
                "conversation_id": "c1",
                "round_no": 2,
                "role": "assistant",
                "content": "大橘很可爱",
                "created_at": "2026-11-01T10:01:00",
            },
        ]
    )
    diagnostics = {}
    hits = store.search(
        "大橘",
        top_k=2,
        gate=0.0,
        raw_rag=True,
        rerank=True,
        diagnostics=diagnostics,
    )
    assert diagnostics["mode"] == "raw_rag_windows"
    assert diagnostics["chunk_tokens"] == 256
    assert hits
    assert hits[0]["id"].startswith("warmwin:")
    assert "window_entries" in hits[0]
