"""Unit tests for GraphRAGBuilder and interactive force-directed graph generation."""

from pathlib import Path
import pytest

from scripcy.core.graph_rag import GraphRAGBuilder, GraphNode, GraphEdge


class MockVectorVault:
    def list_inventory(self):
        return {
            "papers": [
                {"id": "attention_2017", "title": "Attention Is All You Need", "filename": "attention.pdf", "path": "/path/attention.pdf"},
                {"id": "web_bmkg_cuaca", "title": "Peringatan Dini Cuaca BMKG", "filename": "cuaca.web.md", "path": "/path/cuaca.web.md"},
            ],
            "tables": [],
            "images": [
                {
                    "id": "attention_fig1",
                    "paper_id": "attention_2017",
                    "caption": "Figure 1: The Transformer model architecture",
                    "file_path": "/path/fig1.png",
                    "visual_type": "SIMPLE",
                    "ai_analysis": "Diagram alir encoder-decoder",
                },
                {
                    "id": "plot_bmkg_curah",
                    "paper_id": "web_bmkg_cuaca",
                    "caption": "Grafik Curah Hujan Bulanan",
                    "file_path": "/path/curah.png",
                    "visual_type": "COMPLEX",
                    "ai_analysis": "Kurva fluktuasi curah hujan",
                }
            ]
        }

    def _get_connection(self):
        class MockCur:
            def execute(self, *args):
                pass
            def fetchall(self):
                return [
                    {"paper_id": "attention_2017", "content": "Transformer relies on Self-Attention Mechanism and Multi-Head Attention in Encoder."},
                    {"paper_id": "web_bmkg_cuaca", "content": "BMKG memantau fenomena El Nino dan curah hujan di wilayah Indonesia dengan satelit Radar."},
                ]
        class MockConn:
            def cursor(self):
                return MockCur()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return MockConn()


class MockVaultManager:
    def __init__(self):
        self.vector_vault = MockVectorVault()


def test_extract_semantic_entities():
    builder = GraphRAGBuilder()
    text = "Model Transformer and BERT achieve high accuracy on GLUE Benchmark using Self-Attention."
    entities = builder.extract_semantic_entities(text)
    assert "Transformer" in entities or "Self-Attention" in entities or "BERT" in entities


def test_build_graph_structure():
    vm = MockVaultManager()
    builder = GraphRAGBuilder(vault_manager=vm)
    data = builder.build_graph()

    nodes = data["nodes"]
    links = data["links"]

    # Verify presence of paper, web, entity, and visual nodes
    groups = {n["group"] for n in nodes}
    assert "paper" in groups
    assert "web" in groups
    assert "visual" in groups

    # Verify visual node has preview metadata
    vis_node = next((n for n in nodes if n["group"] == "visual"), None)
    assert vis_node is not None
    assert vis_node["is_visual"] is True
    assert "📊" in vis_node["label"]

    # Verify links exist
    assert len(links) > 0


def test_render_interactive_html(tmp_path: Path):
    vm = MockVaultManager()
    builder = GraphRAGBuilder(vault_manager=vm)
    data = builder.build_graph()

    out_file = tmp_path / "rag_network.html"
    rendered_path = builder.render_interactive_html(out_file, data, title="Test Graph")

    assert rendered_path.exists()
    content = rendered_path.read_text(encoding="utf-8")
    assert "d3.v7.min.js" in content
    assert "Information Panel" in content
    assert "colorbar-container" in content
    assert "attention_2017" in content
