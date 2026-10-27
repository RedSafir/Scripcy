"""Unit tests for Direct Multimodal Injection in LLMEngine."""

import base64
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import EvidenceChunk, LLMEngine


class MockVaultConnection:
    def __init__(self, rows: dict[str, str]):
        self.rows = rows

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.last_vid = params[0]
        return self

    def fetchone(self):
        fp = self.rows.get(self.last_vid)
        return {"file_path": fp} if fp else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockVectorVault:
    def __init__(self, rows: dict[str, str]):
        self.rows = rows

    def _get_connection(self):
        return MockVaultConnection(self.rows)


@pytest.mark.asyncio
async def test_stream_query_injects_multimodal_complex_image(tmp_path: Path):
    # Create sample complex image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "plot_fig1.png"
    img.save(img_path)

    vault = MockVectorVault({"plot_fig1": str(img_path)})

    captured_kwargs = {}

    async def mock_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        # Yield fake stream token
        class FakeDelta:
            content = "Hasil analisis visual..."

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        yield FakeChunk()

    cfg = ScripcyConfig()
    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)

    # Monkeypatch retrieve_evidence to return a chunk with [[VISUAL:plot_fig1:COMPLEX]]
    complex_chunk = EvidenceChunk(
        chunk_id="chunk_1",
        source="paper_test",
        page_num=1,
        text="Berikut adalah grafik perbandingan: [[VISUAL:plot_fig1:COMPLEX]]",
        score=0.9,
    )
    engine.evidence_synthesizer.retrieve_evidence = lambda **kw: [complex_chunk]

    tokens = []
    async for tok in engine.stream_query(
        query="Jelaskan grafik plot_fig1",
        documents=[],
        vector_vault=vault,
    ):
        tokens.append(tok)

    assert "".join(tokens) == "Hasil analisis visual..."

    # Verify that captured messages contained the direct multimodal image_url
    messages = captured_kwargs.get("messages", [])
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    assert user_msg is not None

    content = user_msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64," in content[1]["image_url"]["url"]


@pytest.mark.asyncio
async def test_stream_query_keeps_text_for_simple_visual(tmp_path: Path):
    vault = MockVectorVault({})

    captured_kwargs = {}

    async def mock_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        class FakeDelta:
            content = "Hasil analisis bagan alir..."

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        yield FakeChunk()

    cfg = ScripcyConfig()
    engine = LLMEngine(config=cfg, acompletion_fn=mock_acompletion)

    # SIMPLE visual uses [[VISUAL:flowchart_1]] without :COMPLEX suffix
    simple_chunk = EvidenceChunk(
        chunk_id="chunk_2",
        source="paper_test",
        page_num=2,
        text="Berikut adalah alur arsitektur: [[VISUAL:flowchart_1]] [Deskripsi: Alur kotak encoder decoder]",
        score=0.85,
    )
    engine.evidence_synthesizer.retrieve_evidence = lambda **kw: [simple_chunk]

    tokens = []
    async for tok in engine.stream_query(
        query="Jelaskan alur flowchart",
        documents=[],
        vector_vault=vault,
    ):
        tokens.append(tok)

    assert "".join(tokens) == "Hasil analisis bagan alir..."

    # Verify that content is a clean string (text-only, zero pixel overhead)
    messages = captured_kwargs.get("messages", [])
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    assert user_msg is not None
    assert isinstance(user_msg["content"], str)
