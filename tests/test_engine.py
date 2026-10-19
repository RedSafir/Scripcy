"""Comprehensive unit test suite for config management, skills loader, and LLM engine."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scrpcy.core.config import (
    CitationPreferences,
    ConfigManager,
    ScrpcyConfig,
    resolve_config_path,
)
from scrpcy.core.engine import (
    EvidenceChunk,
    EvidenceSynthesizer,
    LLMEngine,
    SynthesisResult,
    estimate_tokens,
    tokenize_query,
)
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.skills import Skill, SkillsLoader
from scrpcy.exceptions import ConfigError, EngineError, SkillNotFoundError


class MockChunkChoiceDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class MockChunkChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = MockChunkChoiceDelta(content)


class MockStreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [MockChunkChoice(content)]


class TestConfigManager:
    """Tests for configuration persistence, validation, and key resolution."""

    def test_default_config_values(self) -> None:
        config = ScrpcyConfig()
        assert config.default_model == "gemini/gemini-2.5-pro"
        assert config.citation.style == "apa"
        assert config.citation.max_evidence_chunks == 5
        assert config.citation.context_token_budget == 4000
        assert config.skills_dir == "./skills"

    def test_config_save_and_load(self, tmp_path: Path) -> None:
        config_path = tmp_path / "test_config.json"
        config = ScrpcyConfig(
            default_model="openai/gpt-4o",
            api_keys={"openai": "sk-test-12345"},
            citation=CitationPreferences(style="ieee", max_evidence_chunks=3),
        )
        saved_path = config.save(config_path)
        assert saved_path == config_path
        assert config_path.is_file()

        loaded = ScrpcyConfig.load(config_path)
        assert loaded.default_model == "openai/gpt-4o"
        assert loaded.api_keys["openai"] == "sk-test-12345"
        assert loaded.citation.style == "ieee"
        assert loaded.citation.max_evidence_chunks == 3

    def test_config_api_key_resolution_and_env_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")

        config = ScrpcyConfig(api_keys={"anthropic": "conf-anthropic-key"})

        # Resolved from stored config
        assert config.get_api_key("anthropic") == "conf-anthropic-key"
        # Resolved from environment fallback
        assert config.get_api_key("gemini") == "env-gemini-key"
        assert config.get_api_key("openai") == "env-openai-key"
        assert config.get_api_key("unknown_provider") is None

    def test_config_api_key_for_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")

        config = ScrpcyConfig()
        assert config.get_api_key_for_model("gemini/gemini-2.5-pro") == "gemini-secret"
        assert config.get_api_key_for_model("openai/gpt-4o") == "openai-secret"
        assert config.get_api_key_for_model("claude-3-5-sonnet") == "anthropic-secret"
        assert config.get_api_key_for_model("ollama/llama3") is None

    def test_config_manager_update(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "cfg.json"
        mgr = ConfigManager(config_path=cfg_file)

        mgr.update(default_model="mistral/mistral-large")
        assert mgr.config.default_model == "mistral/mistral-large"

        reloaded_mgr = ConfigManager(config_path=cfg_file)
        assert reloaded_mgr.config.default_model == "mistral/mistral-large"

    def test_config_validation_error_on_invalid_data(self, tmp_path: Path) -> None:
        bad_cfg_file = tmp_path / "bad_config.json"
        bad_cfg_file.write_text('{"citation": {"max_evidence_chunks": 999999}}', encoding="utf-8")

        with pytest.raises(ConfigError, match="Failed to read or parse"):
            ScrpcyConfig.load(bad_cfg_file)

    def test_config_environment_variable_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / "env_override_config.json"
        cfg = ScrpcyConfig(default_model="groq/llama-3.3-70b")
        cfg.save(env_file)

        monkeypatch.setenv("SCRPCY_CONFIG", str(env_file))
        assert resolve_config_path() == env_file.resolve()
        loaded = ScrpcyConfig.load()
        assert loaded.default_model == "groq/llama-3.3-70b"


class TestSkillsLoader:
    """Tests for discovering, parsing, and hot-reloading markdown skills."""

    def test_auto_init_default_skills(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)

        assert (skills_dir / "academic_writer.md").is_file()
        assert (skills_dir / "literature_synthesizer.md").is_file()

        skills = loader.list_skills()
        names = [s.name for s in skills]
        assert "academic_writer" in names
        assert "literature_synthesizer" in names

    def test_parse_skill_with_frontmatter(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=False)

        custom_file = skills_dir / "critical_reviewer.md"
        skills_dir.mkdir(parents=True, exist_ok=True)
        custom_file.write_text(
            "---\n"
            "name: critical_reviewer\n"
            "description: Rigorously reviews methodology and threats to validity.\n"
            "model: anthropic/claude-3-5-sonnet\n"
            "temperature: 0.1\n"
            "---\n"
            "You are a peer reviewer evaluating empirical validity.\n",
            encoding="utf-8",
        )

        skill = loader.get_skill("critical_reviewer")
        assert skill.name == "critical_reviewer"
        assert "validity" in skill.description
        assert skill.model == "anthropic/claude-3-5-sonnet"
        assert skill.temperature == 0.1
        assert "evaluating empirical validity" in skill.system_prompt

    def test_parse_plain_skill_without_frontmatter(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=False)
        skills_dir.mkdir(parents=True, exist_ok=True)

        plain_file = skills_dir / "plain_summarizer.md"
        plain_file.write_text(
            "Summarize the paper in three bullet points focusing on novel contributions.",
            encoding="utf-8",
        )

        skill = loader.get_skill("plain_summarizer")
        assert skill.name == "plain_summarizer"
        assert skill.model is None
        assert "three bullet points" in skill.system_prompt

    def test_dynamic_hot_reload_on_file_modification(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=False)
        skills_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skills_dir / "dynamic_skill.md"
        skill_file.write_text("Version 1 Prompt", encoding="utf-8")

        initial_skill = loader.get_skill("dynamic_skill")
        assert initial_skill.system_prompt == "Version 1 Prompt"

        # Update file on disk with slight delay to ensure distinct mtime
        time.sleep(0.05)
        skill_file.write_text("Version 2 Updated Prompt", encoding="utf-8")

        reloaded_skill = loader.get_skill("dynamic_skill")
        assert reloaded_skill.system_prompt == "Version 2 Updated Prompt"

    def test_create_or_update_skill(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=False)

        created = loader.create_or_update_skill(
            name="editor",
            system_prompt="Proofread academic text for clarity.",
            description="Grammar and clarity editor",
            temperature=0.0,
        )
        assert created.name == "editor"
        assert (skills_dir / "editor.md").is_file()

        retrieved = loader.get_skill("editor")
        assert retrieved.system_prompt == "Proofread academic text for clarity."
        assert retrieved.temperature == 0.0

    def test_skill_not_found_raises_exception(self, tmp_path: Path) -> None:
        loader = SkillsLoader(skills_dir=tmp_path / "skills", auto_init=False)
        with pytest.raises(SkillNotFoundError, match="not found"):
            loader.get_skill("non_existent_skill")

    def test_skill_format_prompt_substitution(self, tmp_path: Path) -> None:
        skill = Skill(
            name="templated",
            system_prompt="Answer query: {query} for audience: {audience}.",
            source_path=tmp_path / "dummy.md",
        )
        formatted = skill.format_prompt(query="Quantum Entanglement", audience="Undergraduates")
        assert formatted == "Answer query: Quantum Entanglement for audience: Undergraduates."


class TestEvidenceSynthesizer:
    """Tests for document chunking, lexical relevance scoring, and context budget management."""

    def test_chunk_document_single_and_multi_window(self) -> None:
        synthesizer = EvidenceSynthesizer(chunk_size_words=10, chunk_overlap_words=2)

        short_text = "Five word short sample sentence."
        chunks_short = synthesizer.chunk_document(short_text, "doc_short")
        assert len(chunks_short) == 1
        assert chunks_short[0].text == short_text
        assert chunks_short[0].token_estimate > 0

        # Multi-chunk text
        long_words = [f"word{i}" for i in range(25)]
        long_text = " ".join(long_words)
        chunks_long = synthesizer.chunk_document(long_text, "doc_long")
        assert len(chunks_long) > 1
        assert chunks_long[0].chunk_id == "doc_long_c0"
        assert chunks_long[1].chunk_id == "doc_long_c1"

    def test_tokenize_query(self) -> None:
        terms = tokenize_query("What is the attention mechanism in Transformers?")
        assert "attention" in terms
        assert "mechanism" in terms
        assert "transformers" in terms
        assert "is" not in terms
        assert "the" not in terms
        assert "in" not in terms

    def test_score_chunk_relevance(self) -> None:
        synthesizer = EvidenceSynthesizer()
        query = "residual connections"
        terms = tokenize_query(query)

        relevant_text = "Deep networks benefit from residual connections by reformulating the layers."
        unrelated_text = "The quick brown fox jumps over the lazy dog."

        score_rel = synthesizer.score_chunk(terms, query, relevant_text)
        score_unrel = synthesizer.score_chunk(terms, query, unrelated_text)

        assert score_rel > 0.0
        assert score_unrel == 0.0
        assert score_rel > score_unrel

    def test_retrieve_evidence_ranking_and_filtering(self) -> None:
        synthesizer = EvidenceSynthesizer(default_max_chunks=2)

        doc1 = "The Transformer architecture relies entirely on self-attention mechanisms without recurrence."
        doc2 = "Convolutional neural networks extract spatial feature hierarchies from visual inputs."
        doc3 = "Multi-head attention allows the model to jointly attend to information from different representation subspaces."

        selected = synthesizer.retrieve_evidence(
            query="self-attention mechanism in transformers",
            documents=[doc1, doc2, doc3],
        )

        assert len(selected) == 2
        sources = [c.text for c in selected]
        assert any("self-attention mechanisms" in s for s in sources)
        assert not any("Convolutional neural networks" in s for s in sources)

    def test_retrieve_evidence_token_budget_enforcement(self) -> None:
        synthesizer = EvidenceSynthesizer(chunk_size_words=50, default_token_budget=60)
        docs = [
            "Attention is a key mechanism. " * 10,
            "Attention models sequence dependencies. " * 10,
        ]

        selected = synthesizer.retrieve_evidence(
            query="attention",
            documents=docs,
            max_chunks=5,
            token_budget=50,
        )
        assert len(selected) == 1

    def test_format_context_prompt(self) -> None:
        synthesizer = EvidenceSynthesizer()
        chunks = [
            EvidenceChunk(
                chunk_id="c1",
                source="vaswani2017",
                text="Attention is all you need.",
                score=4.5,
                token_estimate=6,
            )
        ]
        prompt = synthesizer.format_context_prompt(chunks)
        assert "RETRIEVED RESEARCH EVIDENCE CONTEXT" in prompt
        assert "vaswani2017" in prompt
        assert "Attention is all you need." in prompt

    def test_retrieve_from_parsed_document(self) -> None:
        synthesizer = EvidenceSynthesizer()
        metadata = DocumentMetadata(
            title="Residual Learning",
            cite_key="he2016resnet",
            year=2016,
        )
        parsed_doc = ParsedDocument(
            file_path=Path("resnet.pdf"),
            metadata=metadata,
            markdown="# Deep Residual Learning\nResidual networks make training deeper models feasible.",
            body_markdown="Residual networks make training deeper models feasible.",
        )

        selected = synthesizer.retrieve_evidence(
            query="residual networks training",
            documents=[parsed_doc],
        )
        assert len(selected) >= 1
        assert selected[0].source == "he2016resnet"


class TestLLMEngine:
    """Tests for LiteLLM streaming orchestration and offline mock capabilities."""

    @pytest.mark.asyncio
    async def test_stream_query_with_mock_litellm(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)
        config = ScrpcyConfig(default_model="gemini/gemini-2.5-pro")

        recorded_kwargs: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any):
            recorded_kwargs.update(kwargs)
            tokens = ["The ", "Transformer ", "uses ", "self-attention."]
            for t in tokens:
                yield MockStreamChunk(t)

        engine = LLMEngine(
            config=config,
            skills_loader=loader,
            acompletion_fn=mock_acompletion,
        )

        stream = engine.stream_query(
            query="What mechanism does Transformer use?",
            documents=["Transformers replace recurrence with self-attention mechanism."],
            skill_name="academic_writer",
        )

        tokens_received = []
        async for token in stream:
            tokens_received.append(token)

        assert "".join(tokens_received) == "The Transformer uses self-attention."
        assert recorded_kwargs["model"] == "gemini/gemini-2.5-pro"
        assert recorded_kwargs["stream"] is True

        messages = recorded_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "academic research assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "RETRIEVED RESEARCH EVIDENCE CONTEXT" in messages[1]["content"]
        assert "What mechanism does Transformer use?" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_query_returns_synthesis_result(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)
        config = ScrpcyConfig(default_model="gemini/gemini-2.5-pro")

        async def mock_acompletion(**kwargs: Any):
            for t in ["Synthesized ", "academic ", "answer."]:
                yield MockStreamChunk(t)

        engine = LLMEngine(
            config=config,
            skills_loader=loader,
            acompletion_fn=mock_acompletion,
        )

        result = await engine.query(
            query="Explain attention",
            documents=["Attention mechanisms compute weighted representations."],
            skill_name="academic_writer",
        )

        assert isinstance(result, SynthesisResult)
        assert result.answer == "Synthesized academic answer."
        assert result.model == "gemini/gemini-2.5-pro"
        assert result.skill == "academic_writer"
        assert len(result.evidence_chunks) == 1

    @pytest.mark.asyncio
    async def test_engine_error_on_empty_query(self) -> None:
        engine = LLMEngine()
        with pytest.raises(EngineError, match="empty"):
            async for _ in engine.stream_query("   "):
                pass

    @pytest.mark.asyncio
    async def test_engine_error_on_litellm_failure(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)

        async def failing_acompletion(**kwargs: Any):
            raise ConnectionError("LLM API endpoint unavailable")
            yield  # pragma: no cover

        engine = LLMEngine(skills_loader=loader, acompletion_fn=failing_acompletion)

        with pytest.raises(EngineError, match="Streaming query failed"):
            async for _ in engine.stream_query("Valid query"):
                pass

    @pytest.mark.asyncio
    async def test_engine_with_model_and_temp_override(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        loader = SkillsLoader(skills_dir=skills_dir, auto_init=True)
        config = ScrpcyConfig(default_model="gemini/gemini-2.5-pro")

        captured_kwargs: dict[str, Any] = {}

        async def mock_acompletion(**kwargs: Any):
            captured_kwargs.update(kwargs)
            yield MockStreamChunk("OK")

        engine = LLMEngine(
            config=config,
            skills_loader=loader,
            acompletion_fn=mock_acompletion,
        )

        await engine.query(
            query="Test query",
            model="openai/gpt-4o",
            temperature=0.8,
        )

        assert captured_kwargs["model"] == "openai/gpt-4o"
        assert captured_kwargs["temperature"] == 0.8
