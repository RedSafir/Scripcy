import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import io
from rich.console import Console

from scrpcy.core.vector_vault import (
    ProjectVectorVault,
    HybridSearchResult,
    serialize_f32,
)
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.core.vault import VaultManager


@pytest.fixture
def temp_workspace(tmp_path: Path) -> WorkspaceManager:
    ws = WorkspaceManager(root_dir=tmp_path / "test_ws")
    ws.ensure_directories()
    return ws


@pytest.fixture
def vector_vault(tmp_path: Path) -> ProjectVectorVault:
    db_p = tmp_path / "test_vault.db"
    vault_d = tmp_path / "vault"
    vault_d.mkdir(parents=True, exist_ok=True)
    return ProjectVectorVault(db_path=db_p, vault_dir=vault_d)


def test_vector_vault_schema_initialization(vector_vault: ProjectVectorVault):
    """Verify that SQLite tables and FTS/vec virtual tables initialize properly."""
    with vector_vault._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "papers" in tables
        assert "chunks" in tables
        assert "tables" in tables
        assert "images" in tables
        assert "fts_chunks" in tables
        assert "vec_chunks" in tables


def test_vector_vault_inventory_and_search(vector_vault: ProjectVectorVault):
    """Test manual insertion and inventory retrieval for papers, tables, and images."""
    with vector_vault._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO papers (id, title, filename, path) VALUES (?, ?, ?, ?)",
            ("vaswani2017", "Attention Is All You Need", "vaswani.pdf", "vault/papers/vaswani.pdf")
        )
        cur.execute(
            "INSERT INTO chunks (id, paper_id, page_num, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
            ("chunk_1", "vaswani2017", 3, 0, "Self-attention mechanism relates different positions of a sequence.")
        )
        cur.execute(
            "INSERT INTO fts_chunks (rowid, content) VALUES (?, ?)",
            (1, "Self-attention mechanism relates different positions of a sequence.")
        )
        # Mock vector embedding (dim 384)
        dummy_vec = [0.01] * 384
        cur.execute(
            "INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)",
            (1, serialize_f32(dummy_vec))
        )
        cur.execute(
            "INSERT INTO tables (paper_id, page_num, md_path, csv_path, caption) VALUES (?, ?, ?, ?, ?)",
            ("vaswani2017", 8, "vault/tables/vaswani_tab1.md", "vault/tables/vaswani_tab1.csv", "Table 1: BLEU scores")
        )
        cur.execute(
            "INSERT INTO images (paper_id, page_num, file_path, width, height, caption, ai_analysis) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("vaswani2017", 3, "vault/images/vaswani_fig1.png", 600, 800, "Figure 1: Transformer Architecture", "Visual diagram of encoder and decoder.")
        )
        conn.commit()

    inventory = vector_vault.list_inventory()
    assert len(inventory["papers"]) == 1
    assert inventory["papers"][0]["title"] == "Attention Is All You Need"
    assert len(inventory["tables"]) == 1
    assert inventory["tables"][0]["caption"] == "Table 1: BLEU scores"
    assert len(inventory["images"]) == 1
    assert "Transformer Architecture" in inventory["images"][0]["caption"]


def test_vault_manager_segregated_listings(temp_workspace: WorkspaceManager):
    """Verify VaultManager discovers segregated items in papers_dir, tables, and images."""
    vm = VaultManager(workspace=temp_workspace)
    assert temp_workspace.papers_dir.exists()
    assert temp_workspace.tables_dir.exists()
    assert temp_workspace.images_dir.exists()

    # Place a dummy paper in papers_dir
    p_file = temp_workspace.papers_dir / "quantum.pdf"
    p_file.write_bytes(b"%PDF-1.4 dummy content")

    active = vm.list_active_files()
    assert len(active) == 1
    assert active[0].filename == "quantum.pdf"

    # Verify tables and images lists run without error
    assert isinstance(vm.list_tables(), list)
    assert isinstance(vm.list_images(), list)


def test_agent_turn_checkpoint_logic():
    """Verify _check_agent_turn_checkpoint threshold passing and interactive prompting."""
    from scrpcy.cli_studio import CLIStudio

    studio = CLIStudio.__new__(CLIStudio)
    studio.console = Console(file=io.StringIO(), force_terminal=False)

    # 1. Turn not multiple of 3 -> return False
    assert studio._check_agent_turn_checkpoint(
        turn=0,
        best_grounding=0.2,
        best_critic=3,
        accumulated_input_tokens=100,
        accumulated_output_tokens=50,
    ) is False

    assert studio._check_agent_turn_checkpoint(
        turn=1,
        best_grounding=0.2,
        best_critic=3,
        accumulated_input_tokens=200,
        accumulated_output_tokens=100,
    ) is False

    # 2. Turn 2 (completed_turns == 3): Quality threshold passed -> return True (no asking)
    assert studio._check_agent_turn_checkpoint(
        turn=2,
        best_grounding=0.85,
        best_critic=8,
        accumulated_input_tokens=1000,
        accumulated_output_tokens=500,
    ) is True

    # 3. Turn 2 (completed_turns == 3): Quality NOT passed, interactive user confirms "y"
    with patch("sys.stdin.isatty", return_value=True):
        with patch.object(studio.console, "input", return_value="y"):
            should_stop = studio._check_agent_turn_checkpoint(
                turn=2,
                best_grounding=0.4,
                best_critic=5,
                accumulated_input_tokens=3000,
                accumulated_output_tokens=800,
            )
            assert should_stop is False  # continue!

    # 4. Turn 2 (completed_turns == 3): Quality NOT passed, interactive user declines "n"
    with patch("sys.stdin.isatty", return_value=True):
        with patch.object(studio.console, "input", return_value="n"):
            should_stop = studio._check_agent_turn_checkpoint(
                turn=2,
                best_grounding=0.4,
                best_critic=5,
                accumulated_input_tokens=3000,
                accumulated_output_tokens=800,
            )
            assert should_stop is True  # stop and finalize!

    # 5. Non-interactive (headless/tests): returns False before turn 15
    with patch("sys.stdin.isatty", return_value=False):
        assert studio._check_agent_turn_checkpoint(
            turn=2,
            best_grounding=0.4,
            best_critic=5,
            accumulated_input_tokens=3000,
            accumulated_output_tokens=800,
        ) is False


def test_cmd_view_table_and_image(temp_workspace: WorkspaceManager):
    """Verify /view table and /view image output formatting."""
    from scrpcy.cli_studio import CLIStudio

    studio = CLIStudio.__new__(CLIStudio)
    output = io.StringIO()
    studio.console = Console(file=output, force_terminal=False)
    studio.vault_manager = MagicMock()

    studio.vault_manager.list_tables.return_value = [
        {
            "id": 1,
            "paper_id": "vaswani2017",
            "page_num": 5,
            "md_path": str(temp_workspace.tables_dir / "tab1.md"),
            "caption": "Ablation Results",
        }
    ]
    tab_p = temp_workspace.tables_dir / "tab1.md"
    tab_p.write_text("| Model | BLEU |\n|---|---|\n| Base | 27.3 |", encoding="utf-8")

    studio._cmd_view_table("1")
    text = output.getvalue()
    assert "Tabel #1" in text
    assert "Ablation Results" in text
    assert "Base" in text

    # Test view image
    img_output = io.StringIO()
    studio.console = Console(file=img_output, force_terminal=False)
    studio.vault_manager.list_images.return_value = [
        {
            "id": 1,
            "paper_id": "vaswani2017",
            "page_num": 3,
            "file_path": str(temp_workspace.images_dir / "fig1.png"),
            "width": 800,
            "height": 600,
            "caption": "Architecture Overview",
            "ai_analysis": "The image represents multi-head attention layout.",
        }
    ]
    studio._cmd_view_image("1")
    img_text = img_output.getvalue()
    assert "Gambar #1" in img_text
    assert "800 x 600" in img_text
    assert "multi-head attention" in img_text
