import pytest
from pathlib import Path
from PIL import Image
from rich.console import Console

from scrpcy.ui.terminal_image import (
    init_virtual_terminal,
    render_image_to_ansi,
    render_visual_panel,
)


def test_init_virtual_terminal():
    # Should run cleanly on all platforms without raising
    init_virtual_terminal()


def test_render_image_to_ansi(tmp_path: Path):
    # Create a small RGB test image
    img_path = tmp_path / "test_fig.png"
    img = Image.new("RGB", (40, 30), color=(73, 109, 137))
    img.save(img_path)

    ansi_output = render_image_to_ansi(img_path, max_cols=20, max_rows=10)
    assert isinstance(ansi_output, str)
    assert len(ansi_output) > 0
    # Truecolor escape sequence should be present
    assert "\x1b[38;2;" in ansi_output
    # Half-block character should be present
    assert "▀" in ansi_output


def test_render_visual_panel_image(tmp_path: Path):
    img_path = tmp_path / "academic_fig.png"
    img = Image.new("RGB", (60, 40), color=(100, 150, 200))
    img.save(img_path)

    v_info = {
        "asset_type": "image",
        "index": 1,
        "paper_id": "Smith2024",
        "paper_title": "Analysis of Forest Fire Dynamics",
        "page_num": 4,
        "file_path": str(img_path),
        "caption": "Figure 4. Comparative burned area metrics across regions.",
        "ai_analysis": "Menunjukkan tren penurunan area terbakar pada zona konservasi.",
    }

    console = Console(record=True, width=100)
    panel = render_visual_panel(v_info, console=console)
    assert panel is not None

    console.print(panel)
    output = console.export_text()
    assert "Rujukan Visual Ilmiah #1" in output
    assert "Analysis of Forest Fire Dynamics" in output
    assert "Halaman 4" in output
    assert "Figure 4." in output


def test_render_visual_panel_table(tmp_path: Path):
    md_path = tmp_path / "tbl_sample.md"
    md_path.write_text("| Model | F1 |\n|---|---|\n| Baseline | 0.85 |\n| Scrpcy | 0.94 |", encoding="utf-8")

    v_info = {
        "asset_type": "table",
        "index": 2,
        "paper_id": "Doe2023",
        "paper_title": "Deep Learning Benchmarks",
        "page_num": 7,
        "md_path": str(md_path),
        "caption": "Table 1. Benchmark performance comparison.",
    }

    console = Console(record=True, width=100)
    panel = render_visual_panel(v_info, console=console)
    assert panel is not None

    console.print(panel)
    output = console.export_text()
    assert "Rujukan Visual Ilmiah #2" in output
    assert "Deep Learning Benchmarks" in output
    assert "Halaman 7" in output
    assert "Baseline" in output
