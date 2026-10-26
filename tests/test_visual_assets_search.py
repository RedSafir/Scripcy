import pytest
from pathlib import Path
from scripcy.core.workspace import WorkspaceManager
from scripcy.core.vault import VaultManager
from scripcy.core.vector_vault import ProjectVectorVault


def test_search_visual_assets_empty(tmp_path: Path):
    ws = WorkspaceManager(tmp_path)
    vm = VaultManager(ws)
    results = vm.search_visual_assets("fire policy graph", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 0


def test_search_visual_assets_with_records(tmp_path: Path):
    ws = WorkspaceManager(tmp_path)
    vm = VaultManager(ws)
    vv = vm.vector_vault

    # Manually populate images and tables for test
    with vv._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO papers (id, title, filename, path)
            VALUES ('paper_a', 'Wildfire Dynamics in Europe', 'paper_a.pdf', 'vault/papers/paper_a.pdf')
        """)
        cur.execute("""
            INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height)
            VALUES ('img_1', 'paper_a', 3, 'vault/images/fig1.png', 'Figure 1. Burned area trends across European countries', 'Grafik tren kebakaran', 800, 600)
        """)
        cur.execute("""
            INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height)
            VALUES ('img_2', 'paper_a', 5, 'vault/images/fig2.png', 'Figure 2. Seasonal temperature deviations', 'Grafik anomali suhu', 800, 600)
        """)
        cur.execute("""
            INSERT INTO tables (id, paper_id, page_num, csv_path, md_path, caption)
            VALUES ('tbl_1', 'paper_a', 6, 'vault/tables/tbl1.csv', 'vault/tables/tbl1.md', 'Table 1. Statistical summary of wildfire occurrences')
        """)
        conn.commit()

    # Search for figure related to burned area
    res_images = vm.search_visual_assets("burned area trends grafik", top_k=2)
    assert len(res_images) > 0
    top_hit = res_images[0]
    assert top_hit["id"] == "img_1"
    assert top_hit["paper_title"] == "Wildfire Dynamics in Europe"
    assert "Figure 1." in top_hit["caption"]

    # Search for table
    res_tables = vm.search_visual_assets("tabel statistical summary", top_k=2)
    assert len(res_tables) > 0
    table_hit = [r for r in res_tables if r.get("asset_type") == "table"]
    assert len(table_hit) > 0
    assert table_hit[0]["id"] == "tbl_1"
