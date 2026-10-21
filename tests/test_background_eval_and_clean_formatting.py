import sys
from unittest.mock import MagicMock, patch
import pytest
from rich.console import Console

from scrpcy.cli_studio import CLIStudio
from scrpcy.core.tasks import TaskInfo


def test_render_system_message_clean():
    """Verify system messages are formatted with clean UI elements and no raw ### or quotes."""
    cli = CLIStudio()
    c = Console(record=True, width=120)
    cli.console = c

    # 1. Background completed
    completed_note = (
        "[PROSES LATAR BELAKANG SELESAI: scout_earth_science]\n"
        "- ID Tugas: task-1 (scout)\n"
        "- Ringkasan Hasil: Downloaded 2/2 papers.\n"
        "- Status Vault: Dokumen siap."
    )
    cli.render_system_message(completed_note)
    output = c.export_text()
    assert "Sistem:" in output
    assert "PROSES LATAR BELAKANG SELESAI: scout_earth_science" in output
    assert "###" not in output
    assert "task-1 (scout)" in output

    # 2. Historical completed with raw ###
    c.clear()
    old_raw_note = (
        "### [PROSES LATAR BELAKANG SELESAI: old_task]\n"
        "- ID Tugas: `task-99`\n"
        "- Ringkasan Hasil: Selesai tanpa kendala."
    )
    cli.render_system_message(old_raw_note)
    output2 = c.export_text()
    assert "###" not in output2
    assert "`" not in output2
    assert "PROSES LATAR BELAKANG SELESAI: old_task" in output2

    # 3. Failed task
    c.clear()
    failed_note = (
        "[PROSES LATAR BELAKANG GAGAL: download_paywall]\n"
        "- ID Tugas: task-2\n"
        "- Kendala: Paywall restricted access."
    )
    cli.render_system_message(failed_note)
    output3 = c.export_text()
    assert "PROSES LATAR BELAKANG GAGAL: download_paywall" in output3
    assert "Paywall restricted access." in output3


def test_background_task_triggers_ai_evaluation():
    """Verify that completing a background task triggers AI evaluation and records assistant briefing."""
    cli = CLIStudio()
    cli.chat_manager = MagicMock()
    cli.chat_manager.get_session.return_value = MagicMock(id="sess-1", name="testing")
    cli.chat_manager.get_messages.return_value = [
        MagicMock(role="user", content="Bandingkan umpan balik iklim.")
    ]
    cli.engine = MagicMock()

    def mock_stream(*args, **kwargs):
        async def _gen():
            yield "Berdasarkan evaluasi terhadap berkas yang baru diunduh, "
            yield "dokumen ini relevan dengan dinamika iklim bumi. [STATUS: OK]"
        return _gen()

    cli.engine.stream_query = mock_stream
    cli.config = MagicMock()
    cli.config.get_api_key_for_model.return_value = "dummy-key"
    cli.get_active_project_model = MagicMock(return_value="gemini/gemini-2.5-flash")
    cli._get_cached_stems = MagicMock(return_value=["earth_paper_1"])

    task = TaskInfo(
        id="task-10",
        name="Scout Earth Literature",
        task_type="scout",
        session_id="sess-1",
        status="completed",
        result_summary="Downloaded 1 paper into vault.",
    )

    with patch.object(cli, "_handle_suggested_actions"):
        cli._on_background_task_completed(task)

    # Check system message added
    system_calls = [
        call for call in cli.chat_manager.add_message.call_args_list
        if call[0][1] == "system"
    ]
    assert len(system_calls) == 1

    # Check AI evaluation briefing added to assistant role
    assistant_calls = [
        call for call in cli.chat_manager.add_message.call_args_list
        if call[0][1] == "assistant"
    ]
    assert len(assistant_calls) == 1
    briefing_text = assistant_calls[0][0][2]
    assert "Berdasarkan evaluasi terhadap berkas yang baru diunduh" in briefing_text
    assert "[STATUS: OK]" in briefing_text
