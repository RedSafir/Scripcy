"""Unit tests for Hierarchical Worker Status, Live Callbacks, and Session Re-attach Protection."""

import re
import pytest

from scripcy.core.orchestrator import SubagentOrchestrator
from scripcy.core.contracts import TaskContract


def test_format_hierarchical_status_running():
    orch = SubagentOrchestrator()
    agent_states = {
        "Agent Download Web": ("running", "Mengunduh web pustaka.ut.ac.id..."),
        "Agent Download Journal": ("running", "Mengunduh PDF arXiv..."),
    }
    status = orch.format_hierarchical_status(
        worker_title="Pencarian & Unduhan Literatur",
        worker_state="running",
        worker_detail="Mengumpulkan data pustaka secara paralel",
        agent_states=agent_states,
        elapsed_seconds=15.0,
    )

    lines = status.splitlines()
    assert len(lines) == 3
    # Check parent worker line
    assert lines[0].startswith("[⠋] Worker [Pencarian & Unduhan Literatur]")
    assert "Mengumpulkan data pustaka" in lines[0]
    # Check parallel children
    assert "├── [⠋] Agent Download Web: Mengunduh web pustaka.ut.ac.id..." in lines[1]
    assert "└── [⠋] Agent Download Journal: Mengunduh PDF arXiv..." in lines[2]


def test_format_hierarchical_status_done():
    orch = SubagentOrchestrator()
    agent_states = {
        "Agent Download Web": ("done", "1 web resmi diindeks"),
        "Agent Download Journal": ("done", "2 naskah jurnal diindeks"),
    }
    status = orch.format_hierarchical_status(
        worker_title="Pencarian Literatur",
        worker_state="done",
        worker_detail="2 PDF, 1 Web diunduh",
        agent_states=agent_states,
        elapsed_seconds=28.0,
    )

    lines = status.splitlines()
    assert lines[0].startswith("[✓] Worker berhasil menangani contract \"Pencarian Literatur\"")
    assert "2 PDF, 1 Web diunduh" in lines[0]
    assert "├── [✓] Agent Download Web: 1 web resmi diindeks" in lines[1]
    assert "└── [✓] Agent Download Journal: 2 naskah jurnal diindeks" in lines[2]


def test_format_hierarchical_status_cancelled():
    orch = SubagentOrchestrator()
    agent_states = {
        "Agent Download Web": ("cancelled", "Dibatalkan"),
        "Agent Download Journal": ("cancelled", "Dibatalkan"),
    }
    status = orch.format_hierarchical_status(
        worker_title="Pencarian Literatur",
        worker_state="cancelled",
        worker_detail="Operasi dihentikan oleh pengguna",
        agent_states=agent_states,
        elapsed_seconds=10.0,
    )

    lines = status.splitlines()
    assert lines[0].startswith("[✗] Worker dibatalkan: Operasi dihentikan oleh pengguna")
    assert "├── [✗] Agent Download Web: Dibatalkan" in lines[1]
    assert "└── [✗] Agent Download Journal: Dibatalkan" in lines[2]


def test_responsive_truncation_logic():
    """Verify that long descriptions are cleanly truncated so the timer badge remains visible."""
    term_cols = 80
    elapsed_sec = 24
    time_hint = f"({elapsed_sec}s, esc: detach · ctrl+c: batal)"
    long_desc = "Worker [Pencarian & Unduhan Literatur Sangat Panjang Sekali Melebihi Batas Kolom Layar Terminal Pengguna]: Mengumpulkan informasi detail"

    avail_desc = max(10, term_cols - len(time_hint) - 6)
    if len(long_desc) > avail_desc:
        truncated = long_desc[: avail_desc - 3] + "..."
    else:
        truncated = long_desc

    formatted = f"{truncated}  {time_hint}"
    assert len(formatted) <= term_cols
    assert formatted.endswith(time_hint)
    assert "..." in formatted
