"""Unit tests for Critic archetype-aware recency and domain-neutral gap query generation."""

from scripcy.core.critic import DevilsAdvocate
from scripcy.core.contracts import TaskContract


def test_critic_archetype_aware_recency():
    critic = DevilsAdvocate()
    # Foundational classical paper citations (e.g. ResNet 2016, Transformer 2017)
    classical_years = [2016, 2017]

    # Under theoretical foundational contract, classical references must NOT be penalized
    contract_foundational = TaskContract(
        id="c1",
        user_query="Analisis arsitektur Transformer dan residual connection",
        objective="Analyze foundational neural network architectures",
        primary_goal="Analyze foundational neural network architectures",
        archetype="THEORETICAL_FOUNDATIONAL",
    )

    is_outdated, msg = critic.check_recency(classical_years, contract=contract_foundational)
    assert is_outdated is False
    assert msg is None

    # Under DATA_FACTUAL contract, outdated references are correctly flagged
    contract_data = TaskContract(
        id="c2",
        user_query="Data produksi padi 5 tahun terakhir",
        objective="Recent agricultural statistics",
        primary_goal="Recent agricultural statistics",
        archetype="DATA_FACTUAL",
    )
    is_outdated_data, msg_data = critic.check_recency(classical_years, contract=contract_data)
    assert is_outdated_data is True
    assert msg_data is not None


def test_critic_gap_queries_are_domain_neutral():
    critic = DevilsAdvocate()
    report = critic.critique_heuristic(
        query="Self-Attention mechanism computational complexity",
        answer="The self-attention mechanism computes pairwise similarity matrices scaled by dimension square root [Vaswani2017]. However, quadratic complexity remains a challenge.",
    )
    # Ensure directed queries do not contain hardcoded Indonesian string injections
    for q in report.directed_web_queries:
        assert "data resmi statistik" not in q
        assert "laporan tahunan portal pemerintah" not in q
        assert "kementerian lembaga bps" not in q
