"""Unit test to verify skills system prompt neutrality."""

from scripcy.core.skills import DEFAULT_SCRIPCY_MANUAL_PROMPT, SCRIPCY_AGENTIC_PROTOCOL


def test_skills_system_prompt_is_territorially_neutral():
    # Verify removal of hardcoded Indonesian territorial bias
    assert "data karhutla indonesia sipongi klhk" not in DEFAULT_SCRIPCY_MANUAL_PROMPT
    assert "BPS, KLHK SiPongi, BMKG" not in DEFAULT_SCRIPCY_MANUAL_PROMPT
    assert "SiPongi KLHK, BPS data" not in DEFAULT_SCRIPCY_MANUAL_PROMPT

    # Verify universal wording
    assert "sovereign statistical agencies" in DEFAULT_SCRIPCY_MANUAL_PROMPT or "sovereign government statistics" in DEFAULT_SCRIPCY_MANUAL_PROMPT
