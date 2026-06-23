from multi_agent_debate.llm import LLMClient


def test_detects_length_based_truncation():
    assert LLMClient._needs_continuation("partial answer", "length")
    assert LLMClient._needs_continuation("## Final Answer\nStarts but cuts", "")
    assert not LLMClient._needs_continuation(
        "## Final Answer\nDone.\n## Accepted Critiques\nC1.\n## Change Log\nChanged.",
        "stop",
    )

