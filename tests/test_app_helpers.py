from multi_agent_debate.app import estimate_calls


def test_estimate_calls_for_modes():
    assert estimate_calls("compact", agent_count=3, rounds=2) == 1
    assert estimate_calls("batched", agent_count=3, rounds=2) == 4
    assert estimate_calls("full", agent_count=3, rounds=2) == 16

