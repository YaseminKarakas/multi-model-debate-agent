import asyncio

from multi_agent_debate.graph import DebateGraph
from multi_agent_debate.models import DEFAULT_AGENTS, DebateConfig


def test_compact_mock_debate_runs_without_provider_calls():
    config = DebateConfig(model="mock/debate-dev", mode="compact", agents=DEFAULT_AGENTS)
    result = asyncio.run(DebateGraph(config).run("What is 2 + 4?"))

    assert result.final_answer
    assert "Mock debate response" in result.final_answer
    assert len(result.responses) == len(DEFAULT_AGENTS)
    assert result.critiques[0].agent_name == "Moderator"


def test_batched_mock_debate_runs_multiple_rounds_with_few_calls():
    config = DebateConfig(
        model="mock/debate-dev",
        mode="batched",
        rounds=2,
        agents=DEFAULT_AGENTS,
    )
    result = asyncio.run(DebateGraph(config).run("What is 2 + 4?"))

    assert result.final_answer
    assert len(result.responses) == 3
    assert len(result.critiques) == 2
    assert result.responses[0].round_number == 0
    assert result.responses[-1].round_number == 2
