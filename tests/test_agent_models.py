from multi_agent_debate.graph import DebateGraph
from multi_agent_debate.models import DEFAULT_AGENTS, DebateConfig


def test_full_mode_uses_agent_specific_models():
    config = DebateConfig(
        model="mock/debate-dev",
        mode="full",
        agents=DEFAULT_AGENTS,
        agent_models={
            "Analyst": "mock/debate-dev",
            "Skeptic": "ollama_chat/llama3.1",
        },
    )

    graph = DebateGraph(config)

    assert graph._agent_llm("Analyst").model == "mock/debate-dev"
    assert graph._agent_llm("Skeptic").model == "ollama_chat/llama3.1"
    assert graph._agent_llm("Synthesizer").model == "mock/debate-dev"

