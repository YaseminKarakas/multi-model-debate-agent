from multi_agent_debate.models import DEFAULT_AGENTS, DebateConfig


def test_default_debate_config_is_valid():
    config = DebateConfig(agents=DEFAULT_AGENTS)

    assert config.mode == "compact"
    assert config.rounds == 2
    assert config.max_tokens == 2400
    assert config.model == "gemini/gemini-2.5-flash"
    assert len(config.agents) == 3
    assert config.agent_models == {}
