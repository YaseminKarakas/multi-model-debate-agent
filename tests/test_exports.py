from multi_agent_debate.app import result_to_markdown, validate_model_selection
from multi_agent_debate.models import DebateResult


def test_result_to_markdown_includes_main_sections():
    result = DebateResult(
        query="What should I do?",
        model="mock/debate-dev",
        final_answer="Do the useful thing.",
        responses=[],
        critiques=[],
    )

    markdown = result_to_markdown(result)

    assert "## Query" in markdown
    assert "## Final Answer" in markdown
    assert "Do the useful thing." in markdown


def test_validate_model_selection_warns_for_missing_provider_keys():
    warnings = validate_model_selection(
        "gpt-4o-mini",
        {"Analyst": "claude-3-5-sonnet-latest"},
    )

    assert "gpt-4o-mini needs its provider API key in `.env`." in warnings
    assert "claude-3-5-sonnet-latest needs its provider API key in `.env`." in warnings

