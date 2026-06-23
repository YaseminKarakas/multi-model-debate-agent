from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from multi_agent_debate.config import get_settings
from multi_agent_debate.graph import DebateGraph
from multi_agent_debate.llm import ProviderSetupError, TemporaryLLMError
from multi_agent_debate.models import DEFAULT_AGENTS, AgentProfile, DebateConfig, DebateResult
from multi_agent_debate.storage import DebateStore


settings = get_settings()


MODEL_OPTIONS = [
    "mock/debate-dev",
    settings.default_model,
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.0-flash",
    "gemini/gemma-4-26b-a4b-it",
    "gemini/gemma-4-31b-it",
    "ollama_chat/llama3.2:1b",
    "ollama_chat/qwen2.5:1.5b",
    "ollama_chat/gemma2:2b",
    "ollama_chat/llama3.1",
    "ollama_chat/qwen2.5:7b",
    "gpt-4o-mini",
    "claude-3-5-sonnet-latest",
]


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def build_agent_profiles() -> list[AgentProfile]:
    agents = []
    for agent in DEFAULT_AGENTS:
        enabled = st.sidebar.checkbox(agent.name, value=True, key=f"agent_{agent.name}")
        if enabled:
            agents.append(agent)
    return agents


def build_agent_models(agents: list[AgentProfile], default_model: str, mode: str) -> dict[str, str]:
    if mode != "full":
        st.sidebar.caption("Per-agent models are available in full mode.")
        return {}

    use_agent_models = st.sidebar.toggle("Use different models per agent", value=False)
    if not use_agent_models:
        return {}

    agent_models = {}
    with st.sidebar.expander("Agent models", expanded=True):
        for agent in agents:
            agent_models[agent.name] = st.selectbox(
                f"{agent.name} model",
                MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0,
                key=f"model_{agent.name}",
            )
    return agent_models


def render_history(store: DebateStore) -> DebateResult | None:
    with st.sidebar.expander("Recent debates", expanded=False):
        try:
            rows = run_async(store.recent())
        except Exception:
            rows = []
        if not rows:
            st.caption("No saved debates yet.")
            return None
        for row in rows:
            label = f"{row['created_at'][:16]} | {row['query'][:70]}"
            if st.button(label, key=f"history_{row['session_id']}"):
                return run_async(store.get(row["session_id"]))
    return None


def estimate_calls(mode: str, agent_count: int, rounds: int) -> int:
    if mode == "compact":
        return 1
    if mode == "batched":
        return rounds + 2
    return agent_count * (1 + 2 * rounds) + 1


def validate_model_selection(model: str, agent_models: dict[str, str]) -> list[str]:
    selected_models = [model, *agent_models.values()]
    warnings = []
    for selected_model in selected_models:
        if selected_model.startswith(("gpt-", "claude-")):
            warnings.append(f"{selected_model} needs its provider API key in `.env`.")
    return sorted(set(warnings))


def result_to_markdown(result: DebateResult) -> str:
    answer_sections = "\n\n".join(
        f"### {response.agent_name} - round {response.round_number}\n\n{response.content}"
        for response in result.responses
    )
    critique_sections = "\n\n".join(
        f"### {critique.agent_name} - round {critique.round_number}\n\n{critique.content}"
        for critique in result.critiques
    )
    return (
        f"# Multi-Agent Debate\n\n"
        f"**Created:** {result.created_at.isoformat()}\n\n"
        f"**Model:** {result.model}\n\n"
        f"## Query\n\n{result.query}\n\n"
        f"## Final Answer\n\n{result.final_answer}\n\n"
        f"## Answers\n\n{answer_sections}\n\n"
        f"## Critiques\n\n{critique_sections}\n"
    )


def render_result(result: DebateResult) -> None:
    st.subheader("Agreed enriched answer")
    st.markdown(result.final_answer)

    export_col_1, export_col_2 = st.columns(2)
    export_col_1.download_button(
        "Download Markdown",
        data=result_to_markdown(result),
        file_name=f"debate-{result.session_id}.md",
        mime="text/markdown",
    )
    export_col_2.download_button(
        "Download JSON",
        data=json.dumps(result.model_dump(mode="json"), indent=2),
        file_name=f"debate-{result.session_id}.json",
        mime="application/json",
    )

    st.subheader("Transcript")
    tab_answers, tab_critiques = st.tabs(["Answers", "Critiques"])
    with tab_answers:
        for response in result.responses:
            with st.expander(f"{response.agent_name} - round {response.round_number}"):
                st.markdown(response.content)
    with tab_critiques:
        for critique in result.critiques:
            with st.expander(f"{critique.agent_name} - round {critique.round_number}"):
                st.markdown(critique.content)


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Debate", page_icon="⚖️", layout="wide")
    st.title("Multi-Agent Debate")
    st.caption("Ask once. Let agents challenge, revise, and converge on a stronger answer.")

    store = DebateStore(Path(settings.database_path))
    run_async(store.initialize())

    st.sidebar.header("Debate Settings")
    model = st.sidebar.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(settings.default_model),
    )
    mode_label = st.sidebar.radio(
        "Execution mode",
        [
            "Compact - 1 model call",
            "Batched - low-call multi-round",
            "Full - separate agent calls",
        ],
        index=1,
    )
    if mode_label.startswith("Compact"):
        mode = "compact"
    elif mode_label.startswith("Batched"):
        mode = "batched"
    else:
        mode = "full"
    rounds = st.sidebar.slider("Debate rounds", min_value=1, max_value=5, value=2)
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.5, value=0.4, step=0.1
    )
    max_tokens = st.sidebar.slider("Max tokens per call", 256, 8000, 2400, step=256)
    answer_style_label = st.sidebar.radio(
        "Answer style",
        ["Debug - detailed", "Production - concise"],
        index=0,
    )
    answer_style = "concise" if answer_style_label.startswith("Production") else "debug"

    st.sidebar.subheader("Agents")
    agents = build_agent_profiles()
    agent_models = build_agent_models(agents, model, mode)
    call_count = estimate_calls(mode, len(agents), rounds)
    st.sidebar.caption(f"Estimated model calls per debate: {call_count}")
    restored_result = render_history(store)
    if restored_result:
        st.session_state["selected_result"] = restored_result

    if not settings.has_gemini_key and model.startswith("gemini/"):
        st.warning("Add your Gemini key to `.env` before running Gemini debates.")
    if model.startswith("gemini/gemma-4"):
        st.info(
            "Gemma API model IDs can vary by Google release/channel. These IDs were verified "
            "with ModelService.ListModels; if Google changes them, copy the exact model ID "
            "from AI Studio and add it to `MODEL_OPTIONS`."
        )
    if model.startswith("ollama_chat/"):
        st.info(
            "Ollama models require the Ollama app/server running locally and the selected "
            "model pulled, for example `ollama pull llama3.1`."
        )
    for warning in validate_model_selection(model, agent_models):
        st.warning(warning)

    query = st.text_area(
        "Your query",
        placeholder="Example: What is the best launch strategy for a niche productivity app?",
        height=160,
    )

    can_run = bool(query.strip()) and len(agents) >= 2
    if len(agents) < 2:
        st.error("Select at least two agents for a debate.")

    if st.button("Run debate", type="primary", disabled=not can_run):
        config = DebateConfig(
            model=model,
            mode=mode,
            rounds=rounds,
            answer_style=answer_style,
            temperature=temperature,
            max_tokens=max_tokens,
            agents=agents,
            agent_models=agent_models,
        )
        graph = DebateGraph(config)
        with st.status("Debating...", expanded=True) as status:
            if mode == "compact":
                st.write("Running a compact one-call debate to stay within low RPM limits.")
            elif mode == "batched":
                st.write("Running a batched multi-round debate with one call per phase.")
            else:
                st.write("Generating initial answers, critiques, revisions, and synthesis.")
            st.write(f"Estimated calls: {call_count}")
            st.write(f"Final answer style: {answer_style}")
            try:
                result = run_async(graph.run(query.strip()))
                run_async(store.save(result))
                st.session_state["selected_result"] = result
                status.update(label="Debate complete", state="complete")
            except TemporaryLLMError as exc:
                status.update(label="Debate paused by provider capacity", state="error")
                st.error(str(exc))
                st.info(
                    "For Gemini overloads, try `gemini/gemini-2.5-flash-lite` or "
                    "`gemini/gemini-2.0-flash`, or lower the debate to 2 agents and 1 round."
                )
                return
            except ProviderSetupError as exc:
                status.update(label="Provider setup needed", state="error")
                st.error(str(exc))
                st.code("ollama serve\nollama pull llama3.1", language="bash")
                return
            except Exception as exc:
                status.update(label="Debate failed", state="error")
                st.exception(exc)
                return

    selected_result = st.session_state.get("selected_result")
    if selected_result:
        render_result(selected_result)


if __name__ == "__main__":
    main()
