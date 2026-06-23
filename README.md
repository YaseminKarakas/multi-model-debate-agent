# Multi-Agent Debate

A Streamlit app that lets a user ask a question, sends it through multiple LLM agents, runs several critique and revision rounds with LangGraph, and returns one agreed enriched answer.

## Features

- Multi-agent debate with Analyst, Skeptic, and Synthesizer roles.
- Three execution modes: compact, batched, and full separate-agent debate.
- Optional per-agent model selection in full mode.
- LiteLLM support for Gemini, Gemma, Ollama, OpenAI, and Anthropic model strings.
- Debug and production answer styles.
- Structured critique/revision prompts with critique IDs and change tracking.
- SQLite debate history with clickable restore.
- Markdown and JSON exports.

## Quick Start

```bash
cd multi-agent-debate
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

Add your Gemini key to `.env`:

```env
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Run the app:

```bash
streamlit run src/multi_agent_debate/app.py
```

If you prefer a requirements file, `pip install -r requirements.txt` also installs the runtime packages. The editable install is nicer during development because local package imports work automatically.

## Usage

1. Choose a base model in the sidebar.
2. Choose an execution mode:
   - `Compact - 1 model call`: cheapest, best for quick checks.
   - `Batched - low-call multi-round`: multiple rounds with fewer calls.
   - `Full - separate agent calls`: most faithful debate behavior.
3. Optionally enable per-agent model selection in full mode.
4. Pick `Debug - detailed` while inspecting reasoning, or `Production - concise` for shorter final answers.
5. Run the debate, review the final answer and transcript, then export Markdown or JSON if useful.

## Default Model

The app defaults to `gemini/gemini-2.5-flash` through LiteLLM. You can switch models in the sidebar if you have other provider keys configured.

## Developing With Low Rate Limits

Google AI Studio's free tier can have very low per-minute limits. Full debate mode is intentionally expensive: with 3 agents and 2 rounds it makes 16 model calls.

Use these options while developing:

- `mock/debate-dev`: zero API calls, good for UI, storage, and workflow work.
- `Compact - 1 model call`: asks one model to simulate the debate internally.
- `Batched - low-call multi-round`: keeps multiple rounds but batches all agents into one call per phase. A 2-round debate costs 4 calls.
- `Full - separate agent calls`: real multi-agent behavior, best when you have enough quota.
- `ollama_chat/llama3.2:1b`, `ollama_chat/qwen2.5:1.5b`, `ollama_chat/gemma2:2b`, `ollama_chat/llama3.1`, or `ollama_chat/qwen2.5:7b`: local model options if you run Ollama.

For Gemini free-tier testing, use compact mode first. Full mode with 2 agents and 1 round still costs 5 calls, which can exhaust a 5 RPM limit in one click.

## Per-Agent Models

In `Full - separate agent calls` mode, enable `Use different models per agent` to assign a model to each debating agent. The main `Model` dropdown is still used for the final moderator synthesis.

Compact and batched modes use one shared model call per phase, so they intentionally use the main model only.

The full debate prompts now force each agent to produce structured sections such as changed points, critique IDs addressed, new points added, and remaining uncertainty. This is meant to reduce repetition and make small/local models behave less lazily.

## Google AI Studio Models

The model list includes text-oriented AI Studio candidates verified with `ModelService.ListModels`, including `gemini/gemini-3.1-flash-lite`, `gemini/gemini-3.5-flash`, `gemini/gemma-4-26b-a4b-it`, and `gemini/gemma-4-31b-it`. If a model fails later, copy the exact API model ID from AI Studio and update `MODEL_OPTIONS` in `src/multi_agent_debate/app.py`.

Live API models are not included because this app uses text chat completions through LiteLLM. Supporting Live API models should be a separate streaming/client integration.

## Ollama Setup

Install Ollama from <https://ollama.com>, then pull a model:

```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:1.5b
ollama pull gemma2:2b
ollama pull llama3.1
```

Make sure the local server is running:

```bash
ollama serve
```

Then select `ollama_chat/llama3.1` in the app. If you use a different local model, update the model string in `src/multi_agent_debate/app.py` or type it into the code as another option.

## Testing

```bash
pytest
```

The test suite covers configuration defaults, compact/batched mock runs, per-agent model routing, export helpers, and truncation detection.

## Project Layout

```text
src/multi_agent_debate/
  app.py       Streamlit user interface
  config.py    Environment and app defaults
  graph.py     LangGraph debate workflow
  llm.py       LiteLLM client wrapper
  models.py    Pydantic data models
  prompts.py   Debate prompt builders
  storage.py   SQLite conversation history
```

## Notes

- The checked-in `.env.example` is safe to commit.
- The real `.env` is ignored by git.
- Debate transcripts are stored locally in `debate_history.sqlite3`.
