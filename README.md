# Multi-Agent Debate

A Streamlit app that lets a user ask a question, sends it through multiple LLM agents, runs several critique and revision rounds with LangGraph, and returns one agreed enriched answer.

## Quick Start

```bash
cd multi-agent-debate
python3 -m venv venv
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

## Default Model

The app defaults to `gemini/gemini-2.5-flash` through LiteLLM. You can switch models in the sidebar if you have other provider keys configured.

## Developing With Low Rate Limits

Google AI Studio's free tier can have very low per-minute limits. Full debate mode is intentionally expensive: with 3 agents and 2 rounds it makes 16 model calls.

Use these options while developing:

- `mock/debate-dev`: zero API calls, good for UI, storage, and workflow work.
- `Compact - 1 model call`: asks one model to simulate the debate internally.
- `Batched - low-call multi-round`: keeps multiple rounds but batches all agents into one call per phase. A 2-round debate costs 4 calls.
- `Full - separate agent calls`: real multi-agent behavior, best when you have enough quota.
- `ollama_chat/llama3.1` or `ollama_chat/qwen2.5:7b`: local model options if you run Ollama.

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
ollama pull llama3.1
```

Make sure the local server is running:

```bash
ollama serve
```

Then select `ollama_chat/llama3.1` in the app. If you use a different local model, update the model string in `src/multi_agent_debate/app.py` or type it into the code as another option.

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
