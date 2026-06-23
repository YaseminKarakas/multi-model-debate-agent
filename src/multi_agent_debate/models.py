from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentProfile(BaseModel):
    name: str
    role: str
    stance: str


class DebateConfig(BaseModel):
    model: str = "gemini/gemini-2.5-flash"
    mode: Literal["compact", "batched", "full"] = "compact"
    answer_style: Literal["concise", "debug"] = "debug"
    rounds: int = Field(default=2, ge=1, le=5)
    temperature: float = Field(default=0.4, ge=0.0, le=1.5)
    max_tokens: int = Field(default=2400, ge=256, le=8000)
    agents: list[AgentProfile]
    agent_models: dict[str, str] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_name: str
    round_number: int
    content: str


class Critique(BaseModel):
    agent_name: str
    round_number: int
    content: str


class DebateResult(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    model: str
    final_answer: str
    responses: list[AgentResponse]
    critiques: list[Critique]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DebateState(TypedDict):
    query: str
    config: DebateConfig
    responses: list[AgentResponse]
    critiques: list[Critique]
    current_round: int
    final_answer: str


DEFAULT_AGENTS = [
    AgentProfile(
        name="Analyst",
        role="Breaks down the problem and identifies structure, assumptions, and missing context.",
        stance="Be precise, practical, and explicit about uncertainty.",
    ),
    AgentProfile(
        name="Skeptic",
        role="Looks for weak reasoning, hidden risks, and unsupported claims.",
        stance="Challenge the answer constructively and demand evidence.",
    ),
    AgentProfile(
        name="Synthesizer",
        role="Connects the strongest points into a clear, useful final response.",
        stance="Favor clarity, balance, and actionable next steps.",
    ),
]
