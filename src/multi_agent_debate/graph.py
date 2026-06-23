from __future__ import annotations

import asyncio

from langgraph.graph import END, START, StateGraph

from .llm import LLMClient
from .models import AgentResponse, Critique, DebateConfig, DebateResult, DebateState
from .prompts import (
    batched_initial_prompt,
    batched_round_prompt,
    batched_synthesis_prompt,
    compact_debate_prompt,
    critique_prompt,
    initial_answer_prompt,
    revision_prompt,
    synthesis_prompt,
)


class DebateGraph:
    def __init__(self, config: DebateConfig) -> None:
        self.config = config
        self.llm = LLMClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.agent_llms = {
            agent.name: LLMClient(
                model=config.agent_models.get(agent.name, config.model),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            for agent in config.agents
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(DebateState)
        workflow.add_node("initial_answers", self._initial_answers)
        workflow.add_node("critique_round", self._critique_round)
        workflow.add_node("revision_round", self._revision_round)
        workflow.add_node("synthesize", self._synthesize)

        workflow.add_edge(START, "initial_answers")
        workflow.add_edge("initial_answers", "critique_round")
        workflow.add_edge("critique_round", "revision_round")
        workflow.add_conditional_edges(
            "revision_round",
            self._route_after_revision,
            {"continue": "critique_round", "finish": "synthesize"},
        )
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    async def run(self, query: str) -> DebateResult:
        if self.config.mode == "compact":
            return await self._run_compact(query)
        if self.config.mode == "batched":
            return await self._run_batched(query)

        initial_state: DebateState = {
            "query": query,
            "config": self.config,
            "responses": [],
            "critiques": [],
            "current_round": 0,
            "final_answer": "",
        }
        state = await self.graph.ainvoke(initial_state)
        return DebateResult(
            query=query,
            model=self.config.model,
            final_answer=state["final_answer"],
            responses=state["responses"],
            critiques=state["critiques"],
        )

    async def _run_compact(self, query: str) -> DebateResult:
        final_answer = await self.llm.complete(
            compact_debate_prompt(query, self.config.agents, self.config.rounds)
        )
        responses = [
            AgentResponse(
                agent_name=agent.name,
                round_number=0,
                content=(
                    f"{agent.name} participated in compact mode. The debate was simulated "
                    "inside one model call to stay within low RPM limits."
                ),
            )
            for agent in self.config.agents
        ]
        critiques = [
            Critique(
                agent_name="Moderator",
                round_number=1,
                content=(
                    "Compact mode compresses critique, revision, and synthesis into one call. "
                    "Use full mode when you have enough quota for separate agent calls."
                ),
            )
        ]
        return DebateResult(
            query=query,
            model=self.config.model,
            final_answer=final_answer,
            responses=responses,
            critiques=critiques,
        )

    async def _run_batched(self, query: str) -> DebateResult:
        responses: list[AgentResponse] = []
        critiques: list[Critique] = []

        initial = await self.llm.complete(batched_initial_prompt(query, self.config.agents))
        responses.append(
            AgentResponse(agent_name="Batched agents", round_number=0, content=initial)
        )

        transcript = f"## Initial Positions\n{initial}"
        for round_number in range(1, self.config.rounds + 1):
            round_output = await self.llm.complete(
                batched_round_prompt(query, self.config.agents, transcript, round_number)
            )
            critiques.append(
                Critique(
                    agent_name="Batched agents",
                    round_number=round_number,
                    content=round_output,
                )
            )
            responses.append(
                AgentResponse(
                    agent_name="Batched agents",
                    round_number=round_number,
                    content=round_output,
                )
            )
            transcript = f"{transcript}\n\n## Round {round_number}\n{round_output}"

        final_answer = await self.llm.complete(
            batched_synthesis_prompt(query, transcript, self.config.answer_style)
        )
        return DebateResult(
            query=query,
            model=self.config.model,
            final_answer=final_answer,
            responses=responses,
            critiques=critiques,
        )

    async def _initial_answers(self, state: DebateState) -> dict:
        async def answer(agent) -> AgentResponse:
            content = await self._agent_llm(agent.name).complete(
                initial_answer_prompt(state["query"], agent)
            )
            return AgentResponse(agent_name=agent.name, round_number=0, content=content)

        responses = await asyncio.gather(*(answer(agent) for agent in self.config.agents))
        return {"responses": list(responses)}

    async def _critique_round(self, state: DebateState) -> dict:
        round_number = state["current_round"] + 1
        latest_responses = self._latest_responses(state["responses"])

        async def critique(agent) -> Critique:
            content = await self._agent_llm(agent.name).complete(
                critique_prompt(state["query"], agent, latest_responses, round_number)
            )
            return Critique(agent_name=agent.name, round_number=round_number, content=content)

        critiques = await asyncio.gather(*(critique(agent) for agent in self.config.agents))
        return {"critiques": state["critiques"] + list(critiques)}

    async def _revision_round(self, state: DebateState) -> dict:
        round_number = state["current_round"] + 1
        latest_responses = {
            response.agent_name: response for response in self._latest_responses(state["responses"])
        }
        round_critiques = [
            critique for critique in state["critiques"] if critique.round_number == round_number
        ]

        async def revise(agent) -> AgentResponse:
            content = await self._agent_llm(agent.name).complete(
                revision_prompt(
                    state["query"],
                    agent,
                    latest_responses[agent.name],
                    round_critiques,
                    round_number,
                )
            )
            return AgentResponse(agent_name=agent.name, round_number=round_number, content=content)

        revisions = await asyncio.gather(*(revise(agent) for agent in self.config.agents))
        return {
            "responses": state["responses"] + list(revisions),
            "current_round": round_number,
        }

    async def _synthesize(self, state: DebateState) -> dict:
        latest_responses = self._latest_responses(state["responses"])
        final_answer = await self.llm.complete(
            synthesis_prompt(
                state["query"],
                latest_responses,
                state["critiques"],
                self.config.answer_style,
            )
        )
        return {"final_answer": final_answer}

    def _route_after_revision(self, state: DebateState) -> str:
        return "continue" if state["current_round"] < self.config.rounds else "finish"

    def _agent_llm(self, agent_name: str) -> LLMClient:
        return self.agent_llms.get(agent_name, self.llm)

    @staticmethod
    def _latest_responses(responses: list[AgentResponse]) -> list[AgentResponse]:
        latest_by_agent: dict[str, AgentResponse] = {}
        for response in responses:
            current = latest_by_agent.get(response.agent_name)
            if current is None or response.round_number > current.round_number:
                latest_by_agent[response.agent_name] = response
        return list(latest_by_agent.values())
