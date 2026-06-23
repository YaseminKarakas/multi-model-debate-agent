from __future__ import annotations

from .models import AgentProfile, AgentResponse, Critique


def _agent_protocol(agent: AgentProfile) -> str:
    protocols = {
        "Analyst": (
            "Create the baseline. Focus on structure, assumptions, evidence, missing context, "
            "and concrete recommendations. Do not polish for tone unless it affects usefulness."
        ),
        "Skeptic": (
            "Do not rewrite the answer. Find weak claims, generic language, missing proof, "
            "role-fit gaps, and risks. Be specific and demanding."
        ),
        "Synthesizer": (
            "Merge only the strongest accepted points into a concise output. Preserve useful "
            "nuance, remove repetition, and make the result immediately usable."
        ),
    }
    return protocols.get(
        agent.name,
        "Contribute a distinct perspective. Avoid repeating prior wording unless quoting it.",
    )


ANTI_REPETITION_RULES = (
    "Anti-repetition rules: do not repeat full sentences from prior answers unless quoting a "
    "specific flaw. Every round after the first must add at least two material new points or "
    "state exactly why no new point is justified. Prefer concrete deltas over restatement."
)


def compact_debate_prompt(query: str, agents: list[AgentProfile], rounds: int) -> list[dict[str, str]]:
    agent_lines = "\n".join(
        f"- {agent.name}: {agent.role} Stance: {agent.stance}" for agent in agents
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a debate moderator simulating a multi-agent reasoning process inside "
                "one response. Keep the internal debate concise, force disagreement where useful, "
                "then return a strong final answer. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Run a compact {rounds}-round debate among these agents:\n{agent_lines}\n\n"
                f"User query:\n{query}\n\n"
                "Return exactly these sections:\n"
                "1. Agent Positions\n"
                "2. Key Critiques\n"
                "3. Changes Made After Critique\n"
                "4. Agreed Enriched Answer\n"
                "5. Practical Next Steps"
            ),
        },
    ]


def batched_initial_prompt(query: str, agents: list[AgentProfile]) -> list[dict[str, str]]:
    agent_lines = "\n".join(
        f"- {agent.name}: {agent.role} Stance: {agent.stance}" for agent in agents
    )
    return [
        {
            "role": "system",
            "content": (
                "You are moderating a multi-agent debate. Generate distinct initial positions "
                "for every listed agent in one response. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\nAgents:\n{agent_lines}\n\n"
                "Return a section for each agent using this format:\n"
                "## Agent: <name>\n"
                "- Role task: <what this agent is optimizing for>\n"
                "- Initial position: <substantive answer>\n"
                "- Assumptions: <bullets>\n"
                "- Open risks: <bullets>"
            ),
        },
    ]


def batched_round_prompt(
    query: str,
    agents: list[AgentProfile],
    transcript: str,
    round_number: int,
) -> list[dict[str, str]]:
    agent_lines = "\n".join(
        f"- {agent.name}: {agent.role} Stance: {agent.stance}" for agent in agents
    )
    return [
        {
            "role": "system",
            "content": (
                "You are running one full debate round in a single model call. For each agent, "
                "critique the transcript and revise that agent's position. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\nAgents:\n{agent_lines}\n\n"
                f"Transcript so far:\n{transcript}\n\n"
                f"Run debate round {round_number}. Return exactly these sections:\n"
                "## Critiques\n"
                "- C<round>.<number> | <agent> | severity=<high/medium/low> | <specific flaw> | <required fix>\n\n"
                "## Revised Positions\n"
                "- <agent>: <changed from previous> | <critique IDs addressed> | <revised answer>"
            ),
        },
    ]


def _style_instruction(answer_style: str) -> str:
    if answer_style == "concise":
        return (
            "Write for production use: keep the final answer concise, direct, and low on "
            "meta-justification. Include only essential reasoning."
        )
    return (
        "Write for development/debugging: include enough reasoning, accepted critiques, and "
        "change details to inspect how the debate improved the answer."
    )


def batched_synthesis_prompt(
    query: str,
    transcript: str,
    answer_style: str = "debug",
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the moderator. Synthesize the batched multi-agent debate into one "
                "agreed enriched answer. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query:\n{query}\n\nDebate transcript:\n{transcript}\n\n"
                f"{_style_instruction(answer_style)}\n\n"
                "Return a polished final answer. Preserve important disagreements only if they "
                "matter, explain the reasoning, and include practical next steps when useful.\n\n"
                "Return exactly these sections:\n"
                "## Final Answer\n"
                "## Accepted Critiques\n"
                "## Rejected Or Deferred Critiques\n"
                "## Change Log"
            ),
        },
    ]


def initial_answer_prompt(query: str, agent: AgentProfile) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are {agent.name}. Role: {agent.role} "
                f"Debate stance: {agent.stance} "
                f"Protocol: {_agent_protocol(agent)} "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Give your initial contribution to the debate. Do not try to do every role. "
                "Optimize for your assigned protocol.\n\n"
                f"Query:\n{query}\n\n"
                "Return exactly these sections:\n"
                "## Position\n"
                "## Assumptions\n"
                "## Evidence Or Experience Needed\n"
                "## Risks / Weak Spots\n"
                "## Best Next Improvement"
            ),
        },
    ]


def critique_prompt(
    query: str,
    agent: AgentProfile,
    latest_responses: list[AgentResponse],
    round_number: int,
) -> list[dict[str, str]]:
    transcript = "\n\n".join(
        f"{response.agent_name}: {response.content}" for response in latest_responses
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are {agent.name}. Role: {agent.role} "
                f"Protocol: {_agent_protocol(agent)} "
                "Your job is to improve the group answer, not to win the debate. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Debate round {round_number}. Critique the current answers. Do not rewrite "
                "the final answer. Produce concrete critique items that a reviser can act on.\n\n"
                f"Original query:\n{query}\n\nCurrent answers:\n{transcript}\n\n"
                "Return exactly these sections:\n"
                "## Strongest Existing Points\n"
                "## Critique Items\n"
                "- C<round>.<number> | severity=<high/medium/low> | target=<agent/all> | flaw=<specific issue> | required_fix=<specific fix>\n"
                "## Missing Evidence Or Context\n"
                "## Non-Negotiable Changes"
            ),
        },
    ]


def revision_prompt(
    query: str,
    agent: AgentProfile,
    own_latest_response: AgentResponse,
    critiques: list[Critique],
    round_number: int,
) -> list[dict[str, str]]:
    critique_text = "\n\n".join(
        f"{critique.agent_name}: {critique.content}" for critique in critiques
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are {agent.name}. Role: {agent.role} "
                f"Protocol: {_agent_protocol(agent)} "
                "Revise your answer using the best critique from all participants. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Debate round {round_number}. Revise your answer.\n\n"
                f"Original query:\n{query}\n\n"
                f"Your previous answer:\n{own_latest_response.content}\n\n"
                f"Critiques:\n{critique_text}\n\n"
                "Return exactly these sections:\n"
                "## Changed From Previous\n"
                "## Critique IDs Addressed\n"
                "## New Points Added\n"
                "## Revised Position\n"
                "## Remaining Uncertainty"
            ),
        },
    ]


def synthesis_prompt(
    query: str,
    latest_responses: list[AgentResponse],
    critiques: list[Critique],
    answer_style: str = "debug",
) -> list[dict[str, str]]:
    answers = "\n\n".join(
        f"{response.agent_name}: {response.content}" for response in latest_responses
    )
    critique_text = "\n\n".join(
        f"{critique.agent_name}: {critique.content}" for critique in critiques[-6:]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the moderator. Create one agreed enriched answer from the debate. "
                "Preserve nuance, resolve disagreements, and be useful to the user. "
                f"{ANTI_REPETITION_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query:\n{query}\n\n"
                f"Final participant answers:\n{answers}\n\n"
                f"Recent critiques:\n{critique_text}\n\n"
                f"{_style_instruction(answer_style)}\n\n"
                "Return a polished answer with clear reasoning and practical next steps when relevant.\n\n"
                "Return exactly these sections:\n"
                "## Final Answer\n"
                "## Why This Version Is Stronger\n"
                "## Accepted Critiques\n"
                "## Rejected Or Deferred Critiques\n"
                "## Practical Next Steps"
            ),
        },
    ]
