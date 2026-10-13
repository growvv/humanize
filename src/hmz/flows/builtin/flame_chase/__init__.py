r"""Flame chase (flowbench: flame_chase) -- two agents take turns on the same task.

    hmz exec -f flame_chase -a first_chaser=claude/claude-opus-5:high \
        -a second_chaser=codex/gpt-5.6-sol:high \
        -p budget.cost=10,budget.rounds=2 "the task"

Each turn is a fresh session, so the two share nothing but the repository. The time, cost and
token budgets end it when spent. `budget.rounds` optionally limits the number of agent turns;
with two rounds, each chaser runs once. Without it, the loop keeps going until another budget
is spent. Whose turn is next is kept, so `--resume` picks up with that chaser.
"""

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FlowContext,
    FlowParams,
    LocalEnv,
    flow,
)


class Agents(AgentCollection):
    """The two that take turns, first the one and then the other."""

    first_chaser: Agent
    second_chaser: Agent


class Envs(EnvCollection):
    """Where both work: the workspace the run was started in."""

    workspace: LocalEnv


@flow(agents=Agents, envs=Envs, params=FlowParams, resumable=True)
async def flame_chase(
    task: str, *, agents: Agents, envs: Envs, params: FlowParams, ctx: FlowContext
) -> None:
    """Two agents take turns on the same task until the budget is spent."""
    state = ctx.state
    assert state is not None  # noqa: S101 -- a resumable flow is always handed its state
    chasers = (agents["first_chaser"], agents["second_chaser"])
    turn = state["turn"] if "turn" in state else 0
    rounds = state["rounds"] if "rounds" in state else 0
    round_limit = ctx.budget.rounds
    while round_limit is None or rounds < round_limit:
        chaser = chasers[turn]
        session = await chaser.spawn()
        await chaser.run(task, session=session, env=envs["workspace"])
        state["turn"] = turn = (turn + 1) % len(chasers)
        state["rounds"] = rounds = rounds + 1
