r"""Fork-and-flush: periodically explore fresh-context branches from the current workspace.

    hmz exec -f fork_and_flush \
        -a explorer_1=claude/claude-opus-5:high \
        -a explorer_2=codex/gpt-5.6-sol:high \
        -a explorer_3=claude/claude-sonnet-5-5:high \
        -a explorer_4=codex/gpt-5.6-sol:high \
        -a judge=claude/claude-opus-5:high \
        -p budget.cost=10,budget.rounds=40 "the task"

Each fork starts from the current workspace in a temporary copy and a fresh conversation.
The configured grader scores each branch, the winner is copied back, then its agent continues
alone for `exploit_turns` turns before the next fork. Use `-p grader_command=./grade.sh` for a
deterministic grader that prints an integer from 0 to 100, or configure the `judge` agent.
"""

from __future__ import annotations

import asyncio
import shlex
from typing import NotRequired

import pydantic

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FlowContext,
    FlowParams,
    LocalEnv,
    Permission,
    PermissionKind,
    Session,
    ShellEnvMixin,
    TemporaryClonedDirEnvMixin,
    flow,
)

_SCORE_MAX = 100
_EXPLOIT_ROUND_COST = 2


class Score(pydantic.BaseModel):
    """A judge's comparable assessment of one candidate workspace."""

    score: int = pydantic.Field(ge=0, le=_SCORE_MAX)
    rationale: str


class Workspace(LocalEnv, ShellEnvMixin, TemporaryClonedDirEnvMixin):
    """The workspace and the isolated copies branches explore in."""


class Judge(Agent):
    """A read-only agent that scores candidate workspaces."""

    _permission = Permission(local=PermissionKind.READ, online=PermissionKind.NONE)


class Agents(AgentCollection):
    """The independent branch explorers and optional agent grader."""

    explorer_1: Agent
    explorer_2: NotRequired[Agent]
    explorer_3: NotRequired[Agent]
    explorer_4: NotRequired[Agent]
    judge: NotRequired[Judge]


class Envs(EnvCollection):
    """The project workspace, which can make temporary copies and run shell commands."""

    workspace: Workspace


class Params(FlowParams):
    """The fork schedule and optional deterministic grader command."""

    forks: int = pydantic.Field(default=4, ge=1, le=4)
    explore_turns: int = pydantic.Field(default=1, ge=1)
    exploit_turns: int = pydantic.Field(default=4, ge=1)
    grader_command: str | None = pydantic.Field(default=None, min_length=1)


def _adopt_script(source: str) -> str:
    """Safely replace worktree files, preserving metadata and restoring on failure."""
    quoted = shlex.quote(source)
    return f"""set -eu
src={quoted}
stage=.hmz-adopt-stage.$$
backup=.hmz-adopt-backup.$$
rollback=0
restore() {{
    result=$?
    trap - EXIT HUP INT TERM
    if [ "$rollback" -eq 1 ]; then
        for item in ./* ./.[!.]* ./..?*; do
            [ -e "$item" ] || [ -L "$item" ] || continue
            name=${{item##*/}}
            case "$name" in .git|.hmz|.humanize|"$stage"|"$backup") continue ;; esac
            rm -rf "$item" || result=1
        done
        for item in "$backup"/* "$backup"/.[!.]* "$backup"/..?*; do
            [ -e "$item" ] || [ -L "$item" ] || continue
            mv "$item" ./ || result=1
        done
    fi
    rm -rf "$stage" "$backup" || result=1
    exit "$result"
}}
trap restore EXIT
trap 'exit 1' HUP INT TERM
mkdir "$stage" "$backup"
for item in "$src"/* "$src"/.[!.]* "$src"/..?*; do
    [ -e "$item" ] || [ -L "$item" ] || continue
    name=${{item##*/}}
    case "$name" in .git|.hmz|.humanize) continue ;; esac
    cp -a "$item" "$stage/"
done
rollback=1
for item in ./* ./.[!.]* ./..?*; do
    [ -e "$item" ] || [ -L "$item" ] || continue
    name=${{item##*/}}
    case "$name" in .git|.hmz|.humanize|"$stage"|"$backup") continue ;; esac
    mv "$item" "$backup/"
done
for item in "$stage"/* "$stage"/.[!.]* "$stage"/..?*; do
    [ -e "$item" ] || [ -L "$item" ] || continue
    mv "$item" ./
done
rollback=0
rm -rf "$backup" "$stage"
trap - EXIT HUP INT TERM
"""


def _grader_script(command: str, task: str) -> str:
    """Run the configured command with the task available as `HMZ_TASK`."""
    return f"""set -eu
HMZ_TASK={shlex.quote(task)}
export HMZ_TASK
{command}
"""


@flow(agents=Agents, envs=Envs, params=Params)
async def fork_and_flush(
    task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext
) -> None:
    """Explore fresh-context branches, keep the best-scored work, and exploit between forks."""
    workspace = envs["workspace"]
    explorers = [
        agents[name]
        for name in ("explorer_1", "explorer_2", "explorer_3", "explorer_4")
        if name in agents
    ][: params.forks]
    if len(explorers) < params.forks:
        raise ValueError(
            f"forks={params.forks} needs that many explorer roles; "
            f"only {len(explorers)} were provided"
        )
    judge = agents["judge"] if "judge" in agents else None
    if params.grader_command is None and judge is None:
        raise ValueError("configure either grader_command or the judge agent")
    round_limit = ctx.budget.rounds
    rounds = 0
    leader: tuple[Agent, Session, Workspace, str] | None = None
    leader_score: int | None = None
    clone_sources: dict[str, Workspace] = {}
    since_fork = params.exploit_turns

    async def adopt(target: Workspace, source: Workspace) -> None:
        status, out, err = await target.exec(
            ["sh", "-c", _adopt_script(str(source.workdir))]
        )
        if status:
            raise RuntimeError(
                err.strip() or out.strip() or "could not adopt workspace"
            )

    async def sync(source: Workspace) -> None:
        await adopt(workspace, source)

    async def score(candidate: Workspace, label: str) -> int:
        if params.grader_command is not None:
            status, out, err = await candidate.exec(
                ["sh", "-c", _grader_script(params.grader_command, task)]
            )
            if status:
                raise RuntimeError(
                    err.strip() or out.strip() or "grader_command failed"
                )
            try:
                value = int(out.strip())
            except ValueError as failure:
                raise RuntimeError(
                    "grader_command must print one integer score from 0 to 100"
                ) from failure
            if not 0 <= value <= _SCORE_MAX:
                raise RuntimeError("grader_command score must be from 0 to 100")
            return value
        assert judge is not None  # noqa: S101 -- checked before any work starts
        result = await judge.run(
            "Read this workspace and assess how well it solves the task. Use the same 0-100 "
            "scale for every candidate: correctness and verified tests matter most, followed "
            "by completeness and maintainability. Do not change files. Return a score and "
            "concise evidence-based rationale.\n\n"
            f"Task:\n{task}\n\n{label}.",
            session=await judge.spawn(),
            env=candidate,
            output_schema=Score,
        )
        return result.score

    async def destroy(clone_ids: list[str]) -> None:
        """Try every cleanup even if one temporary copy cannot be removed."""
        failures: list[BaseException] = []
        for clone_id in clone_ids:
            source = clone_sources.get(clone_id)
            if source is None:
                continue
            try:
                await source.destroy_temp_clone(clone_id)
            except BaseException as failure:  # noqa: BLE001 -- attempt every cleanup first
                failures.append(failure)
            else:
                clone_sources.pop(clone_id, None)
        if failures:
            raise failures[0]

    try:
        while round_limit is None or rounds < round_limit:
            remaining = None if round_limit is None else round_limit - rounds
            if since_fork < params.exploit_turns and (
                remaining is None or remaining >= _EXPLOIT_ROUND_COST
            ):
                assert leader is not None  # noqa: S101 -- a leader follows every completed fork
                agent, session, branch, leader_id = leader
                checkpoint_id = f"fork-and-flush-checkpoint-{id(ctx)}-{rounds}"
                checkpoint = await branch.derive_temp_clone(checkpoint_id)
                clone_sources[checkpoint_id] = branch
                try:
                    await agent.run(
                        "Continue improving the task from this workspace. Preserve working "
                        "behavior, and run the most relevant checks you can.",
                        session=session,
                        env=branch,
                    )
                    rounds += 1
                    candidate_score = await score(branch, "Current trunk")
                    rounds += 1
                    if leader_score is None or candidate_score > leader_score:
                        leader_score = candidate_score
                    else:
                        await adopt(branch, checkpoint)
                        leader = (agent, await agent.spawn(), branch, leader_id)
                except BaseException:
                    await adopt(branch, checkpoint)
                    leader = (agent, await agent.spawn(), branch, leader_id)
                    raise
                finally:
                    await destroy([checkpoint_id])
                since_fork += 1
                await sync(branch)
                continue

            fork_count = len(explorers)
            if remaining is not None:
                # Every branch takes explore_turns and gets one score.
                rounds_per_branch = params.explore_turns + 1
                fork_count = min(
                    fork_count,
                    max(0, remaining // rounds_per_branch),
                )
            if fork_count == 0:
                if leader is None:
                    break
                # The remaining rounds cannot pay for a complete fork and its score(s).
                # Resetting the cadence here would spin forever without spending budget.
                break

            source: Workspace = workspace if leader is None else leader[2]
            branches: list[tuple[Agent, Session, Workspace, str]] = []
            try:
                for index, agent in enumerate(explorers[:fork_count]):
                    clone_id = f"fork-and-flush-{id(ctx)}-{rounds}-{index}"
                    branch = await source.derive_temp_clone(clone_id)
                    clone_sources[clone_id] = source
                    branches.append((agent, await agent.spawn(), branch, clone_id))

                branch_failures: list[Exception] = []
                for turn in range(params.explore_turns):
                    active = list(branches)
                    prompt = (
                        task
                        if turn == 0
                        else (
                            "Continue exploring a strong solution to the task from this workspace. "
                            "Preserve working behavior, and run the most relevant checks you can."
                        )
                    )
                    outcomes = await asyncio.gather(
                        *(
                            agent.run(prompt, session=session, env=branch)
                            for agent, session, branch, _ in active
                        ),
                        return_exceptions=True,
                    )
                    rounds += len(active)
                    failed_ids: list[str] = []
                    for item, outcome in zip(active, outcomes, strict=True):
                        if isinstance(outcome, BaseException):
                            if not isinstance(outcome, Exception):
                                raise outcome
                            branch_failures.append(outcome)
                            failed_ids.append(item[3])
                    if failed_ids:
                        await destroy(failed_ids)
                        failed = set(failed_ids)
                        branches = [item for item in branches if item[3] not in failed]

                if not branches and leader is None:
                    raise branch_failures[0]

                scores: list[tuple[int, int]] = []
                for index, (_, _, branch, _) in enumerate(branches):
                    value = await score(
                        branch, f"Candidate {index + 1} of {len(branches)}"
                    )
                    rounds += 1
                    scores.append((value, index))

                # Keep the incumbent unless a candidate beats its best known score.
                # Otherwise preserve declaration order on candidate ties.
                best_score, winner = (
                    max(scores, key=lambda scored: scored[0]) if scores else (-1, -1)
                )
                if scores and (leader_score is None or best_score > leader_score):
                    leader_agent, leader_session, leader_branch, winner_id = branches[
                        winner
                    ]
                    old_leader = leader
                    await sync(leader_branch)
                    leader = (leader_agent, leader_session, leader_branch, winner_id)
                    leader_score = best_score
                    if old_leader is not None:
                        _, _, _, old_id = old_leader
                        await destroy([old_id])
                since_fork = 0
            finally:
                await destroy(
                    [
                        clone_id
                        for _, _, _, clone_id in branches
                        if leader is None or clone_id != leader[3]
                    ]
                )
    finally:
        try:
            if leader is not None:
                await sync(leader[2])
        finally:
            # A failed final sync must not keep temporary workspaces alive. Cleanup belongs
            # to the outermost finally so it also runs when adopting the trunk fails.
            await destroy(list(clone_sources))
