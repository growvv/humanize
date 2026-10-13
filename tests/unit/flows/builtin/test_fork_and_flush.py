"""`fork_and_flush`: score isolated, fresh-context branches and continue from their winner."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, get_origin, get_type_hints
from unittest import mock

import pytest

from tests.unit.flows import doubles_flows as doubles

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def flow_module(engine: mock.Mock, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return doubles.builtin(monkeypatch, "fork_and_flush")


async def test_fork_and_flush_adopts_the_highest_scored_branch(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorers = [doubles.agent("explored"), doubles.agent("explored")]
    for explorer in explorers:
        explorer.run.side_effect = None
        explorer.run.return_value = "done"
    judge = doubles.agent()
    judge.run.side_effect = [
        flow_module.Score(score=25, rationale="fails a check"),
        flow_module.Score(score=91, rationale="passes the checks"),
    ]
    branches = [mock.Mock(workdir="/copies/first"), mock.Mock(workdir="/copies/second")]
    workspace = mock.Mock(workdir="/project")
    workspace.derive_temp_clone = mock.AsyncMock(side_effect=branches)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))
    await defined.fn(
        "implement the task",
        agents={
            "explorer_1": explorers[0],
            "explorer_2": explorers[1],
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=2, exploit_turns=1),
        ctx=mock.Mock(budget=mock.Mock(rounds=5)),
    )

    assert doubles.prompts(explorers[0]) == ["implement the task"]
    assert doubles.prompts(explorers[1]) == [
        "implement the task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
    ]
    assert workspace.exec.await_args.args[0][:2] == ["sh", "-c"]
    assert "/copies/second" in workspace.exec.await_args.args[0][2]
    assert workspace.destroy_temp_clone.await_count == 2
    judge_sessions = doubles.sessions(judge)
    assert len(judge_sessions) == 2
    assert judge_sessions[0] is not judge_sessions[1]


async def test_fork_width_requires_only_the_declared_explorers(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    hints = get_type_hints(defined.declared["agents"], include_extras=True)
    assert all(
        get_origin(hints[name]) is NotRequired
        for name in ("explorer_2", "explorer_3", "explorer_4")
    )

    workspace = mock.Mock(workdir="/project")
    workspace.derive_temp_clone = mock.AsyncMock()
    with pytest.raises(ValueError, match=r"forks=2 needs that many explorer roles"):
        await defined.fn(
            "task",
            agents={"explorer_1": doubles.agent(), "judge": doubles.agent()},
            envs={"workspace": workspace},
            params=defined.declared["params"](forks=2),
            ctx=mock.Mock(budget=mock.Mock(rounds=4)),
        )

    workspace.derive_temp_clone.assert_not_awaited()


async def test_fork_and_flush_scales_the_fork_to_the_remaining_rounds(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorer = doubles.agent("explored", "exploited")
    judge = doubles.agent()
    judge.run.side_effect = [
        flow_module.Score(score=80, rationale="first branch"),
        flow_module.Score(score=80, rationale="incumbent"),
        flow_module.Score(score=20, rationale="regression"),
    ]
    workspace = mock.Mock(workdir="/project")
    branch = mock.Mock(workdir="/copies/one")
    workspace.derive_temp_clone = mock.AsyncMock(return_value=branch)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))

    await defined.fn(
        "task",
        agents={
            "explorer_1": explorer,
            "explorer_2": doubles.agent(),
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](exploit_turns=1),
        ctx=mock.Mock(budget=mock.Mock(rounds=3)),
    )

    assert doubles.prompts(explorer) == [
        "task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
    ]
    workspace.derive_temp_clone.assert_awaited_once()
    assert judge.run.await_count == 1


async def test_fork_and_flush_stops_when_budget_cannot_pay_for_another_fork(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorer = doubles.agent("explored", "exploited")
    judge = doubles.agent()
    judge.run.side_effect = None
    judge.run.return_value = flow_module.Score(score=80, rationale="current trunk")
    workspace = mock.Mock(workdir="/project")
    branch = mock.Mock(workdir="/copies/one")
    workspace.derive_temp_clone = mock.AsyncMock(return_value=branch)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))

    await defined.fn(
        "task",
        agents={
            "explorer_1": explorer,
            "explorer_2": doubles.agent(),
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=1, exploit_turns=1),
        # Two rounds explore and judge the first branch, one exploits it. The
        # last round cannot fund both a new branch and the incumbent comparison.
        ctx=mock.Mock(budget=mock.Mock(rounds=4)),
    )

    assert doubles.prompts(explorer) == [
        "task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
    ]
    workspace.derive_temp_clone.assert_awaited_once()


async def test_each_explorer_uses_one_fresh_session_for_its_exploration_horizon(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorers = [
        doubles.agent("first pass", "second pass"),
        doubles.agent("first pass", "second pass"),
    ]
    judge = doubles.agent()
    judge.run.side_effect = [
        flow_module.Score(score=55, rationale="candidate one"),
        flow_module.Score(score=75, rationale="candidate two"),
    ]
    workspace = mock.Mock(workdir="/project")
    branches = [mock.Mock(workdir="/copies/first"), mock.Mock(workdir="/copies/second")]
    workspace.derive_temp_clone = mock.AsyncMock(side_effect=branches)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))

    await defined.fn(
        "task",
        agents={
            "explorer_1": explorers[0],
            "explorer_2": explorers[1],
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=2, explore_turns=2, exploit_turns=1),
        ctx=mock.Mock(budget=mock.Mock(rounds=6)),
    )

    assert doubles.prompts(explorers[0]) == [
        "task",
        (
            "Continue exploring a strong solution to the task from this workspace. "
            "Preserve working behavior, and run the most relevant checks you can."
        ),
    ]
    assert doubles.prompts(explorers[1]) == doubles.prompts(explorers[0])
    sessions = [doubles.sessions(explorer) for explorer in explorers]
    assert len(sessions[0]) == len(sessions[1]) == 2
    assert sessions[0][0] is sessions[0][1]
    assert sessions[1][0] is sessions[1][1]
    assert sessions[0][0] is not sessions[1][0]
    assert judge.run.await_count == 2


async def test_equal_scores_keep_the_first_branch(flow_module: ModuleType) -> None:
    defined = flow_module.fork_and_flush
    first, second = doubles.agent("first", "first continues"), doubles.agent("second")
    judge = doubles.agent()
    judge.run.side_effect = None
    judge.run.return_value = flow_module.Score(score=80, rationale="same score")
    workspace = mock.Mock(workdir="/project")
    branches = [mock.Mock(workdir="/copies/first"), mock.Mock(workdir="/copies/second")]
    workspace.derive_temp_clone = mock.AsyncMock(side_effect=branches)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))
    ctx = mock.Mock(budget=mock.Mock(rounds=6))

    await defined.fn(
        "task",
        agents={
            "explorer_1": first,
            "explorer_2": second,
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=2, exploit_turns=1),
        ctx=ctx,
    )

    assert doubles.prompts(first) == [
        "task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
    ]
    assert doubles.prompts(second) == ["task"]
    assert "/copies/first" in workspace.exec.await_args.args[0][2]


async def test_a_failed_branch_is_discarded_without_losing_other_candidates(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    failed, viable = (
        doubles.agent(RuntimeError("branch failed")),
        doubles.agent("done", "exploited", "exploited again"),
    )
    judge = doubles.agent()
    judge.run.side_effect = None
    judge.run.return_value = flow_module.Score(score=75, rationale="verified")
    workspace = mock.Mock(workdir="/project")
    branches = [
        mock.Mock(workdir="/copies/failed"),
        mock.Mock(workdir="/copies/viable"),
    ]
    workspace.derive_temp_clone = mock.AsyncMock(side_effect=branches)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))
    ctx = mock.Mock(budget=mock.Mock(rounds=5))

    await defined.fn(
        "task",
        agents={
            "explorer_1": failed,
            "explorer_2": viable,
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=2, exploit_turns=1),
        ctx=ctx,
    )

    assert judge.run.await_count == 1
    assert "Candidate 1 of 1" in judge.run.await_args.args[0]
    workspace.destroy_temp_clone.assert_any_await(f"fork-and-flush-{id(ctx)}-0-0")
    assert workspace.exec.await_args.args[0][2].find("/copies/viable") >= 0


async def test_the_winning_branch_stays_the_trunk_for_the_next_fork(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorer = doubles.agent(
        "first branch", "exploit", "second branch", "exploit again"
    )
    judge = doubles.agent()
    judge.run.side_effect = [
        flow_module.Score(score=80, rationale="first branch"),
        flow_module.Score(score=80, rationale="incumbent"),
        flow_module.Score(score=20, rationale="regression"),
    ]
    workspace = mock.Mock(workdir="/project")
    first = mock.Mock(workdir="/copies/first")
    second = mock.Mock(workdir="/copies/second")
    workspace.derive_temp_clone = mock.AsyncMock(return_value=first)
    first.derive_temp_clone = mock.AsyncMock(return_value=second)
    workspace.destroy_temp_clone = mock.AsyncMock()
    first.destroy_temp_clone = mock.AsyncMock()
    second.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))
    ctx = mock.Mock(budget=mock.Mock(rounds=7))

    await defined.fn(
        "task",
        agents={
            "explorer_1": explorer,
            "explorer_2": doubles.agent(),
            "explorer_3": doubles.agent(),
            "explorer_4": doubles.agent(),
            "judge": judge,
        },
        envs={"workspace": workspace},
        params=defined.declared["params"](forks=1, exploit_turns=1),
        ctx=ctx,
    )

    first.derive_temp_clone.assert_awaited_once()
    assert doubles.prompts(explorer) == [
        "task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
        "task",
        (
            "Continue improving the task from this workspace. Preserve working behavior, "
            "and run the most relevant checks you can."
        ),
    ]
    first.destroy_temp_clone.assert_awaited_once_with(f"fork-and-flush-{id(ctx)}-3-0")
    workspace.destroy_temp_clone.assert_awaited_once_with(
        f"fork-and-flush-{id(ctx)}-0-0"
    )
    second.destroy_temp_clone.assert_not_awaited()
    assert judge.run.await_count == 3
    judge_sessions = doubles.sessions(judge)
    assert judge_sessions[0] is not judge_sessions[1]
    assert judge_sessions[1] is not judge_sessions[2]


async def test_final_sync_failure_still_destroys_the_winning_clone(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorer = doubles.agent("first branch", RuntimeError("exploit failed"))
    judge = doubles.agent()
    judge.run.side_effect = None
    judge.run.return_value = flow_module.Score(score=80, rationale="good")
    workspace = mock.Mock(workdir="/project")
    branch = mock.Mock(workdir="/copies/winner")
    workspace.derive_temp_clone = mock.AsyncMock(return_value=branch)
    workspace.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(
        side_effect=[(0, "", ""), RuntimeError("final sync failed")]
    )
    ctx = mock.Mock(budget=mock.Mock(rounds=3))

    with pytest.raises(RuntimeError, match="final sync failed"):
        await defined.fn(
            "task",
            agents={
                "explorer_1": explorer,
                "explorer_2": doubles.agent(),
                "explorer_3": doubles.agent(),
                "explorer_4": doubles.agent(),
                "judge": judge,
            },
            envs={"workspace": workspace},
            params=defined.declared["params"](forks=1, exploit_turns=1),
            ctx=ctx,
        )

    workspace.destroy_temp_clone.assert_awaited_once_with(
        f"fork-and-flush-{id(ctx)}-0-0"
    )


async def test_a_failed_cleanup_stays_registered_for_the_final_retry(
    flow_module: ModuleType,
) -> None:
    defined = flow_module.fork_and_flush
    explorer = doubles.agent("first", "exploit", "second")
    judge = doubles.agent()
    judge.run.side_effect = [
        flow_module.Score(score=80, rationale="first"),
        flow_module.Score(score=80, rationale="incumbent"),
        flow_module.Score(score=90, rationale="second"),
    ]
    workspace = mock.Mock(workdir="/project")
    first = mock.Mock(workdir="/copies/first")
    second = mock.Mock(workdir="/copies/second")
    workspace.derive_temp_clone = mock.AsyncMock(return_value=first)
    first.derive_temp_clone = mock.AsyncMock(return_value=second)
    workspace.destroy_temp_clone = mock.AsyncMock(
        side_effect=[RuntimeError("temporary cleanup failure"), None]
    )
    first.destroy_temp_clone = mock.AsyncMock()
    second.destroy_temp_clone = mock.AsyncMock()
    workspace.exec = mock.AsyncMock(return_value=(0, "", ""))
    ctx = mock.Mock(budget=mock.Mock(rounds=6))

    with pytest.raises(RuntimeError, match="temporary cleanup failure"):
        await defined.fn(
            "task",
            agents={
                "explorer_1": explorer,
                "explorer_2": doubles.agent(),
                "explorer_3": doubles.agent(),
                "explorer_4": doubles.agent(),
                "judge": judge,
            },
            envs={"workspace": workspace},
            params=defined.declared["params"](forks=1, exploit_turns=1),
            ctx=ctx,
        )

    assert workspace.destroy_temp_clone.await_count == 2
