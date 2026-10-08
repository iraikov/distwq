"""
Single-process tests for the max_workers limit and the controller statistics.

_limited_roles() decides, for one rank, how many workers are used and
whether this rank is one of them; it is tested for every position of the
controller rank.  limit_workers() applies that decision to the module
values of distwq.mpi; it is tested here with those values replaced, to
stand in for a rank of a larger world.  The statistics tests run a
DistributedEnvironment on one process, where the controller runs every
task itself.
"""

import logging

import pytest
from run_guards import single_process_only

import distwq.mpi as distwq_mpi
from distwq.config import Config, ResourceConfig
from distwq.future import DistributedEnvironment
from distwq.mpi import _limited_roles, limit_workers

pytestmark = single_process_only


def ok(x):
    return x + 1


def fails(x):
    raise ValueError(f"bad value {x}")


def workers_of(max_workers, world_size, controller_rank):
    """The ranks that are workers, and the n_workers every rank computes."""
    workers = []
    counts = set()
    for rank in range(world_size):
        n, is_worker = _limited_roles(
            max_workers,
            world_size - 1,
            rank,
            controller_rank,
            rank == controller_rank,
        )
        counts.add(n)
        if is_worker:
            workers.append(rank)
    assert len(counts) == 1, "all ranks must agree on the number of workers"
    return workers, counts.pop()


@pytest.mark.parametrize("max_workers", [None, -1, 0])
@pytest.mark.parametrize("controller_rank", [0, 2, 4])
def test_no_limit_uses_every_other_rank(max_workers, controller_rank):
    workers, n = workers_of(max_workers, 5, controller_rank)
    assert n == 4
    assert workers == [r for r in range(5) if r != controller_rank]


@pytest.mark.parametrize(
    "controller_rank, expected",
    [
        (0, [1, 2]),
        (2, [0, 1]),
        (4, [0, 1]),
    ],
)
def test_limit_keeps_lowest_worker_indices(controller_rank, expected):
    workers, n = workers_of(2, 5, controller_rank)
    assert n == 2
    assert workers == expected


@pytest.mark.parametrize("controller_rank", [0, 1, 3])
def test_limit_of_one_between_controller_positions(controller_rank):
    workers, n = workers_of(1, 4, controller_rank)
    assert n == 1
    assert len(workers) == 1
    assert controller_rank not in workers


@pytest.mark.parametrize("max_workers", [4, 5, 100])
def test_limit_at_or_above_world_uses_every_worker(max_workers):
    workers, n = workers_of(max_workers, 5, 0)
    assert n == 4
    assert workers == [1, 2, 3, 4]


@pytest.mark.parametrize("bad", [True, False, 2.0, "2", [2]])
def test_limit_workers_rejects_non_int(bad):
    with pytest.raises(TypeError, match="max_workers must be an int or None"):
        limit_workers(bad)


@pytest.fixture
def world_of_five(monkeypatch):
    """Make distwq.mpi look like rank 3 of 5, with the controller on rank 0."""
    monkeypatch.setattr(distwq_mpi, "has_mpi", True)
    monkeypatch.setattr(distwq_mpi, "spawned", False)
    monkeypatch.setattr(distwq_mpi, "size", 5)
    monkeypatch.setattr(distwq_mpi, "rank", 3, raising=False)
    monkeypatch.setattr(distwq_mpi, "controller_rank", 0, raising=False)
    monkeypatch.setattr(distwq_mpi, "is_controller", False)
    monkeypatch.setattr(distwq_mpi, "is_worker", True)
    monkeypatch.setattr(distwq_mpi, "n_workers", 4)
    monkeypatch.setattr(distwq_mpi, "_world_workers", 4)


def test_limit_can_be_raised_again(world_of_five):
    assert limit_workers(2) == 2
    assert distwq_mpi.n_workers == 2
    # Rank 3 has worker index 2, which is outside a limit of 2.
    assert distwq_mpi.is_worker is False
    assert limit_workers(None) == 4
    assert distwq_mpi.n_workers == 4
    assert distwq_mpi.is_worker is True
    assert limit_workers(3) == 3
    assert distwq_mpi.is_worker is True
    assert limit_workers(-1) == 4


def test_bad_limit_changes_nothing(world_of_five):
    limit_workers(2)
    with pytest.raises(TypeError):
        limit_workers(1.5)
    assert distwq_mpi.n_workers == 2
    assert distwq_mpi.is_worker is False


@pytest.mark.parametrize("field", ["one_rank", "no_mpi", "spawned"])
def test_limit_leaves_single_process_unchanged(monkeypatch, world_of_five, field):
    if field == "one_rank":
        monkeypatch.setattr(distwq_mpi, "size", 1)
    elif field == "no_mpi":
        monkeypatch.setattr(distwq_mpi, "has_mpi", False)
    else:
        monkeypatch.setattr(distwq_mpi, "spawned", True)
    assert limit_workers(1) == 4
    assert distwq_mpi.n_workers == 4
    assert distwq_mpi.is_worker is True


def limited_config(max_workers=2):
    return Config(resources=ResourceConfig(max_workers=max_workers))


def test_environment_with_limit_runs_on_one_process():
    with DistributedEnvironment(limited_config()) as env:
        assert env.worker_count == 0
        for controller in env.controller_session():
            assert controller.submit(ok, 1).result() == 2


def test_environment_rejects_bad_limit_before_setup():
    env = DistributedEnvironment(limited_config("2"))
    with pytest.raises(TypeError):
        env.__enter__()
    assert not env.is_active()
    # Nothing was left active: a new environment can be entered.
    with DistributedEnvironment(Config()) as env:
        assert env.is_active()


def test_stats_record_every_task_with_its_error_flag():
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            assert controller.stats == ()
            good = controller.submit(ok, 1)
            bad = controller.submit(fails, 2)
            assert good.result() == 2
            with pytest.raises(ValueError, match="bad value 2"):
                bad.result()
            stats = controller.stats
    assert isinstance(stats, tuple)
    assert [s["id"] for s in stats] == [good.task_id, bad.task_id]
    assert [s["error"] for s in stats] == [False, True]
    assert [s["n_processed"] for s in stats] == [1, 2]
    assert all(s["rank"] == 0 for s in stats)


def test_stats_are_a_copy():
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            controller.submit(ok, 1).result()
            snapshot = controller.stats
            snapshot[0]["id"] = "changed"
            with pytest.raises(AttributeError):
                snapshot.append({})
            assert controller.stats[0]["id"] != "changed"
            assert len(controller.stats) == 1


def test_stats_cover_every_session_of_the_environment():
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            controller.submit(ok, 1).result()
        for controller in env.controller_session():
            controller.submit(ok, 2).result()
            assert len(controller.stats) == 2


def test_info_logs_and_returns_summary(caplog):
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            controller.submit(ok, 1).result()
            controller.submit(ok, 2).result()
            with caplog.at_level(logging.INFO, logger="distwq.future"):
                text = controller.info()
    assert "results collected:         2" in text
    assert any(
        r.name == "distwq.future" and r.getMessage() == text for r in caplog.records
    )


def test_info_without_tasks_is_empty(caplog, capsys):
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            with caplog.at_level(logging.INFO, logger="distwq.future"):
                caplog.clear()
                assert controller.info() == ""
            assert caplog.records == []
    assert capsys.readouterr().out == ""


def test_mpi_controller_info_prints_summary(capsys):
    controller = distwq_mpi.MPIController()
    controller.info()
    assert capsys.readouterr().out == ""
    controller.submit_call("ok", (1,), module_name=__name__)
    controller.info()
    assert capsys.readouterr().out == controller.summary() + "\n"
