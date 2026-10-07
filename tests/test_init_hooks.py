"""
Single-process tests for the init functions of DistributedEnvironment.

worker_init, broker_init and init_args are checked when the environment
is constructed, which every rank does, so a bad setting fails on every
rank before any MPI call.  These tests cover those checks, and that on
one process, where the controller runs every task itself, no init
function is called.
"""

import sys
import threading

import pytest

from distwq.config import Config
from distwq.future import DistributedEnvironment, WorkerGroupConfig
from distwq.mpi import GroupingMethod

CALLS = []


def record_init(server, *args):
    CALLS.append((server, args))


def plus_one(x):
    return x + 1


def spawn_config(broker_is_worker=True):
    return WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=2,
        broker_is_worker=broker_is_worker,
    )


def split_config(broker_is_worker=False, nprocs_per_worker=2, method=None):
    return WorkerGroupConfig(
        grouping_method=method or GroupingMethod.GroupSplit,
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=broker_is_worker,
    )


@pytest.fixture(autouse=True)
def clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def make_nested():
    def nested(server):
        pass  # pragma: no cover

    return nested


class Holder:
    @staticmethod
    def init(server):
        CALLS.append((server, ()))


@pytest.mark.parametrize("hook", ["worker_init", "broker_init"])
@pytest.mark.parametrize(
    "bad",
    [lambda server: None, make_nested(), "record_init"],
    ids=["lambda", "nested", "string"],
)
def test_init_function_must_be_module_level(hook, bad):
    with pytest.raises(TypeError, match="cannot be sent to workers"):
        DistributedEnvironment(Config(), split_config(), **{hook: bad})


def test_unreachable_function_is_rejected():
    def stray(server):
        pass  # pragma: no cover

    stray.__qualname__ = "not_in_module"
    stray.__module__ = __name__
    with pytest.raises(TypeError, match="not reachable"):
        DistributedEnvironment(worker_init=stray)


def test_static_method_is_accepted():
    env = DistributedEnvironment(worker_init=Holder.init)
    assert not env.is_active()


@pytest.mark.parametrize("bad", [[1], "ab", None, 1])
def test_init_args_must_be_a_tuple(bad):
    with pytest.raises(TypeError, match="init_args must be a tuple"):
        DistributedEnvironment(worker_init=record_init, init_args=bad)


def test_broker_init_needs_grouping():
    with pytest.raises(ValueError, match="needs worker grouping"):
        DistributedEnvironment(broker_init=record_init)
    with pytest.raises(ValueError, match="needs worker grouping"):
        DistributedEnvironment(Config(), WorkerGroupConfig(), broker_init=record_init)


def test_no_grouping_with_several_processes_counts_as_grouping():
    env = DistributedEnvironment(
        Config(),
        split_config(method=GroupingMethod.NoGrouping),
        broker_init=record_init,
    )
    assert not env.is_active()


@pytest.mark.parametrize("config", [split_config, spawn_config])
def test_broker_init_rejects_working_broker(config):
    with pytest.raises(ValueError, match="broker_is_worker=True"):
        DistributedEnvironment(
            Config(), config(broker_is_worker=True), broker_init=record_init
        )


def test_spawn_rejects_worker_init_from_main(monkeypatch):
    def main_init(server):
        pass  # pragma: no cover

    main_init.__module__ = "__main__"
    main_init.__qualname__ = "main_init"
    monkeypatch.setattr(sys.modules["__main__"], "main_init", main_init, raising=False)
    with pytest.raises(ValueError, match="defined in __main__"):
        DistributedEnvironment(Config(), spawn_config(), worker_init=main_init)
    # Without spawning, the same function is found in the same process.
    DistributedEnvironment(Config(), split_config(), worker_init=main_init)


def test_spawn_rejects_unpicklable_init_args():
    with pytest.raises(TypeError, match="must be picklable"):
        DistributedEnvironment(
            Config(),
            spawn_config(),
            worker_init=record_init,
            init_args=(threading.Lock(),),
        )
    # Without spawning, the arguments never leave the process.
    DistributedEnvironment(
        Config(), split_config(), worker_init=record_init, init_args=(threading.Lock(),)
    )


def test_spawn_accepts_broker_init_from_main(monkeypatch):
    # broker_init runs on the broker, which is not spawned.
    def main_broker_init(server):
        pass  # pragma: no cover

    main_broker_init.__module__ = "__main__"
    main_broker_init.__qualname__ = "main_broker_init"
    monkeypatch.setattr(
        sys.modules["__main__"], "main_broker_init", main_broker_init, raising=False
    )
    DistributedEnvironment(
        Config(), spawn_config(broker_is_worker=False), broker_init=main_broker_init
    )


def test_init_is_not_called_without_workers():
    with DistributedEnvironment(
        Config(), worker_init=record_init, init_args=(1, 2)
    ) as env:
        assert env.server is None
        for controller in env.controller_session():
            assert controller.submit(plus_one, 1).result() == 2
    assert CALLS == []


def test_server_is_none_before_entering():
    env = DistributedEnvironment(worker_init=record_init)
    assert env.server is None
    assert CALLS == []
