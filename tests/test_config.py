"""
Single-process tests for distwq.config, the type checks on the settings
given to DistributedEnvironment, and the time_limit argument of run().

Config holds only settings that distwq applies.  Anything else, including
the settings of earlier versions that were never applied, raises when it
is given.  The same holds for a time limit given to run() or
MPIController.
"""

import dataclasses

import pytest
from run_guards import single_process_only

import distwq.config as config_module
import distwq.mpi as distwq_mpi
from distwq.config import Config, LoggingConfig, ResourceConfig, WorkerGroupConfig
from distwq.future import DistributedEnvironment

pytestmark = single_process_only


def test_defaults():
    config = Config()
    assert config.resources.max_workers == -1
    assert config.logging.verbose is False


def test_helpers_return_new_configs():
    config = Config()
    limited = config.with_workers(3).with_verbose()
    assert limited.resources.max_workers == 3
    assert limited.logging.verbose is True
    assert config == Config()


def test_settings_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().resources.max_workers = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().logging = LoggingConfig(verbose=True)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ResourceConfig(cores_per_worker=2),
        lambda: ResourceConfig(memory_per_worker="2GB"),
        lambda: ResourceConfig(gpu_per_worker=1),
        lambda: LoggingConfig(log_level="DEBUG"),
        lambda: LoggingConfig(profile=True),
        lambda: LoggingConfig(stats_callback=print),
        lambda: Config(execution=None),
        lambda: Config(mpi=None),
        lambda: Config(grouping=WorkerGroupConfig()),
        lambda: dataclasses.replace(Config(), grouping=WorkerGroupConfig()),
    ],
    ids=[
        "cores_per_worker",
        "memory_per_worker",
        "gpu_per_worker",
        "log_level",
        "profile",
        "stats_callback",
        "execution",
        "mpi_settings",
        "grouping",
        "grouping_by_replace",
    ],
)
def test_unknown_settings_raise(make):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        make()


@pytest.mark.parametrize(
    "name",
    [
        "ExecutionConfig",
        "MPIConfig",
        "SpawnMethod",
    ],
)
def test_unused_setting_classes_are_gone(name):
    assert not hasattr(config_module, name)


@pytest.mark.parametrize(
    "name", ["for_development", "for_production", "for_neuroscience", "with_timeout"]
)
def test_presets_are_gone(name):
    assert not hasattr(Config, name)


@pytest.mark.parametrize(
    "bad",
    [{"resources": {"max_workers": 2}}, WorkerGroupConfig(), ResourceConfig()],
    ids=["dict", "worker_group_config", "resource_config"],
)
def test_environment_rejects_config_of_wrong_type(bad):
    with pytest.raises(TypeError, match="config must be a Config"):
        DistributedEnvironment(bad)


def test_environment_rejects_worker_config_of_wrong_type():
    with pytest.raises(TypeError, match="worker_config must be a WorkerGroupConfig"):
        DistributedEnvironment(Config(), Config())


def test_environment_accepts_trimmed_config():
    with DistributedEnvironment(Config().with_workers(2).with_verbose()) as env:
        assert env.is_active()


def never_called(controller):
    raise AssertionError("run() called the controller function")


def record_call(controller, calls):
    calls.append(1)


@pytest.mark.parametrize("time_limit", [0, 10, 3.5])
def test_run_rejects_time_limit(time_limit):
    with pytest.raises(ValueError, match="time_limit"):
        distwq_mpi.run(
            fun_name="never_called", module_name=__name__, time_limit=time_limit
        )


def test_run_accepts_no_time_limit():
    calls = []
    distwq_mpi.run(
        fun_name="record_call", module_name=__name__, time_limit=None, args=(calls,)
    )
    assert calls == [1]


def test_controller_takes_no_time_limit():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        distwq_mpi.MPIController(None, time_limit=10)
    assert not hasattr(distwq_mpi.MPIController(None), "time_limit")
