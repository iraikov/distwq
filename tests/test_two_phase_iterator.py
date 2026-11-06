#!/usr/bin/env python3
"""
Pytest-based test suite for Two-Phase Iterator with MPI.

Install pytest-mpi: pip install pytest-mpi
Run with: mpirun -n 4 pytest test_two_phase_iterator_pytest.py

Alternative: pytest test_two_phase_iterator_pytest.py (runs with 1 process)
"""

import sys
import pytest

try:
    from mpi4py import MPI

    HAS_MPI = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    HAS_MPI = False
    COMM = None
    RANK = 0
    SIZE = 1
    pytest.skip("MPI not available", allow_module_level=True)

from distwq.future import (
    DistributedEnvironment,
    TaskController,
    DistributedFuture,
    Config,
)


# Test functions
def simple_add(a: float, b: float) -> float:
    return a + b


def simple_multiply(x: float, factor: float = 2.0) -> float:
    return x * factor


# def test_model(tau_m: float, tau_s: float, duration: float = 100.0) -> Dict[str, Any]:
#     firing_rate = max(0, 10.0 - abs(tau_m - 20.0) - abs(tau_s - 5.0))
#     return {
#         'tau_m': tau_m,
#         'tau_s': tau_s,
#         'firing_rate': firing_rate,
#         'duration': duration,
#         'spikes': int(firing_rate * duration / 1000.0)
#     }


@pytest.fixture
def config():
    """Provide a Config instance."""
    return Config()


@pytest.fixture
def is_controller():
    """Check if this is the controller rank."""
    return RANK == 0


# Markers
pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")


# Tests
class TestDistributedEnvironment:
    """Test DistributedEnvironment functionality."""

    def test_creation(self, config):
        """Test environment creation."""
        env = DistributedEnvironment(config)
        assert env is not None
        assert not env.is_active()

    def test_context_manager(self, config):
        """Test environment as context manager."""
        with DistributedEnvironment(config) as env:
            assert env.is_active()
            if SIZE > 1 and RANK == 0:
                assert env.worker_count > 0

        if RANK == 0:
            assert not env.is_active()

    def test_controller_session(self, config, is_controller):
        """Test controller session iterator."""
        controller_count = 0

        with DistributedEnvironment(config) as env:
            for controller in env.controller_session():
                controller_count += 1
                assert isinstance(controller, TaskController)

        if is_controller:
            assert controller_count == 1
        else:
            assert controller_count == 0


class TestTaskExecution:
    """Test task submission and execution."""

    def test_simple_task(self, config, is_controller):
        """Test simple task submission."""
        result = None

        with DistributedEnvironment(config) as env:
            for controller in env.controller_session():
                future = controller.submit(simple_add, 2.0, 3.0)
                assert isinstance(future, DistributedFuture)
                result = future.result(timeout=None)
                assert result == 5.0
                print(f"result on controller is {result}")
                sys.stdout.flush()

        if is_controller:
            print(f"result on controller is {result}")
            sys.stdout.flush()
            assert result == 5.0

    # def test_task_with_kwargs(self, config, is_controller):
    #     """Test task with keyword arguments."""
    #     result = None

    #     with DistributedEnvironment(config) as env:
    #         for controller in env.controller_session():
    #             future = controller.submit(simple_multiply, 5.0, factor=3.0)
    #             result = future.result(timeout=5.0)

    #     if is_controller:
    #         assert result == 15.0

    # def test_multiple_tasks(self, config, is_controller):
    #     """Test multiple task submission."""
    #     results = []

    #     with DistributedEnvironment(config) as env:
    #         for controller in env.controller_session():
    #             futures = []
    #             for i in range(5):
    #                 future = controller.submit(simple_add, float(i), 10.0)
    #                 futures.append(future)

    #             for future in futures:
    #                 result = future.result(timeout=5.0)
    #                 results.append(result)

    #     if is_controller:
    #         assert len(results) == 5
    #         assert results == [10.0, 11.0, 12.0, 13.0, 14.0]

    # def test_batch_submit(self, config, is_controller):
    #     """Test batch submission."""
    #     results = []

    #     with DistributedEnvironment(config) as env:
    #         for controller in env.controller_session():
    #             tasks = [
    #                 (simple_add, (1.0, 2.0), {}),
    #                 (simple_multiply, (3.0,), {'factor': 2.0}),
    #                 (simple_add, (4.0, 5.0), {}),
    #             ]

    #             futures = controller.batch_submit(tasks)

    #             for future in futures:
    #                 result = future.result(timeout=5.0)
    #                 results.append(result)

    #     if is_controller:
    #         assert len(results) == 3
    #         assert results[0] == 3.0
    #         assert results[1] == 6.0
    #         assert results[2] == 9.0


# class TestMapOperations:
#     """Test map operations."""

#     def test_map(self, config, is_controller):
#         """Test map operation."""
#         results = None

#         with DistributedEnvironment(config) as env:
#             for controller in env.controller_session():
#                 data = [1.0, 2.0, 3.0, 4.0]
#                 results = list(controller.map(simple_multiply, data))

#         if is_controller:
#             assert results == [2.0, 4.0, 6.0, 8.0]

#     def test_imap_unordered(self, config, is_controller):
#         """Test imap_unordered operation."""
#         results = []

#         with DistributedEnvironment(config) as env:
#             for controller in env.controller_session():
#                 data = [1.0, 2.0, 3.0]

#                 for result in controller.imap_unordered(simple_multiply, data):
#                     results.append(result)

#         if is_controller:
#             assert len(results) == 3
#             assert sorted(results) == [2.0, 4.0, 6.0]


# class TestParameterSweep:
#     """Test parameter sweep workflow."""

#     def test_parameter_sweep(self, config, is_controller):
#         """Test parameter sweep."""
#         results = []

#         with DistributedEnvironment(config) as env:
#             for controller in env.controller_session():
#                 param_sets = [
#                     {'tau_m': 15.0, 'tau_s': 3.0},
#                     {'tau_m': 20.0, 'tau_s': 5.0},
#                     {'tau_m': 25.0, 'tau_s': 7.0},
#                 ]

#                 futures = []
#                 for params in param_sets:
#                     future = controller.submit(test_model, **params)
#                     futures.append(future)

#                 for future in futures:
#                     result = future.result(timeout=5.0)
#                     results.append(result)

#         if is_controller:
#             assert len(results) == 3
#             for result in results:
#                 assert 'tau_m' in result
#                 assert 'tau_s' in result
#                 assert 'firing_rate' in result


# @pytest.mark.parametrize("n_tasks", [1, 5, 10])
# def test_varying_task_counts(config, is_controller, n_tasks):
#     """Test with varying numbers of tasks."""
#     results = []

#     with DistributedEnvironment(config) as env:
#         for controller in env.controller_session():
#             futures = []
#             for i in range(n_tasks):
#                 future = controller.submit(simple_add, float(i), 1.0)
#                 futures.append(future)

#             for future in futures:
#                 result = future.result(timeout=5.0)
#                 results.append(result)

#     if is_controller:
#         assert len(results) == n_tasks
#         expected = [float(i + 1) for i in range(n_tasks)]
#         assert results == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
