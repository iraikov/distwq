#!/usr/bin/env python3
"""
Test script for worker grouping functionality in the futures interface.

Run with: mpirun -n 8 python test_worker_grouping.py
"""

import sys
import time

import numpy as np

# Add current directory to path for imports
sys.path.insert(0, ".")

try:
    from config import Config
    from future import (
        DistributedTaskExecution,
        DistributedWorkQueue,
        GroupingMethod,
        WorkerGroupConfig,
        create_spawn_queue,
        create_split_queue,
    )
    from mpi import has_mpi, is_controller, rank, world_comm
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def simple_task(x: int) -> int:
    """Simple task for testing regular workers"""
    time.sleep(0.1)  # Simulate some work
    return x * x


def collective_matrix_multiply(matrix_size: int) -> dict:
    """
    Simulated collective matrix multiplication task.
    """
    # Simulate work that benefits from multiple processes
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    # Simulate collective computation
    time.sleep(0.2)  # Simulate work
    result = np.dot(matrix_a, matrix_b)

    return {
        "matrix_size": matrix_size,
        "result_shape": result.shape,
        "trace": np.trace(result),
        "max_value": np.max(result),
        "worker_group": True,
    }


def parallel_simulation(params: dict) -> dict:
    """
    Simulated parallel simulation that benefits from worker groups.
    """
    simulation_id = params["id"]
    complexity = params.get("complexity", 1)

    # Simulate parallel computation
    time.sleep(0.1 * complexity)

    return {
        "simulation_id": simulation_id,
        "complexity": complexity,
        "result": simulation_id * complexity + np.random.randint(0, 100),
        "processed_by_group": True,
    }


def test_basic_worker_grouping():
    """Test basic worker grouping functionality"""
    print("=== Testing Basic Worker Grouping ===")

    if not has_mpi or world_comm.size < 4:
        print("Skipping: Need at least 4 MPI processes for worker grouping test")
        return

    # Configure worker groups
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=2,
        broker_is_worker=True,
        collective_mode="gather",
    )

    config = Config()
    config.verbose = True

    try:
        with DistributedWorkQueue(config, worker_config) as queue:
            print("Successfully created work queue with worker grouping")

            # Submit a few group tasks
            futures = []
            for i in range(3):
                matrix_size = 10 + i * 5
                future = queue.submit_to_group(collective_matrix_multiply, matrix_size)
                futures.append(future)
                print(f"Submitted matrix multiplication task {i} (size: {matrix_size})")

            # Collect results
            for i, future in enumerate(futures):
                result = future.result(timeout=30)
                print(f"Task {i} result: {result}")
                print(f"  Is group task: {future.is_group_task()}")
                worker_info = future.worker_info()
                if worker_info:
                    print(f"  Worker info: {worker_info}")

    except Exception as e:  # noqa: BLE001
        print(f"Error in basic worker grouping test: {e}")
        import traceback

        traceback.print_exc()


def test_split_grouping():
    """Test split-based worker grouping"""
    print("\n=== Testing Split-based Worker Grouping ===")

    if not has_mpi or world_comm.size < 6:
        print("Skipping: Need at least 6 MPI processes for split grouping test")
        return

    try:
        # Use convenience function
        with create_split_queue(nprocs_per_worker=2, broker_is_worker=True) as queue:
            print("Successfully created split-based work queue")

            # Submit simulation tasks
            param_sets = [
                {"id": 1, "complexity": 1},
                {"id": 2, "complexity": 2},
                {"id": 3, "complexity": 1},
            ]

            futures = []
            for params in param_sets:
                future = queue.submit_to_group(parallel_simulation, params)
                futures.append(future)
                print(f"Submitted simulation task {params['id']}")

            # Process results as they complete
            from future import as_completed

            for future in as_completed(futures, timeout=30):
                result = future.result()
                print(f"Simulation result: {result}")

    except Exception as e:  # noqa: BLE001
        print(f"Error in split grouping test: {e}")
        import traceback

        traceback.print_exc()


def test_mixed_workload():
    """Test mixed regular and grouped tasks"""
    print("\n=== Testing Mixed Workload ===")

    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn
        if has_mpi and world_comm.size >= 4
        else GroupingMethod.NoGrouping,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )

    config = Config()
    config.verbose = True

    try:
        with DistributedTaskExecution(config, worker_config) as executor:
            print("Testing mixed workload...")

            # Submit regular tasks
            regular_futures = []
            for i in range(3):
                future = executor.submit(simple_task, i + 1)
                regular_futures.append(future)

            # Submit grouped tasks if available
            group_futures = []
            if worker_config.grouping_method != GroupingMethod.NoGrouping:
                for i in range(2):
                    future = executor.submit(collective_matrix_multiply, 8 + i * 2)
                    group_futures.append(future)

            # Process all results
            all_futures = regular_futures + group_futures
            for i, future in enumerate(all_futures):
                result = future.result(timeout=30)
                task_type = "group" if future.is_group_task() else "regular"
                print(f"{task_type} task {i} result: {result}")

    except Exception as e:  # noqa: BLE001
        print(f"Error in mixed workload test: {e}")
        import traceback

        traceback.print_exc()


def test_map_with_groups():
    """Test map operations with worker groups"""
    print("\n=== Testing Map with Worker Groups ===")

    if not has_mpi or world_comm.size < 4:
        print("Skipping: Need at least 4 MPI processes for map with groups test")
        return

    try:
        with create_spawn_queue(nprocs_per_worker=2) as queue:
            print("Testing map with worker groups...")

            # Test map with worker groups
            matrix_sizes = [5, 6, 7, 8]
            group_results = list(queue.map(collective_matrix_multiply, matrix_sizes))
            print(f"Group map completed {len(group_results)} tasks")
            for i, result in enumerate(group_results):
                print(
                    f"  Task {i}: size {result['matrix_size']}, trace {result['trace']:.2f}"
                )

    except Exception as e:  # noqa: BLE001
        print(f"Error in map with groups test: {e}")
        import traceback

        traceback.print_exc()


def test_configuration_validation():
    """Test configuration validation"""
    print("\n=== Testing Configuration Validation ===")

    # Test invalid configuration
    try:
        invalid_config = WorkerGroupConfig(
            grouping_method=GroupingMethod.GroupSpawn,
            nprocs_per_worker=1,
            broker_is_worker=True,
        )
        print("ERROR: Invalid configuration was accepted!")
        print(invalid_config)
    except Exception as e:  # noqa: BLE001
        print(f"Invalid configuration correctly rejected: {e}")

    # Test valid configuration
    try:
        valid_config = WorkerGroupConfig(
            grouping_method=GroupingMethod.GroupSpawn,
            nprocs_per_worker=2,
            broker_is_worker=True,
        )
        print("Valid configuration accepted")
        print(valid_config)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Valid configuration rejected: {e}")


def main():
    """Main test function - only runs on controller"""

    if not is_controller:
        # Workers will be managed by the futures interface
        return

    print("Running worker grouping tests...")
    print(f"MPI available: {has_mpi}")
    if has_mpi:
        print(f"World size: {world_comm.size}, Rank: {rank}")
    print()

    # Run tests
    test_configuration_validation()
    test_basic_worker_grouping()
    test_split_grouping()
    test_mixed_workload()
    test_map_with_groups()

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
