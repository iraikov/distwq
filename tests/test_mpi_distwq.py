import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def do_work(x):
    return x**2


try:
    from distwq.future import DistributedWorkQueue
    from distwq.config import Config, LoggingConfig
    from distwq.mpi import is_controller, rank, world_comm, has_mpi
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def test_basic():
    """Test basic distwq functionality"""

    if not has_mpi or world_comm.size < 2:
        logger.error("Skipping: Need at least 2 MPI processes for distwq test")
        return

    config = Config(logging=LoggingConfig(verbose=True))

    try:
        with DistributedWorkQueue(config) as queue:
            logger.info("Successfully created work queue")

            # Submit a few group tasks
            futures = []
            for i in range(3):
                x = 10 + i * 5
                future = queue.submit(do_work, x)
                futures.append(future)
                logger.info(f"Submitted task {i} (x: {x})")

            # Collect results
            for i, future in enumerate(futures):
                result = future.result(timeout=30)
                logger.info(f"Task {i} result: {result}")
                worker_info = future.worker_info()
                if worker_info:
                    logger.info(f"  Worker info: {worker_info}")

    except Exception as e:
        logger.error(f"Error in basic distwq test: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main test function - only runs on controller"""

    if not is_controller:
        # Workers will be managed by the futures interface
        return

    logger.info("Running distwq tests...")
    logger.info(f"MPI available: {has_mpi}")
    if has_mpi:
        logger.info(f"World size: {world_comm.size}, Rank: {rank}")

    # Run tests
    test_basic()


if __name__ == "__main__":
    main()
