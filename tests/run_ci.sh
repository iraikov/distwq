#!/bin/bash
# Run the whole distwq test suite as the CI does: once in one process, then
# under MPI at several world sizes, then the spawned worker groups.
#
# mpi4py must load an MPI library that can spawn processes, such as Open MPI
# (MPI4PY_MPIABI=openmpi).  MPIRUN gives the launcher command; its default
# suits a small machine with few cores.
set -euo pipefail

MPIRUN=${MPIRUN:-"mpirun --oversubscribe"}
PYTHON=$(command -v python)
cd "$(dirname "$0")/.."

# The tests that start their own MPI jobs read the launcher from here.
export DISTWQ_MPIRUN=$MPIRUN

echo "== one process"
timeout 1200 "$PYTHON" -m pytest -rs tests

# World sizes 2, 3, 4, 5 and 7 together fit every worker-group layout the
# tests use.  The interpreter is given by its full path, because some
# launchers change PATH on the ranks they start.
for n in 2 3 4 5 7; do
    echo "== MPI, $n ranks"
    timeout 900 $MPIRUN -x MPI4PY_MPIABI -n "$n" "$PYTHON" -m pytest -rs tests
done

# Spawned processes start with a fresh sys.path, so tests/ goes on
# PYTHONPATH for them to import the test modules.
for n in 3 5; do
    echo "== spawned worker groups, $n ranks"
    DISTWQ_TEST_SPAWN=1 PYTHONPATH="$PWD/tests${PYTHONPATH:+:$PYTHONPATH}" \
        timeout 600 $MPIRUN -x MPI4PY_MPIABI -x DISTWQ_TEST_SPAWN -x PYTHONPATH \
        -n "$n" "$PYTHON" -m pytest -rs -k spawn \
        tests/test_init_hooks_mpi.py tests/test_worker_grouping.py tests/test_session_protocol.py
done
