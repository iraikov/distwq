"""
Entry point for spawned worker processes.

A broker that spawns a worker group starts each new process with
``python -c "...; runpy.run_module('distwq', run_name='__main__')"``,
after appending ``- distwq:spawned <json config>`` to ``sys.argv`` (see
:func:`distwq.mpi.do_spawn_workers`).  Running the package executes this
module, which hands control to :func:`distwq.mpi.worker_main`.

The package is run, rather than :mod:`distwq.mpi` itself, because the
package imports :mod:`distwq.mpi` first; running that module as
``__main__`` would then load a second copy of it.
"""

import sys

from distwq import mpi

if __name__ == "__main__":
    if not mpi.spawned:
        sys.exit(
            "distwq: 'python -m distwq' only starts spawned worker processes; "
            "import distwq and call distwq.run() instead"
        )
    mpi.worker_main()
