"""
Check whether MPI_Comm_spawn works with the current MPI library and launcher.

This is a diagnostic script, not a pytest module.  Each parent rank spawns
children from MPI.COMM_SELF, as each distwq broker does under GroupSpawn,
merges the intercommunicator, checks an allreduce over the merged
communicator, and then leaves the group as chosen by --disconnect.  It
reports which MPI library mpi4py loaded and the PMIx and launcher
environment, so that a failure can be traced to its layer.

mpi4py can load either MPI library; MPI4PY_MPIABI selects one:

    # MPICH (the default here): fails in MPI_Comm_spawn
    /usr/bin/mpirun.openmpi -n 2 python tests/spawn_probe.py

    # Open MPI: works
    MPI4PY_MPIABI=openmpi /usr/bin/mpirun.openmpi -x MPI4PY_MPIABI -n 2 \
        python tests/spawn_probe.py

    # Open MPI, leaving the group through the merged communicator: hangs
    MPI4PY_MPIABI=openmpi /usr/bin/mpirun.openmpi -x MPI4PY_MPIABI -n 2 \
        python tests/spawn_probe.py --disconnect merged

A watchdog ends the process with a traceback after --timeout seconds, so
that a call that never returns does not hang the terminal.  Output from
the spawned children can contain NUL bytes; use grep -a on saved output.
"""

import argparse
import faulthandler
import logging
import os
import sys

logger = logging.getLogger("spawn_probe")

CHILD_FLAG = "--child"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nchildren", type=int, default=2)
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="watchdog in seconds"
    )
    parser.add_argument(
        "--info",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="MPI_Info entry passed to Spawn; may be repeated",
    )
    parser.add_argument(
        "--disconnect",
        choices=["intercomm", "merged", "none"],
        default="intercomm",
        help=(
            "which communicator both sides disconnect before the children "
            "call MPI_Finalize: the intercommunicator from Spawn (default), "
            "the merged intracommunicator, as distwq does, or neither"
        ),
    )
    parser.add_argument(
        CHILD_FLAG, action="store_true", help=argparse.SUPPRESS, dest="child"
    )
    return parser.parse_args(argv)


def describe_environment(MPI):
    library = MPI.Get_library_version().strip().splitlines()[0]
    logger.info("MPI library: %s", library)
    logger.info("MPI standard version: %s.%s", *MPI.Get_version())
    universe = MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE)
    logger.info("world size %d, universe size %s", MPI.COMM_WORLD.Get_size(), universe)
    for key in sorted(os.environ):
        if key.startswith(("PMIX_", "OMPI_COMM_WORLD", "PMI_", "MPI4PY_")):
            logger.info("env %s=%s", key, os.environ[key])


def disconnect(merged, intercomm, which):
    """Disconnect one of the two communicators that join parent and children."""
    logger.info("disconnecting: %s", which)
    if which == "merged":
        merged.Disconnect()
    else:
        merged.Free()
        if which == "intercomm":
            intercomm.Disconnect()
    logger.info("disconnected: %s", which)


def run_parent(MPI, args):
    world = MPI.COMM_WORLD
    if world.Get_rank() == 0:
        describe_environment(MPI)
    info = MPI.Info.Create()
    for item in args.info:
        key, _, value = item.partition("=")
        info.Set(key, value)
    logger.info(
        "parent %d: spawning %d children (info %s)",
        world.Get_rank(),
        args.nchildren,
        args.info,
    )
    intercomm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=[os.path.abspath(__file__), CHILD_FLAG, "--disconnect", args.disconnect],
        maxprocs=args.nchildren,
        info=info,
    )
    info.Free()
    merged = intercomm.Merge(False)
    total = merged.allreduce(merged.Get_rank())
    expected = sum(range(args.nchildren + 1))
    logger.info(
        "parent %d: merged size %d, allreduce %d (expected %d)",
        world.Get_rank(),
        merged.Get_size(),
        total,
        expected,
    )
    disconnect(merged, intercomm, args.disconnect)
    ok = total == expected and world.allreduce(1) == world.Get_size()
    world.Barrier()
    if world.Get_rank() == 0:
        logger.info("SPAWN %s", "OK" if ok else "FAILED")
    return 0 if ok else 1


def run_child(MPI, args):
    parent = MPI.Comm.Get_parent()
    if parent == MPI.COMM_NULL:
        logger.error("child started without a parent communicator")
        return 1
    merged = parent.Merge(True)
    merged.allreduce(merged.Get_rank())
    disconnect(merged, parent, args.disconnect)
    logger.info("child: calling MPI_Finalize")
    MPI.Finalize()
    logger.info("child: MPI_Finalize returned")
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    role = "child" if args.child else "parent"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {role} pid {os.getpid()}: %(message)s",
    )
    faulthandler.dump_traceback_later(args.timeout, exit=True)
    from mpi4py import MPI

    status = run_child(MPI, args) if args.child else run_parent(MPI, args)
    faulthandler.cancel_dump_traceback_later()
    return status


if __name__ == "__main__":
    sys.exit(main())
