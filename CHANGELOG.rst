Changelog
=========

2.0.0
------------------

Added
~~~~~

- The task interface, ``distwq.future``. ``DistributedEnvironment`` sets
  up the controller and workers on every rank, and ``controller_session()``
  gives the controller a ``TaskController``. Its ``submit()``, ``map()``,
  ``imap_unordered()`` and ``batch_submit()`` take functions and return
  futures. ``wait()`` and ``as_completed()`` wait for them, and
  ``broadcast()`` sends a value from the controller to every rank. A task's
  exception is raised by ``result()`` on the controller, with the worker's
  traceback.
- ``DistributedEnvironment`` takes ``worker_init``, ``broker_init`` and
  ``init_args`` to set up the worker processes, and has a ``server``
  property.
- ``distwq.functional`` (``map_distributed``, ``map_reduce``,
  ``pipeline``, ``run_session``, ``quick_map`` and others) and
  ``distwq.workflow`` (``parameter_sweep``,
  ``adaptive_parameter_exploration``) build on sessions.
- ``TaskResult`` and ``CollectiveTaskResult`` tag each result from a
  single worker or a worker group.
- ``DistributedEnvironment`` applies ``Config.resources.max_workers``.
  ``limit_workers()`` sets the same limit for ``run()``.
- ``TaskController.stats`` and ``TaskController.info()``;
  ``MPIController.summary()`` returns the text that ``info()`` prints.

Changed
~~~~~~~

- distwq is now a package: ``distwq.mpi`` holds the controller/worker
  interface, which ``import distwq`` still exposes as before.
- ``get_result()``, ``get_next_result()``, ``probe_next_result()`` and
  ``probe_all_next_results()`` return a ``TaskResult`` or
  ``CollectiveTaskResult``, not a bare value or list. Read ``.value`` or
  ``.values``.
- If the controller function given to ``run()`` raises, the workers are
  released and the exception is raised again on the controller. Before,
  the job could stop with exit status 0 or hang.
- An init or worker function that raises stops the whole job with exit
  status 1. ``abort()`` also exits with status 1.
- ``WorkerGroupConfig`` checks a split layout when the environment is
  entered, against the worker ranks actually used.
- ``DistributedEnvironment`` raises ``TypeError`` if ``config`` is not a
  ``Config`` or ``worker_config`` is not a ``WorkerGroupConfig``.
- ``get_result()`` for a call whose result was already obtained, or that
  was never submitted, raises ``KeyError`` with a message saying so.
  ``get_next_result()`` is documented as returning the result that
  arrived first, which need not be the earliest submitted call.

Removed
~~~~~~~

- The ``Config`` settings that distwq never applied: ``ExecutionConfig``
  (timeouts, retries, ``fail_fast``, ``progress_callback``),
  ``MPIConfig``, ``SpawnMethod``, ``Config.grouping``, the
  ``ResourceConfig`` fields other than ``max_workers``, the
  ``LoggingConfig`` fields other than ``verbose``, ``Config.with_timeout()``,
  and the presets ``Config.for_development()`` and
  ``Config.for_production()``. Giving one now raises ``TypeError``. Worker
  groups are set with ``worker_config``.
- The time limit, which distwq never enforced: ``run()`` raises
  ``ValueError`` if ``time_limit`` is not None, and ``MPIController`` no
  longer takes ``time_limit``.

Fixed
~~~~~

- ``wait()``, ``as_completed()`` and ``result()`` no longer hang when the
  controller has already received the result.
- ``max_workers``: the limit is computed afresh on every call, it keeps
  the right number of workers when the controller is not rank 0, and it
  no longer breaks single-rank runs or valid split layouts.
- Worker groups keep the results of processes that return ``None``, and no
  longer mistake a returned dict for a failure.
- A broker that only relays calls no longer runs the worker init function.
- Split worker groups with a broker that only relays calls no longer hang
  under Open MPI 5. The broker is now always rank 0 of its group, where
  the workers expect it.
- ``run()`` restores the previous ``SIGINT`` handler when the controller
  function raises. ``MPIController.info()`` counts the brokers under
  worker grouping.

1.3.4
-----

Changed
~~~~~~~

- distwq requires Python 3.11 or later.

Fixed
~~~~~

- Collective mode no longer hangs at exit (issue 26): a broker now leaves
  the worker group together with its workers.
