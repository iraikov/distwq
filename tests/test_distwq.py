import numpy as np

import distwq


def do_work(x):
    return x**2


def init(worker):
    pass


def main(controller):
    n = 5
    ans = 0
    for i in range(0, n):
        x = i + 1
        controller.submit_call("do_work", (x,), module_name="test_distwq")
        ans += x**2
    s = []
    for i in range(0, n):
        _, res = controller.get_next_result()
        s.append(res)
    print(f"s = {s} ans = {ans}")
    assert np.sum(s) == ans
    controller.info()


def test_basic():
    if distwq.is_controller:
        distwq.run(
            fun_name="main",
            module_name="test_distwq",
            verbose=True,
        )
    else:
        distwq.run(
            fun_name="init",
            module_name="test_distwq",
            verbose=True,
        )


_BIG_PAYLOAD_LEN = 200_000


def do_big_work(x):
    return bytes((x + i) % 256 for i in range(_BIG_PAYLOAD_LEN))


def main_big(controller):
    x = 7
    controller.submit_call("do_big_work", (x,), module_name="test_distwq")
    _, res = controller.get_next_result()
    assert res == bytes((x + i) % 256 for i in range(_BIG_PAYLOAD_LEN))
    controller.info()


def test_large_payload_round_trips_across_multiple_messages(monkeypatch):
    """A result larger than one physical MPI message must still arrive
    intact, split transparently across several. Shrinks the per-message
    size limit so a modest payload exercises the same splitting/
    reassembly path a payload too large for a single message would."""
    monkeypatch.setattr(distwq, "_MAX_MSG_BYTES", 4096)
    if distwq.is_controller:
        distwq.run(
            fun_name="main_big",
            module_name="test_distwq",
            verbose=True,
        )
    else:
        distwq.run(
            fun_name="init",
            module_name="test_distwq",
            verbose=True,
        )
