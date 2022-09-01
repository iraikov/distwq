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
            spawn_workers=False,
        )
    else:
        distwq.run(
            fun_name="init",
            module_name="test_distwq",
            spawn_workers=False,
            verbose=True,
        )
