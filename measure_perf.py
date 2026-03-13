import time
import numpy as np

def test_optimization():
    # Test loop overhead of zip vs formatting in loop
    n, m = 1000, 10000
    H = np.random.rand(n, m) < 0.01

    v_idx, e_idx = np.nonzero(H)

    start_time = time.time()
    memberships2 = [(f"v{v}", f"e{e}") for v, e in zip(v_idx, e_idx)]
    time2 = time.time() - start_time

    start_time = time.time()
    memberships3 = []
    for i in range(len(v_idx)):
        memberships3.append((f"v{v_idx[i]}", f"e{e_idx[i]}"))
    time3 = time.time() - start_time

    assert memberships2 == memberships3
    print(f"zip: {time2:.4f}s")
    print(f"range: {time3:.4f}s")

if __name__ == '__main__':
    test_optimization()
