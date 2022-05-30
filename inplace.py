import time

import numpy as np


if __name__ == "__main__":
    a = np.random.rand(2048, 2048)
    b = np.random.rand(2048, 2048)
    c = np.zeros((2048, 2048))

    iters = 5

    start = time.time()
    for i in range(iters):
        c = np.dot(a, b)
    end = time.time()

    print(f"np time: {end-start}")

    start = time.time()
    for i in range(iters):
        np.matmul(a, b, out=c)
    end = time.time()

    print(f"np inline time: {end-start}")
