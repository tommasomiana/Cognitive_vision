import random
import numpy as np

n = random.randint(1, 3)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
input = np.random.rand(n, H, W).astype(np.float32)
template = np.random.rand(kH, kW).astype(np.float32)

oH = H - kH + 1
oW = W - kW + 1

out = np.zeros((n, oH, oW))

for row in range(oH):
    for col in range(oW):
        this_input = input[:, row:row+kH, col:col+kW]
        out[:, row, col] = np.sum((this_input - template)**2, axis=(-1, -2))


