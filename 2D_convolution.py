import random
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
iH = random.randint(10, 20)
iW = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = np.random.rand(n, iC, iH, iW)
kernel = np.random.rand(oC, iC, kH, kW)

oH = iH - kH + 1
oW = iW - kW + 1

out = np.zeros((n, oC, oH, oW))

for row in range(oH):
    for col in range(oW):
        # input[:, :, row:row+kH, col:col+kW] ==> (n,  iC, kH, kW)
        # kernel ==>                              (oC, iC, kH, kW)

        this_input = np.expand_dims(input[:, :, row:row+kH, col:col+kW], 1)
        # (n, 1,  iC, kH, kW)
        this_kernel = np.expand_dims(kernel, 0)
        # (1, oC, iC, kH, kW)

        # this_input * this_kernel
        #     (n, 1,  iC, kH, kW)
        #     (1, oC, iC, kH, kW)
        # ==> (n, oC, iC, kH, kW)

        # np.sum(tensor) = np.sum(tensor, axis=[all axes]) ==> scalar

        # Summation over the last three axes
        # np.sum(np.sum(np.sum(.., axis=-1), axis=-1), axis=-1)
        # np.sum(.., axis=(-1, -2, -3))
        # ==> (n, oC)

        out[:, :, row, col] = np.sum(this_input * this_kernel, axis=(-1, -2, -3))
        # out[:, :, row, col] ==> (n, oC)

