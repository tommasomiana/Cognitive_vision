import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, T, H, W)
kernel = torch.rand(oC, iC, kT, kH, kW)
bias = torch.rand(oC)

oH = H - kH + 1
oW = W - kW + 1
oT = T - kT + 1

out = torch.zeros((n, oC, oT, oH, oW))

for t in range(oT):
    for row in range(oH):
        for col in range(oW):
            # input[:, :, t:t+kT, row:row+kH, col:col+kW] ==> (n, iC, kT, kH, kW)
            # kernel ==>                                      (oC, iC, kT, kH, kW)

            this_input = input[:, :, t:t+kT, row:row+kH, col:col+kW].unsqueeze(1)
            # (n, 1,  iC, kH, kW)
            this_kernel = kernel.unsqueeze(0)
            # (1, oC, iC, kH, kW)

            # this_input * this_kernel
            #     (n, 1,  iC, kT, kH, kW)
            #     (1, oC, iC, kT, kH, kW)
            # ==> (n, oC, iC, kT, kH, kW)

            # Summation over the last three axes
            # np.sum(np.sum(np.sum(.., axis=-1), axis=-1), axis=-1)
            # np.sum(.., axis=(-1, -2, -3))
            # ==> (n, oC)

            out[:, :, t, row, col] = torch.sum(this_input * this_kernel, (-1, -2, -3, -4))
            # out[:, :, t, row, col] ==> (n, oC)

out1 = torch.conv3d(input, kernel, bias)
for b in range(oC):
    out[:, b, :, :, :] = out[:, b, :, :, :].add(bias[b])

