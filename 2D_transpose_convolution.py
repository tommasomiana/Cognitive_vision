import random
import torch
from math import floor

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)

oH = (H - 1) * s + kH
oW = (W - 1) * s + kW

out = torch.zeros((n, oC, oH, oW))

for row in range(H):
    out_row = row * s
    for col in range(W):
        out_col = col * s
        this_input = input[:, :, row, col, None, None, None]
        this_kernel = kernel[None, :, :, :, :]
        portion = torch.sum(this_kernel * this_input, dim=1)
        out[:, :, out_row:out_row + kH, out_col:out_col + kW] += portion
