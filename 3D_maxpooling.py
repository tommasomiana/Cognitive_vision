import random
import torch
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand(n, iC, T, H, W)

oH = int((H - kH)/s + 1)
oW = int((W - kW)/s + 1)
oT = int((T - kT)/s + 1)

print('input shape: [{}, {}, {}, {}]'.format(n, iC, H, W))
print('s: {}\nkH, kW: {}, {}'.format(s, kH, kW))
print('oH, oW: {}, {}'.format(oH, oW))

out = torch.zeros((n, iC, oT, oH, oW))

t = 0
out_t = 0
while out_t != oT - 1:
    out_t = int(t / s)
    row = 0
    out_row = 0
    while out_row != oH - 1:
        out_row = int(row / s)
        col = 0
        out_col = 0
        while out_col != oW - 1:
            out_col = int(col / s)
            this_input = input[:, :, t:t + kT, row:row + kH, col:col + kW]
            max_input = np.max(this_input.numpy(), (2, 3, 4))
            out[:, :, out_t, out_row, out_col] = torch.from_numpy(max_input)
            col = col + s
            # print('x, y : {}, {} ---> ix, iy : {}, {}'.format(row, col, out_row, out_col))
        row = row + s
    t = t +s
out1 = torch.max_pool3d(input, [kT, kH, kW], stride=s)
print(out[0, 0, 0, 0])
print(out1[0, 0, 0, 0])

#skimage.measure.block_reduce(input, (kH, kW), np.max)