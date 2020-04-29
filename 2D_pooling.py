import random
import numpy as np

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(1, 2)
input = np.random.rand(n, iC, H, W)

oH = int((H - kH)/s + 1)
oW = int((W - kW)/s + 1)

print('input shape: [{}, {}, {}, {}]'.format(n, iC, H, W))
print('s: {}\nkH, kW: {}, {}'.format(s, kH, kW))
print('oH, oW: {}, {}'.format(oH, oW))

out = np.random.rand(n, iC, oH, oW)

row = 0
while True:
    out_row = int(row / s)
    col = 0
    out_col = 0
    while True:
        out_col = int(col / s)
        this_input = input[:, :, row:row + kH, col:col + kW]
        out[:, :, out_row, out_col] = np.max(this_input, axis=(2, 3))
        if out_col == oW - 1:
            break
        col = col + s
        print('x, y : {}, {} ---> ix, iy : {}, {}'.format(row, col, out_row, out_col))
    if out_row == oH - 1:
        break
    row = row + s
print(out)

#skimage.measure.block_reduce(input, (kH, kW), np.max)