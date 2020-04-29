import torch
import numpy as np
import time

d = 3000
a = np.random.rand(d, d)
b = np.random.rand(d, d)
start = time.time()
c = np.matmul(a, b)
end = time.time()
print(f"elapsed time numpy: {end-start}")

cpu = torch.device('cpu')
tensor = torch.Tensor(5, 3)
tensor.numpy()
a = torch.rand(d, d, device=cpu)
b = torch.rand(d, d, device=cpu)
start = time.time()
c = torch.mm(a, b)
end = time .time()
print(f"elapsed time pytorch: {end-start}")
