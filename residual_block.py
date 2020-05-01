import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(planes)
        self.convg = nn.Conv2d(inplanes, planes, 1, stride=stride, padding=0, bias=False)
        self.batch_normg = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.batch_norm1(self.conv1(x)))
        f = self.batch_norm1(self.conv1(f))
        g = x if x.shape == f.shape else self.batch_normg(self.convg(x))
        return self.relu(f + g)


"""
if __name__ == '__main__':
    rb = ResidualBlock(inplanes, planes, stride)

    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = rb(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""