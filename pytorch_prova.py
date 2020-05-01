import torch

device = torch.device('cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.rand(N, D_in)
y = torch.rand(N, D_out)
w1 = torch.rand(D_in, H, requires_grad=True)
w2 = torch.rand(H, D_out, requires_grad=True)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out))

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):

    # --- this operations have been substituted by the model creation
    # h = x.mm(w1)
    # h_relu = h.clamp(min=0)
    # y_pred = h_relu.mm(w2)
    # loss = (y_pred -y).pow(2).sum()
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()

    # --- first version of updating the parameters
    # with torch.no_grad():
        # w1 -= learning_rate * w1.grad
        # w2 -= learning_rate * w2.grad
        # w1.grad.zero_()
        # w2.grad.zero_()
        # --- second version of updating the parameters
        # for param in model.parameters():
        #    param -= learning_rate * param.grad
    # last and higher level parameters updating thanks to optim.Adam
    optimizer.step()
    model.zero_grad()
