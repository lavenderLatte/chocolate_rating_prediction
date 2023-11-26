import torch
import numpy as np
import matplotlib.pyplot as plt

# x = torch.randn(100, 1)  # feature
x_raw = torch.rand(10, 1)  # feature
# print(torch.mean(x_raw))
x = x_raw - torch.mean(x_raw)
y = 0.6 * x_raw + 5  # output
# print(x)
# print(y)

# plt.plot(x, y)
# plt.show()


class LinearRegression(torch.nn.Module):
    # constructor
    def __init__(self):
        super(LinearRegression, self).__init__()
        # class variable
        # linear regression y = wx +b, w is single param
        # hence torch.nn.Linear(1, 1) means w is single param and output is single param
        # torch.nn.Linear(4, 1) means 4 features 1 output
        self.linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearRegression()
criterion = torch.nn.MSELoss()  # Mean Squared Error
# Stochastic Gradient Descent, learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_array = []
param_array = []
w_array = []

for epoch in range(10000):
    # forward pass
    pred_y = model(x)
    loss = criterion(pred_y, y)

    optimizer.zero_grad()  # wipe out prev gradient
    loss.backward()
    optimizer.step()  # update params w & b

    for params in model.named_parameters():
        if params[0] != "linear.weight":
            continue
        w_array.append(params[1].item())

    if epoch % 1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    loss_array.append(loss.item())

plt.yscale("log")
plt.plot(loss_array)
plt.show()

plt.plot(w_array)
plt.show()

plt.plot(x, y, 'b')
plt.plot(x, model(x).detach().numpy(), 'go')
plt.show()
