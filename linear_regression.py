import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


raw_data = pd.read_csv("./chocolate_bars.csv")
# print(raw_data.shape)  # (2530, 11)
# print(raw_data.head().T)
# print(raw_data.isnull().sum())  # 87 null num_ingredients & ingredients

raw_x = torch.tensor(raw_data['cocoa_percent'].values, dtype=torch.float)
mean_x = torch.mean(raw_x)
std_x = torch.std(raw_x)
norm_x = (raw_x - mean_x)/std_x  # mean normalization
x = norm_x.reshape(-1, 1)  # input

raw_y = torch.tensor(raw_data['rating'].values, dtype=torch.float)
# mean_y = torch.mean(raw_y)
# std_y = torch.std(raw_y)
# norm_y = (raw_y - mean_y)/std_y
y = raw_y.reshape(-1, 1)  # true output


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
# Stochastic Gradient Descent, learning rate = 0.01, 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# Adaptive Moment Estimation, learning rate = 0.01, 0.005
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

loss_array = []
w_array = []

for epoch in range(2000):
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

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    loss_array.append(loss.item())

plt.yscale("log")
plt.plot(loss_array)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("LOSS")
plt.show()

plt.plot(x, y, 'bo', label="actual")
plt.plot(x, model(x).detach().numpy(), 'rx', label="predicted")
plt.title("TRUE vs. PRED")
plt.xlabel("cocoa percentage")
plt.ylabel("rating")
plt.legend()
plt.show()
