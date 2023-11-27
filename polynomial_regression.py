import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


raw_data = pd.read_csv("./chocolate_bars.csv")

raw_x = torch.tensor(raw_data['cocoa_percent'].values, dtype=torch.float)
mean_x = torch.mean(raw_x)
std_x = torch.std(raw_x)
norm_x = (raw_x - mean_x)/std_x  # mean normalization
x = norm_x.reshape(-1, 1)  # input

raw_y = torch.tensor(raw_data['rating'].values, dtype=torch.float)
y = raw_y.reshape(-1, 1)  # true output


class PolynomialRegression(torch.nn.Module):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.polynom = torch.nn.Linear(
            3, 1, bias=True)  # 2 inputs: w1x1 + w2(x1^2) + b

    def forward(self, x):
        # contcat along dim=1
        y_pred = self.polynom(torch.cat([x, torch.square(x), x**3], dim=1))
        return y_pred


model = PolynomialRegression()
criterion = torch.nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_array = []
# w_array = []

for epoch in range(10000):
    # forward pass
    pred_y = model(x)
    loss = criterion(pred_y, y)

    optimizer.zero_grad()  # wipe out prev gradient
    loss.backward()
    optimizer.step()  # update params w & b

    # for params in model.named_parameters():
    #     if params[0] != "linear.weight":
    #         continue
    #     w_array.append(params[1].item())

    if epoch % 1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    loss_array.append(loss.item())

plt.yscale("log")
plt.plot(loss_array)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("LOSS")
plt.show()

# plt.plot(w_array)
# plt.title("WEIGHT")
# plt.show()

plt.plot(x, y, 'bo', label="actual")
plt.plot(x, model(x).detach().numpy(), 'rx', label="predicted")
plt.title("TRUE vs. PRED")
plt.xlabel("cocoa percentage")
plt.ylabel("rating")
plt.legend()
plt.show()
