import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

raw_data = pd.read_csv("./chocolate_bars.csv")
cleaned_data = raw_data.dropna()  # cleaned
# print(cleaned_data.shape)
# print(cleaned_data.isnull().sum())
# print(len(cleaned_data))

ingredients = cleaned_data['ingredients']
ing_set = set()
for ing_str in ingredients:
    ing_list = ing_str.split(",")
    for ing in ing_list:
        ing_set.add(ing.strip())

print(ing_set)

ing_dict = {}

for idx, ing in enumerate(sorted(ing_set)):
    print(idx, ing)
    ing_dict[ing] = idx

# print(ing_dict)  # ingredient to number mapping

num_rows = len(cleaned_data)
num_ing = len(ing_set)
ing_raw_data = np.zeros((num_rows, num_ing), dtype=float)

for row_num, ing_str in enumerate(ingredients):
    ing_list = ing_str.split(",")
    for ing in ing_list:
        idx = ing_dict[ing.strip()]
        ing_raw_data[row_num][idx] += 1
# print(ing_raw_data[53])

x = torch.tensor(ing_raw_data, dtype=torch.float)

raw_y = torch.tensor(cleaned_data['rating'].values, dtype=torch.float)
y = raw_y.reshape(-1, 1)  # true output


class LinearRegression(torch.nn.Module):
    # constructor
    def __init__(self):
        super(LinearRegression, self).__init__()
        # class variable
        # linear regression y = wx +b, w is single param
        # hence torch.nn.Linear(1, 1) means w is single param and output is single param
        # torch.nn.Linear(4, 1) means 4 features 1 output
        self.linear = torch.nn.Linear(num_ing, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearRegression()
criterion = torch.nn.MSELoss()  # Mean Squared Error
# Stochastic Gradient Descent, learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_array = []

for epoch in range(10000):
    # forward pass
    pred_y = model(x)
    loss = criterion(pred_y, y)

    optimizer.zero_grad()  # wipe out prev gradient
    loss.backward()
    optimizer.step()  # update params w & b

    if epoch % 1000 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    loss_array.append(loss.item())

plt.yscale("log")
plt.plot(loss_array)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("LOSS")
plt.show()

y_pred = model(x).detach().numpy()

# for y_t, y_pred_t in zip(y, y_pred):
#     print(y_t, y_pred_t)

# plt.plot(x, y, 'bo', label="actual")
# plt.plot(x, model(x).detach().numpy(), 'rx', label="predicted")
# plt.title("TRUE vs. PRED")
# plt.xlabel("cocoa percentage")
# plt.ylabel("rating")
# plt.legend()
# plt.show()
