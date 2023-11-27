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

coco_pct_raw = torch.tensor(
    cleaned_data['cocoa_percent'].values, dtype=torch.float)
coco_pct_mean = torch.mean(coco_pct_raw)
coco_pct_std = torch.std(coco_pct_raw)
coco_pct_norm = (coco_pct_raw - coco_pct_mean) / \
    coco_pct_std  # mean normalization
coco_pct = coco_pct_norm.reshape(-1, 1)  # input

ingredients = cleaned_data['ingredients']
ing_set = set()
for ing_str in ingredients:
    ing_list = ing_str.split(",")
    for ing in ing_list:
        ing_set.add(ing.strip())

# print(ing_set)

ing_dict = {}

for idx, ing in enumerate(sorted(ing_set)):
    # print(idx, ing)
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

ing_raw = torch.tensor(ing_raw_data, dtype=torch.float)
ing_mean = torch.mean(ing_raw)
ing_std = torch.std(ing_raw)
ing = (ing_raw - ing_mean)/ing_std
# print("shape: ", x_mean.shape, x_std.shape)

x = torch.cat([ing, coco_pct], dim=1)  # 7 ingredients + coco%

raw_y = torch.tensor(cleaned_data['rating'].values, dtype=torch.float)
y = raw_y.reshape(-1, 1)  # true output


class NeuralNetwork(torch.nn.Module):
    # constructor
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # class variable
        # linear regression y = wx +b, w is single param
        # hence torch.nn.Linear(1, 1) means w is single param and output is single param
        # torch.nn.Linear(4, 1) means 4 features 1 output
        self.hidden_size = 100
        self.linear0 = torch.nn.Linear(
            num_ing+1, self.hidden_size, bias=True)  # 7 ingredients + coco%
        self.linear1 = torch.nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(self.hidden_size, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        z0 = self.linear0(x)
        y0 = self.relu(z0)
        z1 = self.linear1(y0)
        y1 = self.relu(z1)
        y_pred = self.linear2(y1)

        return y_pred


model = NeuralNetwork()
criterion = torch.nn.MSELoss()  # Mean Squared Error
# Stochastic Gradient Descent, learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

loss_array = []
for epoch in range(5000):
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

for i in range(10):
    # for y_t, y_pred_t in zip(y, y_pred):
    print(y[i], y_pred[i])

# plt.plot(x, y, 'bo', label="actual")
# plt.plot(x, model(x).detach().numpy(), 'rx', label="predicted")
# plt.title("TRUE vs. PRED")
# plt.xlabel("cocoa percentage")
# plt.ylabel("rating")
# plt.legend()
# plt.show()

coco_ptc_test_raw = [[77], [77], [77], [77], [77], [77], [77],  # single ingredients
                     [77], [77], [77], [77],  # ingredients combos
                     [10], [20], [30]]  # coco variants
coco_ptc_test_raw = torch.tensor(coco_ptc_test_raw)
coco_ptc_test = (coco_ptc_test_raw - coco_pct_mean)/coco_pct_std


ing_test_raw = [[1, 0, 0, 0, 0, 0, 0],  # beans - highly valued ingredient
                [0, 1, 0, 0, 0, 0, 0],  # cocoa butter - also important
                [0, 0, 1, 0, 0, 0, 0],  # Lecithin
                [0, 0, 0, 1, 0, 0, 0],  # sugar - second highest
                [0, 0, 0, 0, 1, 0, 0],  # sweetner
                [0, 0, 0, 0, 0, 1, 0],  # salt
                [0, 0, 0, 0, 0, 0, 1],  # vanilla
                [1, 0, 0, 1, 0, 0, 0],  # beans + sugar  - good combo
                [1, 0, 0, 1, 1, 0, 0],  # beans + sugar + sweetner - good combo
                [1, 1, 0, 1, 0, 0, 0],  # beans + sugar + cocoa butter - good combo
                # beans + sugar + cocoa butter + vanilla - lowest combo
                [1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 30% coco
                [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 60% coco
                [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 90% coco
                ]
ing_test_raw = torch.tensor(ing_test_raw)
ing_test = (ing_test_raw - ing_mean)/ing_std

x_test = torch.cat([ing_test, coco_ptc_test], dim=1)
y_test = model(torch.tensor(x_test, dtype=torch.float))  # y_pred
print(y_test)
