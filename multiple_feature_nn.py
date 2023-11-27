import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from text_to_onehotvect import onehotEncode, onehotEncode_fortest

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
ing_raw_data, num_ing, ing_dict = onehotEncode(ingredients)
ing_raw = torch.tensor(ing_raw_data, dtype=torch.float)
ing_mean = torch.mean(ing_raw)
ing_std = torch.std(ing_raw)
ing = (ing_raw - ing_mean)/ing_std
# print("shape: ", x_mean.shape, x_std.shape)

borigin = cleaned_data['bean_origin']
borigin_raw_data, num_borigin, borigin_dict = onehotEncode(borigin)
borigin_raw = torch.tensor(borigin_raw_data, dtype=torch.float)
borigin_mean = torch.mean(borigin_raw)
borigin_std = torch.std(borigin_raw)
borigin = (borigin_raw - borigin_mean)/borigin_std
# print("num_borigin: ", num_borigin)
# print("borigin: ", borigin)


x = torch.cat([ing, coco_pct, borigin], dim=1)  # 7 ingredients + coco%

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
            num_ing+1+num_borigin, self.hidden_size, bias=True)  # 7 ingredients + coco% + bean origin
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
for epoch in range(2000):
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

# coco_ptc_test_raw = [[77], [77], [77], [77], [77], [77], [77],  # single ingredients
#                      [77], [77], [77], [77],  # ingredients combos
#                      [10], [20], [30]]  # coco variants
# coco_ptc_test_raw = torch.tensor(coco_ptc_test_raw)
# coco_ptc_test = (coco_ptc_test_raw - coco_pct_mean)/coco_pct_std


def create_test_vect(test_ing_list, test_cocoa_pct, test_borigin):
    test_ing_onehot = torch.tensor(
        onehotEncode_fortest(test_ing_list, ing_dict), dtype=torch.float)
    test_borigin_onehot = torch.tensor(onehotEncode_fortest(
        test_borigin, borigin_dict), dtype=torch.float)

    test_ing_onehot = (test_ing_onehot-ing_mean)/ing_std
    test_cocoa_pct = torch.tensor(
        [(test_cocoa_pct-coco_pct_mean)/coco_pct_std])
    test_borigin_onehot = (test_borigin_onehot-borigin_mean)/borigin_std

    print("shapes: ", test_ing_onehot.shape,
          test_cocoa_pct.shape, test_borigin_onehot.shape)
    test_vect = torch.cat(
        [test_ing_onehot, test_cocoa_pct, test_borigin_onehot])
    print("shapes: ", test_vect)

    return test_vect


test_vecs = []
for key in ing_dict.keys():
    test_vecs.append(create_test_vect([key], 77, ["Blend"]))

test_vecs.append(create_test_vect(["B", "S"], 77, ["Blend"]))

test_vecs = torch.stack(test_vecs)
# test_vecs = torch.tensor(test_vecs)
y_test = model(test_vecs)
print(y_test)


# ing_test_raw = [[1, 0, 0, 0, 0, 0, 0],  # beans - highly valued ingredient
#                 [0, 1, 0, 0, 0, 0, 0],  # cocoa butter - also important
#                 [0, 0, 1, 0, 0, 0, 0],  # Lecithin
#                 [0, 0, 0, 1, 0, 0, 0],  # sugar - second highest
#                 [0, 0, 0, 0, 1, 0, 0],  # sweetner
#                 [0, 0, 0, 0, 0, 1, 0],  # salt
#                 [0, 0, 0, 0, 0, 0, 1],  # vanilla
#                 [1, 0, 0, 1, 0, 0, 0],  # beans + sugar  - good combo
#                 [1, 0, 0, 1, 1, 0, 0],  # beans + sugar + sweetner - good combo
#                 [1, 1, 0, 1, 0, 0, 0],  # beans + sugar + cocoa butter - good combo
#                 # beans + sugar + cocoa butter + vanilla - lowest combo
#                 [1, 0, 0, 1, 0, 0, 0],
#                 [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 30% coco
#                 [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 60% coco
#                 [1, 0, 0, 1, 0, 0, 0],  # beans + sugar + 90% coco
#                 ]
# ing_test_raw = torch.tensor(ing_test_raw)
# ing_test = (ing_test_raw - ing_mean)/ing_std

# x_test = torch.cat([ing_test, coco_ptc_test], dim=1)
# y_test = model(torch.tensor(x_test, dtype=torch.float))  # y_pred
# print(y_test)
