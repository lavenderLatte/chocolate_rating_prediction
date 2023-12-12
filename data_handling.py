import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from text_to_onehotvect import onehotEncode, onehotEncode_fortest


def get_splited_dataset(data):
    raw_data = pd.read_csv(data)
    removed_null = raw_data.dropna()
    dropped_id = removed_null.drop("id", axis=1)
    cleaned_data = dropped_id
    # print(raw_data.info())
    # print(cleaned_data.head().T)
    # print(cleaned_data.info())
    # print(cleaned_data.describe().T)
    # print(f"cocoa_percent : {cleaned_data['cocoa_percent'].value_counts()}")
    # cleaned_data.hist(layout=(5, 4), figsize=(20, 15), bins=20)
    # plt.show()

    num_row = cleaned_data.shape[0]
    num_row_train = int(num_row * 0.7)
    num_row_dev = int(num_row * 0.2)
    num_row_test = num_row - num_row_train - num_row_dev

    shuffled_data = cleaned_data.sample(frac=1)
    train = shuffled_data[: num_row_train]
    dev = shuffled_data[num_row_train: num_row_train+num_row_dev]
    test = shuffled_data[num_row_train+num_row_dev:]

    # print(
    #     f"num_row: {num_row}, num_row_train: {num_row_train}, num_row_dev: {num_row_dev}, num_row_test: {num_row_test}")
    # print(
    #     f"len shuffled_data: {len(shuffled_data)},  train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    # print(train['rating'].value_counts())
    # print(dev['rating'].value_counts())
    # print(test['rating'].value_counts())

    return train, dev, test


# def get_input_output_from_df(dataframe, mean_std_coco=None, mean_std_ing=None, mean_std_borigin=None):
#     coco_pct_raw = torch.tensor(
#         dataframe['cocoa_percent'].values, dtype=torch.float)
#     # for training df, we need to calcuate mean and std
#     # but for dev and test, we reuse mean and std from training set
#     if mean_std_coco == None:
#         mean_std_coco = (torch.mean(coco_pct_raw), torch.std(coco_pct_raw))

#     coco_pct_norm = (coco_pct_raw - mean_std_coco[0]) / \
#         mean_std_coco[1]  # mean normalization
#     coco_pct = coco_pct_norm.reshape(-1, 1)  # input

#     ingredients = dataframe['ingredients']
#     ing_raw_data, num_ing, ing_dict = onehotEncode(ingredients)
#     ing_raw = torch.tensor(ing_raw_data, dtype=torch.float)
#     ing_mean = torch.mean(ing_raw)
#     ing_std = torch.std(ing_raw)
#     ing = (ing_raw - ing_mean)/ing_std


# def main():
#     get_splited_dataset("./chocolate_bars.csv")


# if __name__ == "__main__":
#     main()
