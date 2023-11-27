import numpy as np


def onehotEncode(input_col):
    inset = set()
    for instr in input_col:
        inlist = instr.split(",")
        for item in inlist:
            inset.add(item.strip())
    # print("inset: ", inset)

    indict = {}
    for idx, item in enumerate(sorted(inset)):
        # print(idx, ing)
        indict[item] = idx
    # print("indict: ", indict)  # ingredient to number mapping

    num_rows = len(input_col)
    num_category = len(inset)
    in_raw_data = np.zeros((num_rows, num_category), dtype=float)

    for row_num, instr in enumerate(input_col):
        inlist = instr.split(",")
        for item in inlist:
            idx = indict[item.strip()]
            in_raw_data[row_num][idx] += 1
    # print(in_raw_data[53])

    return in_raw_data, num_category, indict


def onehotEncode_fortest(categories, indict):
    print("indict: ", indict)
    test_vect = np.zeros(len(indict))

    for catetory in categories:
        idx = indict[catetory]
        test_vect[idx] = 1

    return test_vect
