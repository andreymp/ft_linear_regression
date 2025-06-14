import sys
import numpy as np

from common import read_dataset, estimate_price, read_thetas

def r2_score(y, y_pred, mean):
    ss_total = np.sum((y - mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        if not filename.endswith('.csv'):
            print("Wrong file format")
            sys.exit()
        dataset = read_dataset(filename)
        if dataset is None:
            sys.exit()
        if len(dataset) == 0:
            print("Dataset is empty")
            sys.exit()
        coeff = read_thetas()
        mean = coeff["mean"]
        X = np.array(dataset["km"])
        X_std = (X - mean) / coeff["std"]

        y = np.array(dataset["price"])
        y_pred = estimate_price(X_std, coeff["thetas"])

        print(f"Precision [0 -> 1]: {r2_score(y,  y_pred, mean)}")
    else:
        print("Wrong number of arguments")