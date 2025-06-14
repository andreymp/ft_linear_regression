import numpy as np
import sys

from common import estimate_price, read_thetas

if __name__ == "__main__":
    coeff = read_thetas()

    if coeff["thetas"] is []:
        sys.exit()
    while True:
        mileage = input("Enter mileage: ")
        if (mileage == "q"):
           break
        print(
            "Mileage must be an integer" if not mileage.isdigit() else 
            f"Estimate price: {estimate_price((float(mileage) - coeff["mean"]) / coeff["std"], coeff["thetas"])}"
        )