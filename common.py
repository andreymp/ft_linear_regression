import pandas as pd
import pickle

def estimate_price(mileage, thetas):
    return thetas[0] + mileage * thetas[1]

def read_dataset(dataset_file):
    try:
        return pd.read_csv(dataset_file)
    except IOError:
        print("Error: could not read thetas' file")
        return None

def read_thetas(filename="thetas.pkl"):
    try:
        with open(filename, "rb") as file:
           return pickle.load(file)
    except FileNotFoundError:
        print("Error: could not read thetas' file")
        return {}