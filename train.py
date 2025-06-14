import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

from common import estimate_price, read_dataset

def write_tethas(tethas, mean, std):
    with open("thetas.pkl", 'wb') as file:
       pickle.dump({
           "thetas": tethas,
           "mean": mean,
           "std": std
       }, file)

def train(dataset,  learning_rate = 0.01, iter_nbr = 1000):
    mileage = np.array(dataset['km'])
    mean = np.mean(mileage)
    std = np.std(mileage)
    mileage_std = (mileage - mean) / std

    price = np.array(dataset['price'])
    thetas = np.zeros(2)
    m = len(dataset)

    plt.scatter(mileage, price, color='blue', label='Data points')
    line, = plt.plot([], [], 'r-', linewidth=2, label='Current fit')
    plt.title('Linear Regression Training')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()

    plt.xlim(min(mileage) - 10000, max(mileage) + 10000)
    plt.ylim(min(price) - 1000, max(price) + 1000)

    plt.ion()
    plt.show()

    for idx in range(iter_nbr):
        predictions = estimate_price(mileage_std, thetas)
        errors = predictions - price
        thetas[0] -= learning_rate * np.sum(errors) / m
        thetas[1] -= learning_rate * np.sum(errors * mileage_std) / m

        if not idx % 10:
            X = np.array([min(mileage), max(mileage)])
            X_std = (X - mean) / std
            y = estimate_price(X_std, thetas)
            line.set_data(X, y)
            plt.draw()
            plt.pause(0.03)
    
    plt.ioff()
    plt.show()

    return thetas, mean, std

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
        tethas, mean, std = train(dataset)
        write_tethas(tethas if not tethas is None else [0, 0], mean, std)
    else:
        print("Wrong number of arguments")