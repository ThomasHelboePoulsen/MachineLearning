#Regularization
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from generalTools import set_0_column, standardize

def main():
    file = "Data5_constant_columns_removed.csv"
    Data = set_0_column(pd.read_csv(file), "byg026Opførelsesår")
    Data = standardize(Data)
    attributeNames = list(pd.read_csv(file).columns)
    attributeNames[0], attributeNames[attributeNames.index("byg026Opførelsesår")] = attributeNames[attributeNames.index("byg026Opførelsesår")], attributeNames[0]
    
    X = Data[:, 1:]
    y = Data[:, 0]
    
    lambdas = [0.001,0.01, 0.1, 1, 2.5, 5, 7.5, 10, 25, 50,75,100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
    results = []
    
    for lamb in lambdas:
        print("Current lambda: " + str(lamb))
        model = Ridge(alpha=lamb)
        mse = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")
        results.append(-np.mean(mse))
        print(mse)
        print()
    
    plt.plot(lambdas, results)
    plt.xscale('log') #Makes it easier to visualize.
    plt.xlabel('Lambda values')
    plt.ylabel('MSE')
    plt.title('Regularization parameter - Ridge')   
    plt.show()

if __name__ == "__main__":
    main()