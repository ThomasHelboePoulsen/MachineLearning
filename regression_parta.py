import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def main():
    file_path = 'Data5_constant_columns_removed.csv'
    df = pd.read_csv(file_path)
    # Target attribute
    target = 'byg026Opførelsesår'
    # Splitting the data into X and y
    X = df.drop(columns=[target])
    y = df[target]
    
    scaler = StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
    y = np.array(y)
    
    lambdas = np.logspace(-5,5,10)
    
    ridgeResults = []
    
    for lamb in lambdas:
        print("Current lambda: " + str(lamb))
        model = Ridge(alpha=lamb, max_iter= 10000)
        mse = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")
        ridgeResults.append(-np.mean(mse))
        print(mse, "\n")
    
    plt.plot(lambdas, ridgeResults)
    plt.xscale('log') #Makes it easier to visualize.
    plt.xlabel('Lambda values')
    plt.ylabel('MSE')
    plt.title('Regularization parameter - Ridge')   
    plt.show()
    
    return ridgeResults

    
    
if __name__ == "__main__":
    main()