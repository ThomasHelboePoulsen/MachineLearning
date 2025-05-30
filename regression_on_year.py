# =============================================================================
# pca analysis - attempt to have it guess the year of creation based on area, commune, etc.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor



def main():
    # Load data in numpy format, easier for doing the analysis
    Columns = list(pd.read_csv("Data4_final.csv").columns)
    Data = pd.read_csv("Data4_final.csv")
    
    #Choose what column it should try to guess - Year of construction chosen
    Column = "byg026Opførelsesår"
    #Reorders so Byg026 is first value
    Data = reorder_dataframe(Data, Columns.index(Column),0)
    
    # List of all years of construction
    n = list(set(Data[:, 0]))
    
    
    #Split into Train and Test
    Split = int(0.8 * len(Data))
    Train_Data = Data[0:Split]
    Test_Data = Data[Split + 1:len(Data)]
    
    #Set X and y
    X_Train = Train_Data[:, 1:]
    y_Train = Train_Data[:, 0]
    
    X_Test = Test_Data[:, 1:]
    y_Test = Test_Data[:, 0]
    
    #Standardize data
    X_Train_Scaled = StandardScaler().fit_transform(X_Train)
    X_Test_Scaled = StandardScaler().fit_transform(X_Test)
    
    #PCA
    pca = PCA(10)
    X_Train_PCA = pca.fit_transform(X_Train_Scaled)
    X_Test_PCA = pca.transform(X_Test_Scaled)
    
    #Regression - Train & Test
    #model = Lasso(alpha=0.2)
    model = BayesianRidge()
    model.fit(X_Train_PCA, y_Train)
    
    prediction = model.predict(X_Test_PCA)
    
    #Evaluate Errors
    MeanSquaredError = mean_squared_error(y_Test,prediction)
    r2 = r2_score(y_Test, prediction)
    
    print("Mean Squared Error: " + str(MeanSquaredError))
    print("R2:                 " + str(r2))
    
    return X_Train_PCA

if __name__ == "__main__":
    X_Train_PCA = main()

# =============================================================================
# make boxplot of each attribute
# =============================================================================
