import pandas
import numpy as np

# Reorders dataframe and turns it into a numpy array
def reorder_dataframe(DF, index1, index2):
    nums = [x for x in range(len(DF.columns))]
    nums[index1] = index2
    nums[index2] = index1
    
    return DF.iloc[:, nums].to_numpy()

# Sets specific column to 0. column. Also changes to numpy array.
def set_0_column(DF, Column):
    index1 = list(DF.columns).index(Column)
    
    return reorder_dataframe(DF, index1, 0)

def standardize(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)