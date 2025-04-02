from regression_on_year import reorder_dataframe, set_0_column

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def main():
    # Open correct file and set 0. column to desired testing value
    file = "Data5_constant_columns_removed.csv"
    
    Data = set_0_column(pd.read_csv(file), "byg026Opførelsesår")
    
    # Number of principal components
    Principal_Components = [5,10,15,20,25]
    
    #Load in training data
    X = Data[:, 1:]
    y = Data[:, 0]
    
    n = [x for x in range(int(min(y)),int(max(y))+1)]
    N,M = X.shape
    C = len(n)
    
    classValues = n
    classNames =  [str(num) for num in n]
    classDict = dict(zip(classNames, classValues))
    
    class_mask = np.zeros(N).astype(bool)
    
    for v in n:
        cmsk = y == v
        
        class_mask = class_mask | cmsk
    
    X = X[class_mask, :]
    y = y[class_mask]
    
    N = X.shape[0]
    
    # Data centering (subtracting mean and dividing by std)
    Y = (X - np.ones((N,1)) * X.mean(0)) / X.std(0)
    print("Standard Deviation of Y: " + str(Y.std(0)))
    
    # PCA by computing SVD of Y
    U, S, Vh = svd(Y, full_matrices=False)
    
    V = Vh.T
    
    #Plot vectors
    plt.plot(V[:, 0], label="PC1")
    plt.plot(V[:, 1], label="PC2")
    plt.title("PC1 and PC2 in terms of attributes")
    plt.xlabel("Attribute")
    plt.ylabel("Principal direction")
    plt.legend()
    plt.show()
    
    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()
    
    # Project data onto principal component space
    Z = Y @ V
    
    #Finding max loadings for PC1 and PC2
    V_Loadings = [max(V[:, 0]), max(V[:, 1])]
    
    # plt.plot()
    # plt.plot(rho, "o-")
    # plt.title("Variance explained by principal components")
    # plt.xlabel("Principal component")
    # plt.ylabel("Variance explained value")
    
    cumm_rho = rho.copy()
    for x in range(1,len(rho)):
        cumm_rho[x] += cumm_rho[x-1]
        
    plt.plot()
    plt.plot(cumm_rho, "o-")
    plt.title("Cumulative variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance explained value")
    
    # Plot PCA of the data
    f = plt.figure()
    plt.title("pixel vectors of years of construction projected on PCs")
    
    # Plot the PCA components, color points based on the year of construction
    # Use np.clip to clip at 1850, since most values are from then, so heatmap is skewed towards a minority when 1650 - 2025 is all counted.
    y_clipped = np.clip(y, 1850, 2025)  # Clamp values below 1850 to 1850

    # Normalize the clipped years from 1850 to 2025
    norm_y = (y_clipped - 1850) / (2025 - 1850)
    #Use heatmap coolwarm to distinguish between older and newer buildings
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=norm_y, cmap='coolwarm', edgecolors='k')
    
    # Add a colorbar to show the mapping of colors to years
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year of Construction', rotation=270, labelpad=15)
    
    #set correct ticks
    
    cbar.set_ticks(np.linspace(0,1,6))
    cbar.set_ticklabels(["<1850","1885","1920","1955","1990","2025"])
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # =============================================================================
    # Variance graph explained - show vectors as heat map
    # Show atleast PC1 and PC2, more if anything interesting is found
    # Show variance as a result of components.
    # Color code according to year / areacode.
    # =============================================================================

if __name__ == "__main__":
    main()