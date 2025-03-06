from regression_on_year import reorder_dataframe, set_0_column

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


# Open correct file and set 0. column to desired testing value
file = "Data5_constant_columns_removed.csv"

Data = set_0_column(pd.read_csv(file), "byg026Opførelsesår")

# Number of principal components
Principal_Components = [5,10,15,20,25]

#Load in training data
X = Data[:, 1:]
y = Data[:, 0]

n = [x for x in range(int(min(y)),int(max(y))+1 )]
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

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)

V = Vh.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# Project data onto principal component space
Z = Y @ V

plt.plot()
plt.plot(rho[1:], "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained value")

# Plot PCA of the data
f = plt.figure()
plt.title("pixel vectors of years of construction projected on PCs")
# for c in n:
#     class_mask = y == c
#     plt.plot(Z[class_mask, 0], Z[class_mask, 1], "o", color="red")
# Plot the PCA components, color points based on the year of construction
norm_y = (y - np.min(y)) / (np.max(y) - np.min(y))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=norm_y, cmap='coolwarm', edgecolors='k')

# Add a colorbar to show the mapping of colors to years
cbar = plt.colorbar(scatter)
cbar.set_label('Year of Construction', rotation=270, labelpad=15)
plt.xlabel("PC1")
plt.ylabel("PC2")

# =============================================================================
# Variance graph explained - show vectors as heat map
# Show atleast PC1 and PC2, more if anything interesting is found
# Show variance as a result of components.
# Color code according to year / areacode.
# =============================================================================
