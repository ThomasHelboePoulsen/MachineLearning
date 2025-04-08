import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load excel file
file_path = 'Data5_constant_columns_removed.csv'
df = pd.read_csv(file_path)

# Column names
# print(df.columns)

# Target attribute
target = 'byg026Opførelsesår'

# Splitting the data into X and y
X = df.drop(columns=[target])
y = df[target]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data is already one-hot encoded and standardised, right?
# Convert to numpy for sklearn compatibility
X = np.array(X)
y = np.array(y)


# ANN model
from sklearn.neural_network import MLPRegressor

# Set number of folds
K1 = 2 # Outer CV
K2 = 1 # Inner CV

# Hyperparameter values (hidden units)
hidden_units_range = [1, 2, 4, 8, 16,32,64,128]
hidden_units_range = [2**i for i in range(15)]

# Outer loop: to estimate generalisation error
kf_outer = KFold(n_splits=K1, shuffle=True, random_state=24)
outer_test_errors = []

# Loop over outer folds
for outer_train_idx, outer_test_idx in kf_outer.split(X):
    # Split data into D^par (train) and D^test (test)
    X_par, X_test = X[outer_train_idx], X[outer_test_idx]
    y_par, y_test = y[outer_train_idx], y[outer_test_idx]

    # Initialise list to hold validation errors for each model (hidden unit count)
    avg_val_errors = []

    # Loop over each model M_s (each h in hidden_units_range)
    for h in hidden_units_range:
        inner_val_errors = []

        model = make_pipeline(
            StandardScaler(),  # Scaled using only X_train in each fold
            MLPRegressor(
                hidden_layer_sizes=(h,),
                activation='tanh',
                solver='adam',
                max_iter=1000,
                early_stopping=True,
                random_state=49
            )
        )

        model.fit(X_par, y_par)
        # Evaluate on D^val to get E^val_(M_s,j)
        y_test_pred = model.predict(X_test)
        val_mse = mean_squared_error(y_test, y_test_pred)
        print(f"h: {h} - mse: {val_mse}")
        inner_val_errors.append(val_mse)

        # Compute average validation error across inner folds for model M_s
        avg_val_errors.append(np.mean(inner_val_errors))

    # Select the best model M*
    best_model_index = np.argmin(avg_val_errors)
    best_h = hidden_units_range[best_model_index]
    print(f"best h: {best_h} - mse: {avg_val_errors[best_model_index]}")
    continue
# Compute the estimate of the generalisation error, Ê_gen
E_gen = np.mean(outer_test_errors)
print(f"Estimated generalisation error (ANN): {E_gen:.2f}")

