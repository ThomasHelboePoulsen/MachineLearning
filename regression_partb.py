import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_predict_ann(X_train,y_train, X_test,h):
    model = make_pipeline(
                StandardScaler(),  # Scaled using only X_train in each fold
                MLPRegressor(
                    hidden_layer_sizes=(h,),
                    #activation='identity',
                    solver='adam',
                    max_iter=10000,
                    early_stopping=True,
                    random_state=49
                )
            )
    model.fit(X_train, y_train)
    # Evaluate on D^val to get E^val_(M_s,j)
    return model.predict(X_test)

# Load excel file
file_path = 'Data5_constant_columns_removed.csv'
df = pd.read_csv(file_path)
#df = df.head(50)

# Column names
# print(df.columns)

# Target attribute
target = 'byg026Opførelsesår'

# Compute the mean of construction year (Baseline)
baseline = df[target].mean()
print(f'Baseline: {baseline:.2f}')

# Splitting the data into X and y
X = df.drop(columns=[target])
y = df[target]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data is already one-hot encoded and standardised, right?
# Convert to numpy for sklearn compatibility
X = np.array(X)
y = np.array(y)

# Outer CV loop
K1 = 10
kf_outer = KFold(n_splits=K1, shuffle=True, random_state=8)

test_errors = []

for train_idx, test_idx in kf_outer.split(X):
    # Outer split D_par (training), D_test (test)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train baseline model on D_par (training set)
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)

    # Evaluate on D_test
    y_pred = baseline_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    test_errors.append(mse)

# Final generalization error
E_gen = np.mean(test_errors)
print(f'Estimated generalisation error (baseline): {E_gen:.2f}')


# ANN model
from sklearn.neural_network import MLPRegressor

# Set number of folds
K1 = 2 # Outer CV
K2 = 2 # Inner CV

# Hyperparameter values (hidden units)
hidden_units_range = [1, 2, 4, 8, 16]

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

    # Inner loop: to select the best model/hyperparameter
    kf_inner = KFold(n_splits=K2, shuffle=True, random_state=37)

    # Loop over each model M_s (each h in hidden_units_range)
    for h in hidden_units_range:
        inner_ann_val_errors = []

        # Loop over inner folds
        for inner_train_idx, val_idx in kf_inner.split(X_par):
            # Split D^par into D^train and D^val
            X_train, X_val = X_par[inner_train_idx], X_par[val_idx]
            y_train, y_val = y_par[inner_train_idx], y_par[val_idx]

            # Train model M_s (ANN with h hidden units) on D^train
            y_ann = fit_predict_ann(X_train,y_train, X_val,h)
            inner_ann_val_errors.append(mean_squared_error(y_val, y_ann))

        # Compute average validation error across inner folds for model M_s
        avg_val_errors.append(np.mean(inner_ann_val_errors))

    # Select the best model M*
    best_model_index = np.argmin(avg_val_errors)
    best_h = hidden_units_range[best_model_index]
    y_best_ann = fit_predict_ann(X_par,y_par, X_test,best_h)
    test_mse = mean_squared_error(y_test, y_best_ann)
    outer_test_errors.append(test_mse)

    print(f"Outer fold: Best h = {best_h}, Test MSE = {test_mse:.2f}")

# Compute the estimate of the generalisation error, Ê_gen
E_gen = np.mean(outer_test_errors)
print(f"Estimated generalisation error (ANN): {E_gen:.2f}")

