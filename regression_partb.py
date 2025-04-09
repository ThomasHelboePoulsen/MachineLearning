import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_predict_ann(X_train,y_train, X_test,h):
    model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(h,),
                    learning_rate_init=0.01,
                    solver='adam',
                    max_iter=10000,
                    early_stopping=True,
                    random_state=49
                )
            )
    model.fit(X_train, y_train)
    # Evaluate on D^val to get E^val_(M_s,j)
    return model.predict(X_test)

def fit_predict_baseline(X_train,y_train, X_test):
    model = make_pipeline(
                StandardScaler(),
                DummyRegressor(strategy='mean')
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def fit_predict_linreg(X_train,y_train, X_test,lamda):
    model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=lamda, max_iter=10000)
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Load excel file
file_path = 'Data5_constant_columns_removed.csv'
df = pd.read_csv(file_path)



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
# ANN model
from sklearn.neural_network import MLPRegressor

# Set number of folds
K1 = 10 # Outer CV
K2 = 10 # Inner CV

# Hyperparameter values (hidden units)
hidden_units_range = [1, 2, 4, 8, 16,32,64,128]

# Outer loop: to estimate generalisation error
kf_outer = KFold(n_splits=K1, shuffle=True, random_state=24)
outer_ann_test_errors = []
outer_baseline_test_errors = []
outer_linreg_test_errors = []
ls = []
hs = []
linreg_all_errors = []
ann_all_errors = []
baseline_all_errors = []

# Loop over outer folds
for outer_train_idx, outer_test_idx in kf_outer.split(X):
    # Split data into D^par (train) and D^test (test)
    X_par, X_test = X[outer_train_idx], X[outer_test_idx]
    y_par, y_test = y[outer_train_idx], y[outer_test_idx]

    # Initialise list to hold validation errors for each model (hidden unit count)
    avg_val_ann_errors = []
    avg_val_baseline_errors = []
    avg_val_linreg_errors = []


    # Inner loop: to select the best model/hyperparameter
    kf_inner = KFold(n_splits=K2, shuffle=True, random_state=37)

    for inner_train_idx, val_idx in kf_inner.split(X_par):
        # Split D^par into D^train and D^val
        X_train, X_val = X_par[inner_train_idx], X_par[val_idx]
        y_train, y_val = y_par[inner_train_idx], y_par[val_idx]

        #ANN
        inner_ann_val_errors = []
        for h in hidden_units_range:
            # Train model M_s (ANN with h hidden units) on D^train
            y_ann = fit_predict_ann(X_train,y_train, X_val,h)
            inner_ann_val_errors.append(mean_squared_error(y_val, y_ann))
        # Compute average validation error across inner folds for model M_s
        avg_val_ann_errors.append(inner_ann_val_errors)


        #LINREG
        inner_linreg_val_errors = []
        lamdas = [100,500,1000,1500]
        for l in lamdas:
            y_linreg = fit_predict_linreg(X_train,y_train, X_val,l)
            inner_linreg_val_errors.append(mean_squared_error(y_val, y_ann))
        avg_val_linreg_errors.append(inner_linreg_val_errors)


    # Select the best model ANN M*
    avg_val_ann_errors = np.mean(np.array(avg_val_ann_errors), axis=0)
    best_model_index_ann = np.argmin(avg_val_ann_errors)
    best_h = hidden_units_range[best_model_index_ann]
    y_best_ann = fit_predict_ann(X_par,y_par, X_test,best_h)
    ann_all_errors.extend((y_best_ann-y_test)**2)
    test_mse_ann = mean_squared_error(y_test, y_best_ann)
    outer_ann_test_errors.append(test_mse_ann)
    hs.append(best_h)

    # Select the best model linreg M*
    avg_val_linreg_errors = np.mean(np.array(avg_val_linreg_errors), axis=0)
    best_model_index_linreeg = np.argmin(avg_val_linreg_errors)
    best_l = lamdas[best_model_index_linreeg]
    y_best_linreg = fit_predict_linreg(X_par,y_par, X_test,best_l)
    linreg_all_errors.extend((y_best_linreg-y_test)**2)
    test_mse_linreg = mean_squared_error(y_test, y_best_linreg)
    outer_linreg_test_errors.append(test_mse_linreg)
    ls.append(l)
    #baseline

    y_baseline = fit_predict_baseline(X_par,y_par, X_test)
    baseline_all_errors.extend((y_baseline-y_test)**2)
    outer_baseline_test_errors.append(
        mean_squared_error(
            y_test,
            y_baseline
        )
    )
    print(f"Outer fold: Best h = {best_h}, ANN Test MSE = {test_mse_ann:.2f}")
    print(f"Outer fold: baseline Test MSE = {outer_baseline_test_errors[-1]:.2f}")
    print(f"Outer fold: linreg Test MSE = {test_mse_linreg:.2f}")

# Compute the estimate of the generalisation error, Ê_gen
E_gen = np.mean(outer_ann_test_errors)
print(f"Estimated generalisation error (ANN): {E_gen:.2f}")
print(f"Estimated generalisation error (baseline): {(np.mean(outer_baseline_test_errors)):.2f}")
print(f"Estimated generalisation error (linreg): {(np.mean(outer_linreg_test_errors)):.2f}")


dict_ = {
    "h_i" : hs,
    "E_ANN": outer_ann_test_errors,
    "l_i": ls,
    "E_linreg": outer_linreg_test_errors,
    "E_baseline": outer_baseline_test_errors
}
results = pd.DataFrame(dict_)
results.to_excel("project2Table1.xlsx")

tTestResults = {
    "ann": ann_all_errors,
    "linreg": linreg_all_errors,
    "baseline": baseline_all_errors
}

for key,val in tTestResults.items():
    print(f"size of {key}: {len(val)}")

pd.DataFrame(tTestResults).to_excel("tTestResults.xlsx")
