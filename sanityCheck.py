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
                    hidden_layer_sizes=(1000,1000,1000,1000),
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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data is already one-hot encoded and standardised, right?
# Convert to numpy for sklearn compatibility
X = np.array(X)
y = np.array(y)

kf_outer = KFold(n_splits=2, shuffle=True, random_state=24)

# Loop over outer folds
for outer_train_idx, outer_test_idx in kf_outer.split(X):
    X_par, X_test = X[outer_train_idx], X[outer_test_idx]
    y_par, y_test = y[outer_train_idx], y[outer_test_idx]
    y_pred = fit_predict_ann(X_par,y_par,X_test,0)
    print(mean_squared_error(y,y_pred))

