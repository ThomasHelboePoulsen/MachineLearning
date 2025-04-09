import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load excel file
file_path = 'Data5_constant_columns_removed.csv'
df = pd.read_csv(file_path)

# Target attribute
target = 'kommunekode_101'

# Splitting the data into X and y
X = df.drop(columns=[target, 'kommunekode_173'])
y = df[target]

#DISABLED: remove high correlation columns, to actually get a challenge?
#corrs = df.corr(numeric_only=True)[target].abs()
#X = df.drop(columns=corrs[corrs > 0.8].index.tolist())
#print(corrs.sort_values(ascending=False))

# Convert to numpy for sklearn compatibility
X = np.array(X)
y = np.array(y)

# Set number of folds
K1 = 10 # Outer CV
K2 = 10 # Inner CV

# k-neighbor values
k_values = [1, 2, 4, 8, 16,32,64,128]

def fit_predict_baseline(X_train,y_train, X_test):
    model = make_pipeline(
                StandardScaler(),
                DummyClassifier(random_state=13)
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def fit_predict_knn(X_train,y_train, X_test, k):
    model = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k)
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def fit_predict_logreg(X_train,y_train, X_test,lamda):
    model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1/lamda, max_iter=10000, random_state=33)
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def classification_error(prediction, test):
    return np.mean(prediction != test)


# Outer loop: to estimate generalisation error
kf_outer = KFold(n_splits=K1, shuffle=True, random_state=24)
outer_knn_test_errors = []
outer_baseline_test_errors = []
outer_logreg_test_errors = []
ls = []
ks = []
logreg_all_errors = []
knn_all_errors = []
baseline_all_errors = []

# Loop over outer folds
for outer_train_idx, outer_test_idx in kf_outer.split(X):
    # Split data into D^par (train) and D^test (test)
    X_par, X_test = X[outer_train_idx], X[outer_test_idx]
    y_par, y_test = y[outer_train_idx], y[outer_test_idx]

    # Initialise list to hold validation errors for each model (hidden unit count)
    avg_val_knn_errors = []
    avg_val_baseline_errors = []
    avg_val_logreg_errors = []


    # Inner loop: to select the best model/hyperparameter
    kf_inner = KFold(n_splits=K2, shuffle=True, random_state=37)

    for inner_train_idx, val_idx in kf_inner.split(X_par):
        # Split D^par into D^train and D^val
        X_train, X_val = X_par[inner_train_idx], X_par[val_idx]
        y_train, y_val = y_par[inner_train_idx], y_par[val_idx]

        #KNN
        inner_knn_val_errors = []
        for h in k_values:

            # Train model M_s (KNN with h hidden units) on D^train
            y_knn = fit_predict_knn(X_train,y_train, X_val,h)
            err = classification_error(y_val, y_knn)
            inner_knn_val_errors.append(err)
        # Compute average validation error across inner folds for model M_s
        avg_val_knn_errors.append(inner_knn_val_errors)


        #LOGREG
        inner_logreg_val_errors = []
        lamdas = [0.1,1,10,100,500,1000,1500,10000]
        for l in lamdas:
            y_logreg = fit_predict_logreg(X_train,y_train, X_val,l)
            err = classification_error(y_val, y_logreg)
            inner_logreg_val_errors.append(err)
        avg_val_logreg_errors.append(inner_logreg_val_errors)

    # Select the best model KNN M*
    avg_val_knn_errors = np.mean(np.array(avg_val_knn_errors), axis=0)
    best_model_index_knn = np.argmin(avg_val_knn_errors)
    best_k = k_values[best_model_index_knn]
    y_best_knn = fit_predict_knn(X_par,y_par, X_test,best_k)
    knn_all_errors.extend((y_best_knn != y_test).astype(int))
    test_mse_knn = classification_error(y_test, y_best_knn)
    outer_knn_test_errors.append(test_mse_knn)
    ks.append(best_k)

    # Select the best model logreg M*
    avg_val_logreg_errors = np.mean(np.array(avg_val_logreg_errors), axis=0)
    best_model_index_logreeg = np.argmin(avg_val_logreg_errors)
    best_l = lamdas[best_model_index_logreeg]
    y_best_logreg = fit_predict_logreg(X_par,y_par, X_test,best_l)
    logreg_all_errors.extend((y_best_logreg != y_test).astype(int))
    test_mse_logreg = classification_error(y_test, y_best_logreg)
    outer_logreg_test_errors.append(test_mse_logreg)
    ls.append(best_l)

    #baseline
    y_baseline = fit_predict_baseline(X_par,y_par, X_test)
    baseline_all_errors.extend((y_baseline != y_test).astype(int))
    outer_baseline_test_errors.append(
        classification_error(
            y_test,
            y_baseline
        )
    )
    print(f"Outer fold: Best k = {best_k}, KNN Test classification error = {test_mse_knn:.2f}")
    print(f"Outer fold: baseline Test classification error = {outer_baseline_test_errors[-1]:.2f}")
    print(f"Outer fold: logreg Test classification error = {test_mse_logreg:.2f}")

# Compute the estimate of the generalisation error, Ê_gen
E_gen = np.mean(outer_knn_test_errors)
print(f"Estimated generalisation error (KNN): {E_gen:.2f}")
print(f"Estimated generalisation error (baseline): {(np.mean(outer_baseline_test_errors)):.2f}")
print(f"Estimated generalisation error (logreg): {(np.mean(outer_logreg_test_errors)):.2f}")


dict_ = {
    "h_i" : ks,
    "E_KNN": outer_knn_test_errors,
    "l_i": ls,
    "E_logreg": outer_logreg_test_errors,
    "E_baseline": outer_baseline_test_errors
}
results = pd.DataFrame(dict_)
results.to_excel("project2Table1Classification.xlsx")

tTestResults = {
    "knn": knn_all_errors,
    "logreg": logreg_all_errors,
    "baseline": baseline_all_errors
}

for key,val in tTestResults.items():
    print(f"size of {key}: {len(val)}")

pd.DataFrame(tTestResults).to_excel("tTestResultsClassification.xlsx")
