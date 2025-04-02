# =============================================================================
# All purpose testing tool
# =============================================================================
import tkinter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Machine learning specific
from sklearn import model_selection
import importlib_resources
import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr

# Homemade
from regression_on_year import reorder_dataframe, set_0_column, standardize

def main():
    # Load in all data
    file = "Data5_constant_columns_removed.csv"
    Data = set_0_column(pd.read_csv(file), "byg026Opførelsesår")
    Data = standardize(Data)
    attributeNames = list(pd.read_csv(file).columns)
    attributeNames[0], attributeNames[attributeNames.index("byg026Opførelsesår")] = attributeNames[attributeNames.index("byg026Opførelsesår")], attributeNames[0]
    
    kFold(Data[0:200], attributeNames)
    
def kFold(Data,attributeNames):
    # Set up data
    X = Data[:, 1:]
    y = Data[:, 0]
    
    N, M = X.shape
    
    #Cross validation
    K = 5
    CV = model_selection.KFold(n_splits=K, shuffle=True) #Cross validation
    
    # Initialize variables
    Features = np.zeros((M,K))
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_fs = np.empty((K,1))
    Error_test_fs = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    
    k = 0
    
    for train_index, test_index in CV.split(X):
        #Extract training and test set for current cross validation fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        internal_cross_validation = 10
        
        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
        
        # Compute squared error with all features selected (no feature selection)
        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        
        #Compute squared error wtih feature subset selection
        textout = ''
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
        
        Features[selected_features, k] = 1
        if len(selected_features) == 0:
            print("No features were selected")
        else:
            print(X_train[:,selected_features])
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
            Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
            Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
            plt.plot(range(1,len(loss_record)), loss_record[1:])
            plt.xlabel('Iteration')
            plt.ylabel('Squared error (crossvalidation)')    
            
            plt.subplot(1,3,3)
            bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
            plt.clim(-1.5,0)
            plt.xlabel('Iteration')
    
        print('Cross validation fold {0}/{1}'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))

        k+=1
        
    # Display results
    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print("\n")
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

    plt.figure(k)
    plt.subplot(1,3,2)
    bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
    plt.clim(-1.5,0)
    plt.xlabel('Crossvalidation fold')
    plt.ylabel('Attribute')
    

if __name__ == "__main__":
    main()