# utils file

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
import cupy as cp
import cupyx.scipy.sparse as csp

def load_data():
    X = scipy.sparse.load_npz('vectorised_data/X.npz')
    y = np.load('vectorised_data/y.npy')
    

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(X.shape, y.shape)
    #X_train = X[:200000]
    #X_test = X[200000:]
    #y_train = y[:200000]
    #y_test = y[200000:]
    print('Loaded data')

    return X_train, X_test, y_train, y_test

def hit_rate(predicted, true):
    # Use element-wise logical AND to check if at least one class is predicted correctly
    correct_predictions = np.any(np.logical_and(predicted, true), axis=1)
    
    # Calculate the hit rate as the percentage of correct samples
    hit_rate = (np.sum(correct_predictions) / len(correct_predictions))
    
    return hit_rate