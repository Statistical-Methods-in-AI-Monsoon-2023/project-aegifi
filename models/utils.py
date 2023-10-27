# utils file

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split

def load_data():
    X = scipy.sparse.load_npz('vectorised_data/X.npz')
    y = np.load('vectorised_data/y.npy')

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test