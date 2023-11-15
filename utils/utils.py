# utils file

import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split

def load_data(w2v=False):
    X = scipy.sparse.load_npz('vectorised_data/X.npz')
    y = np.load('vectorised_data/y.npy')
    

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if w2v:
        X_train = np.load('embeddings/train_embed.npy', allow_pickle=True)
        X_test = np.load('embeddings/test_embed.npy', allow_pickle=True)
    
    print('Loaded data')
    
    
    # if w2v:
    #     X_train = np.load('embeddings/train_embed.npy', allow_pickle=True)
    #     X_test = np.load('embeddings/test_embed.npy', allow_pickle=True)
    #     y = np.load('embeddings/y.npy')
    #     y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    # else:
    #     X = scipy.sparse.load_npz('vectorised_data/X.npz')
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def hit_rate(predicted, true):
    # Use element-wise logical AND to check if at least one class is predicted correctly
    correct_predictions = np.any(np.logical_and(predicted, true), axis=1)
    
    # Calculate the hit rate as the percentage of correct samples
    hit_rate = (np.sum(correct_predictions) / len(correct_predictions))
    
    return hit_rate