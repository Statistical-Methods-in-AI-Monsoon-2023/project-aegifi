from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import numpy as np
from time import time
from joblib import dump, load

import sys
sys.path[0] += '/../utils/'
from utils import load_data, hit_rate

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    model = OneVsRestClassifier(ComplementNB())
    train_time = time()
    model.fit(X_train, y_train)
    train_time = time() - train_time

    # predict on test data
    predict_time = time()
    y_pred = model.predict(X_test)
    predict_time = time() - predict_time

    for i in range(10):
        print(y_pred[i], y_test[i])

    # save metrics to file
    file_name = f'multinomial_nb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
    file_path = f'./metrics/{file_name}'

    with open(file_path, 'w') as f:
        f.write(f'Train time: {train_time}\n')
        f.write(f'Predict time: {predict_time}\n')
        f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
        f.write(f'Hamming Score: {1 - hamming_loss(y_test, y_pred)}\n')
        f.write(f'Jaccard Score: {jaccard_score(y_test, y_pred, average="micro")}\n')
        f.write(f'Hit Rate: {hit_rate(y_test, y_pred)}\n')
        f.write('Classification Report:\n')
        f.write(f'{classification_report(y_test, y_pred, zero_division=True)}\n')

class MultinomailNBRunner:
    def __init__(self, load_models=False):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        if load_models:
            self.load_model()
        else:
            self.model = OneVsRestClassifier(MultinomialNB())
        self.train_time = 0
        self.predict_time = 0
        self.preds = None
        
    def load_data(self):
        print("loading data...")
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()
    
    def save_model(self):
        # save using joblib
        dump(self.model, f'./src/naive_bayes/pretrained/multinomial_nb.joblib')
    
    def load_model(self):
        # load using joblib
        self.model = load(f'./src/naive_bayes/pretrained/multinomial_nb.joblib')
    
    def run_training(self):
        self.load_data()
        print("Training...")
        self.train_time = time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time() - self.train_time
        print("Trained")
        self.save_model()
        print(f"Train time: {self.train_time}")


    
    def write_metrics(self):
        file_name = f'binary_nb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
        file_path = f'./src/naive_bayes/metrics/{file_name}'
        with open(file_path, 'w') as f:
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(self.y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(self.y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(self.y_test, self.preds, average="micro")}\n')
            f.write(f'Hit Rate: {hit_rate(self.y_test, self.preds)}\n')
            f.write('Classification Report:\n')
            f.write(f'{classification_report(self.y_test, self.preds, zero_division=True)}\n')
    
    def run_inference(self):
        self.load_data()
        self.load_model()
        print("Predicting...")
        self.predict_time = time()
        self.preds = self.model.predict(self.X_test)
        self.predict_time = time() - self.predict_time
        self.write_metrics()
        print("Predicted")
