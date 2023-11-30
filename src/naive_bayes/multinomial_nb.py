from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, precision_score, recall_score
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import numpy as np
from time import time
from joblib import dump, load

import sys
sys.path[0] += '/../../utils/'
from utils import load_data, hit_rate

class MultinomialNBRunner:
    def __init__(self, load_models=False, word_embeddings='w2v'):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.word_embeddings = word_embeddings
        if load_models:
            self.load_model()
        else:
            self.model = OneVsRestClassifier(MultinomialNB(alpha=0.5))
        self.train_time = 0
        self.predict_time = 0
        self.preds = None
        
    def load_data(self):
        print("loading data...")
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(word_embeddings=self.word_embeddings)
    
    def save_model(self):
        # save using joblib
        dump(self.model, f'./src/naive_bayes/pretrained/multinomial_nb_{self.word_embeddings}.joblib')
    
    def load_model(self):
        # load using joblib
        self.model = load(f'./src/naive_bayes/pretrained/multinomial_nb_{self.word_embeddings}.joblib')
    
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
        file_name = f'multi_nb_{self.word_embeddings}_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
        file_path = f'./src/naive_bayes/metrics/{file_name}'
        with open(file_path, 'w') as f:
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(self.y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(self.y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(self.y_test, self.preds, average="samples")}\n')
            f.write(f'Hit Rate: {hit_rate(self.y_test, self.preds)}\n')
            f.write(f'F1 Score: {f1_score(self.y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Precision Score: {precision_score(self.y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Recall Score: {recall_score(self.y_test, self.preds, average="samples", zero_division=True)}\n')
    
    def run_inference(self, save_preds=False):
        self.load_data()
        self.load_model()
        print("Predicting...")
        self.predict_time = time()
        self.preds = self.model.predict(self.X_test)
        self.predict_time = time() - self.predict_time
        
        if save_preds:
            np.save(f'EDA/preds/multi_nb_{self.word_embeddings}.npy', self.preds)
            np.save(f'EDA/preds/y_test_multi_nb_{self.word_embeddings}.npy', self.y_test)
        
        self.write_metrics()
        print("Predicted")
