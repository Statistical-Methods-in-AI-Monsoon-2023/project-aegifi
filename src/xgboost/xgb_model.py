import xgboost as xgb
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, precision_score, recall_score
import numpy as np

import sys
sys.path[0] += '/../utils/'
from utils import load_data, hit_rate

class XGBModel:
    def __init__(self, error_type="constant", load_models=False, word_embeddings='w2v'):
        self.clf = xgb.XGBClassifier(verbosity=2, tree_method="hist", n_jobs=39)
        self.reg = xgb.XGBRegressor(verbosity=2, tree_method="hist", n_jobs=39)
        if load_models:
            self.clf.load_model(f'./src/xgboost/pretrained/xgb_clf_{word_embeddings}.json')
            self.reg.load_model(f'./src/xgboost/pretrained/xgb_reg_{word_embeddings}.json')
        self.train_time = 0
        self.predict_time = 0
        self.error_type = error_type
        self.preds = None
        self.word_embeddings = word_embeddings
    
    def error(self, a, b):
        if self.error_type == "constant":
            return 1
        if self.error_type == "absolute":
            return abs(a-b)
        if self.error_type == "squared":
            return (a-b)**2
    
    def get_thresholds(self, y_prob, y):
        thresholds = []
        for prob, true_labels in zip(y_prob, y):
            errors = np.sum(((prob > prob[:, None]) & (true_labels == 0) | (prob <= prob[:, None]) & (true_labels == 1)) * self.error(prob[:, None],prob), axis=1)
            thresholds.append(prob[np.argmin(errors)])
        return thresholds

    def fit(self, X, y):
        st = time.time()
        print("Fitting classifier...")
        self.clf.fit(X, y)
        y_prob = self.clf.predict_proba(X)
        print("Getting thresholds...")
        thresholds = self.get_thresholds(y_prob, y)
        print("Fitting regressor...")
        self.reg.fit(y_prob, thresholds)
        self.train_time = time.time() - st
        print("Done fitting model")
        print(f"Train time: {self.train_time}")
    
    def predict(self, X):
        st = time.time()
        print("Predicting...")
        y_prob = self.clf.predict_proba(X)
        y_thresh = self.reg.predict(y_prob)
        y_pred = (y_prob >= y_thresh[:, None]).astype(int)
        self.predict_time = time.time() - st
        self.preds = y_pred
        print(f"Predict time: {self.predict_time}")
        
        return y_pred
    
    def predict_proba(self, X):
        print("Predicting probabilities...")
        y_prob = self.clf.predict_proba(X)
        return y_prob
    
    def write_metrics(self, y_test):
        file_name = f'xgb_{self.word_embeddings}_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

        file_path = f'./src/xgboost/metrics/{file_name}'

        with open(file_path, 'w') as f:
            # f.write(f'Error type: {self.error_type}\n')
            # f.write(f'Word embeddings: {self.word_embeddings}\n')
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(y_test, self.preds, average="samples")}\n')
            f.write(f'Hit Rate: {hit_rate(y_test, self.preds)}\n')
            f.write(f'F1 Score: {f1_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Precision Score: {precision_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Recall Score: {recall_score(y_test, self.preds, average="samples", zero_division=True)}\n')

    def save_model(self):
        self.clf.save_model(f'./src/xgboost/pretrained/xgb_clf_{self.word_embeddings}.json')
        self.reg.save_model(f'./src/xgboost/pretrained/xgb_reg_{self.word_embeddings}.json')

class XGBRunner:
    def __init__(self, load_models=False, word_embeddings='w2v'):
        self.load_models = load_models
        self.word_embeddings = word_embeddings
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.init_model()
    
    def load_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(word_embeddings=self.word_embeddings)
    
    def init_model(self):
        self.model = XGBModel(load_models=self.load_models, word_embeddings=self.word_embeddings)

    def run_training(self):
        self.load_data()
        
        self.model.fit(self.X_train, self.y_train)
        self.model.save_model()
    
    def run_inference(self, save_preds=False):
        self.load_data()
        
        preds = self.model.predict(self.X_test)
        
        if save_preds:
            np.save(f'EDA/preds/xgb_{self.word_embeddings}.npy', preds)
            np.save(f'EDA/preds/y_test_xgb_{self.word_embeddings}.npy', self.y_test)
        
        self.model.write_metrics(self.y_test)
