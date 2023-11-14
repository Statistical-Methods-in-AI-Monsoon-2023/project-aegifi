import xgboost as xgb
from utils import load_data, hit_rate
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, classification_report
import numpy as np

class XGBModel:
    def __init__(self, error_type="constant", load_models=False, word_embeddings='w2v'):
        self.clf = xgb.XGBClassifier(verbosity=2, tree_method="hist", n_jobs=39)
        self.reg = xgb.XGBRegressor(verbosity=2, tree_method="hist", n_jobs=39)
        if load_models:
            self.clf.load_model('pretrained/xgb_clf_w2v.json')
            self.reg.load_model('pretrained/xgb_reg_w2v.json')
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
        self.clf.fit(X, y)
        y_prob = self.clf.predict_proba(X)
        thresholds = self.get_thresholds(y_prob, y)
        self.reg.fit(y_prob, thresholds)
        self.train_time = time.time() - st
    
    def predict(self, X):
        st = time.time()
        y_prob = self.clf.predict_proba(X)
        y_thresh = self.reg.predict(y_prob)
        y_pred = (y_prob >= y_thresh[:, None]).astype(int)
        self.predict_time = time.time() - st
        self.preds = y_pred
        return y_pred
    
    def write_metrics(self, y_test):
        file_name = f'xgb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

        file_path = f'metrics/{file_name}'

        with open(file_path, 'w') as f:
            f.write(f'Error type: {self.error_type}\n')
            f.write(f'Word embeddings: {self.word_embeddings}\n')
            f.write(f'Train time: {self.train_time}\n')
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(y_test, self.preds, average="micro")}\n')
            f.write(f'Hit Rate: {hit_rate(y_test, self.preds)}\n')
            f.write('Classification Report:\n')
            f.write(f'{classification_report(y_test, self.preds, zero_division=True)}\n')

    def save_model(self):
        self.clf.save_model(f'pretrained/xgb_clf_{self.word_embeddings}.json')
        self.reg.save_model(f'pretrained/xgb_reg_{self.word_embeddings}.json')

class XGBRunner:
    def __init__(self, load_models=False, word_embeddings='w2v'):
        self.load_models = load_models
        self.word_embeddings = word_embeddings
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        if self.word_embeddings == 'w2v':
            self.X_train, self.X_test, self.y_train, self.y_test = load_data(w2v=True)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = load_data()
    
    def init_model(self):
        self.model = XGBModel(load_models=self.load_models, word_embeddings=self.word_embeddings)

    def run_training(self, save_model=False):
        self.load_data()
        self.init_model()
        
        self.model.fit(self.X_train, self.y_train)
        if save_model:
            self.model.save_model()
    
    def run_inference(self):
        self.load_data()
        self.init_model()
        
        self.model.predict(self.X_test)
        self.model.write_metrics(self.y_test)