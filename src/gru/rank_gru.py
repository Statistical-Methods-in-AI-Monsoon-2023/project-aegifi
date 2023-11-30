import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LayerNormalization, RNN, GRUCell, SpatialDropout1D
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from keras.layers import Dropout
import time
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score, jaccard_score, classification_report, hamming_loss, f1_score, precision_score, recall_score

import sys
sys.path[0] += '/../../utils/'
from utils import load_data as ld, hit_rate
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 60000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save after every epoch
        if epoch % 5 == 0:
            self.model.save(f'./src/gru/pretrained/rank_gru_{epoch}.keras')

class RankGRU:
    def __init__(self, load_models=False):
        self.train_time = 0
        self.predict_time = 0
        self.preds = None
        self.params = {
            'units': 128,
            'dropout': 0.2,
            'layers': 2,
            'batch_size': 128,
            'epochs': 10,
            'lr': 0.001,
        }
        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']
        self.train_time = 0
        self.predict_time = 0
        self.create_model(load_models)
    
    def create_model(self, load_models=False):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        
        self.xgb_model = xgb.XGBRegressor(verbosity=2, tree_method="hist", n_jobs=39)
        if load_models:
            self.model_name = 'rank_gru.keras'
            self.model = tf.keras.models.load_model(f'./src/gru/pretrained/{self.model_name}')
            print(self.model.summary())
            self.xgb_model.load_model(f'./src/gru/pretrained/xgb_reg_rank.json')
        else:
            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
            model.add(SpatialDropout1D(self.params['dropout']))
            for i in range(self.params['layers']):
                model.add(GRU(self.params['units'], return_sequences=i != self.params['layers']-1, recurrent_dropout=self.params['dropout'], dropout=self.params['dropout']))
                # model.add(Dropout(self.params['dropout']))
                model.add(LayerNormalization())
            model.add(Dense(20, activation='sigmoid'))
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['lr'])
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])
            print(model.summary())
            self.model = model
    
    def get_thresholds(self, y_prob, y):
        thresholds = []
        for prob, true_labels in zip(y_prob, y):
            errors = np.sum(((prob > prob[:, None]) & (true_labels == 0) | (prob <= prob[:, None]) & (true_labels == 1)) , axis=1)
            thresholds.append(prob[np.argmin(errors)])
        return thresholds

    
    def fit(self, X, y):
        saver = CustomSaver()
        
        st = time.time()
        print("Fitting GRU...")
        
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks=[saver])
        
        print("Done fitting GRU")
        
        probs = self.model.predict(X)
        thresholds = self.get_thresholds(probs, y)
        
        print("Fitting XGB...")
        
        self.xgb_model.fit(probs, thresholds)
        
        print("Done fitting XGB")
        
        
        self.train_time = time.time() - st
        print("Done fitting model")
        print(f"Train time: {self.train_time}")
    
    def predict(self, X):
        st = time.time()
        print("Predicting...")
        
        preds = self.model.predict(X)
        pred_thresholds = self.xgb_model.predict(preds)
        self.preds = (preds >= pred_thresholds[:, None]).astype(int)
        
        self.predict_time = time.time() - st
        print(f"Predict time: {self.predict_time}")
        return self.preds
    
    def predict_proba(self, X):
        print("Predicting...")
        
        preds = self.model.predict(X)
        return preds
    
    def write_metrics(self, y_test):
        file_name = f'rank_gru_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

        file_path = f'./src/gru/metrics/{file_name}'

        with open(file_path, 'w') as f:
            # f.write(f'Loaded from {self.model_name}\n')
            # f.write(f'params: {self.params}\n')
            # self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(y_test, self.preds, average="samples")}\n')
            f.write(f'Hit Rate: {hit_rate(y_test, self.preds)}\n')
            f.write(f'F1 Score: {f1_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Precision Score: {precision_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Recall Score: {recall_score(y_test, self.preds, average="samples", zero_division=True)}\n')

    def save_model(self):
        self.xgb_model.save_model(f'./src/gru/pretrained/xgb_reg_rank.json')

class RankGRURunner:
    def __init__(self, load_models=False):
        self.load_models = load_models
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.init_model()
    
    def load_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = ld(word_embeddings='gru')
    
    
    def init_model(self):
        self.model = RankGRU(load_models=self.load_models)
    
    def run_training(self):
        self.load_data()
        
        self.model.fit(self.X_train, self.y_train)
        self.model.save_model()
    
    def run_inference(self, save_preds=False):
        self.load_data()
        
        preds = self.model.predict(self.X_test)
        
        if save_preds:
            np.save(f'EDA/preds/rank_gru.npy', preds)
            np.save(f'EDA/preds/y_test_rank_gru.npy', self.y_test)
            
        self.model.write_metrics(self.y_test)
