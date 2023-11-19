import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LayerNormalization, RNN, GRUCell, SpatialDropout1D
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, jaccard_score, classification_report, hamming_loss

import sys
sys.path[0] += '/../utils/'
from utils import load_data as ld, hit_rate

MAX_NB_WORDS = 60000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save after every epoch
        self.model.save(f'./src/gru/pretrained/binary_gru_{epoch}.keras')

class BinaryGRU:
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
                tf.config.set_visible_devices(gpus[2], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)
        
        if load_models:
            self.model_name = 'binary_gru_1.keras'
            self.model = tf.keras.models.load_model(f'./src/gru/pretrained/{self.model_name}')
            print(self.model.summary())
        else:
            model = Sequential(name='BinaryGRU')
            model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
            model.add(SpatialDropout1D(self.params['dropout']))
            for i in range(self.params['layers']):
                model.add(GRU(self.params['units'], return_sequences=i != self.params['layers']-1, recurrent_dropout=self.params['dropout'], dropout=self.params['dropout']))
                model.add(Dropout(self.params['dropout']))
                model.add(LayerNormalization())
            # model.add(Dropout(self.params['dropout']))
            model.add(Dense(20, activation='sigmoid'))
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['lr'])
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])
            print(model.summary())
            self.model = model
    
    def fit(self, X, y):
        saver = CustomSaver()
        st = time.time()
        print("Fitting model...")
        
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks=[saver, EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_delta=0.0001)])
        
        self.train_time = time.time() - st
        print("Done fitting model")
        print(f"Train time: {self.train_time}")
    
    def predict(self, X):
        st = time.time()
        print("Predicting...")
        
        self.preds = self.model.predict(X)
        self.preds = np.round(self.preds)
        
        self.predict_time = time.time() - st
        print(f"Predict time: {self.predict_time}")
        return self.preds
    
    def write_metrics(self, y_test):
        file_name = f'binary_gru_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

        file_path = f'./src/gru/metrics/{file_name}'

        with open(file_path, 'w') as f:
            f.write(f'Loaded from {self.model_name}\n')
            f.write(f'params: {self.params}\n')
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(y_test, self.preds, average="micro")}\n')
            f.write(f'Hit Rate: {hit_rate(y_test, self.preds)}\n')
            f.write('Classification Report:\n')
            f.write(f'{classification_report(y_test, self.preds, zero_division=True)}\n')

    def save_model(self):
        pass
        # self.model.save(f'./src/gru/pretrained/binary_gru.keras')

class BinaryGRURunner:
    def __init__(self, load_models=False):
        self.load_models = load_models
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = ld(gru=True)
    
    def init_model(self):
        self.model = BinaryGRU(load_models=self.load_models)
    
    def run_training(self):
        self.load_data()
        self.init_model()
        
        self.model.fit(self.X_train, self.y_train)
        self.model.save_model()
    
    def run_inference(self):
        self.load_data()
        self.init_model()
        
        self.model.predict(self.X_test)
        self.model.write_metrics(self.y_test)