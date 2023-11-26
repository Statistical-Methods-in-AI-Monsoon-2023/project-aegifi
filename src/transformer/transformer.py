import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Embedding, GRU, LayerNormalization, RNN, GRUCell, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, jaccard_score, classification_report, hamming_loss, f1_score, precision_score, recall_score

import sys
sys.path[0] += '/../utils/'
from utils import load_data as ld, hit_rate

MAX_NB_WORDS = 60000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save after every epoch
        self.model.save(f'./src/transformer/pretrained/transformer_{epoch}.keras')

class Transformer:
    def __init__(self, load_models=False):
        self.train_time = 0
        self.predict_time = 0
        self.preds = None
        self.params = {
            'units': 128,
            'dropout': 0.1,
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
                print(e)
        
        if load_models:
            self.model_name = 'transformer_9.keras'
            self.model = tf.keras.models.load_model(f'./src/transformer/pretrained/{self.model_name}', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding})
            print(self.model.summary())
        else:
            model = Sequential(name='Transformer')
            model.add(TokenAndPositionEmbedding(MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM))
            model.add(TransformerBlock(EMBEDDING_DIM, 16, 256, 0.1))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dropout(self.params['dropout']))
            model.add(Dense(20, activation='sigmoid'))
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['lr'])
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])
            model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
            print(model.summary(expand_nested=True))
            self.model = model
    
    def fit(self, X, y):
        saver = CustomSaver()
        st = time.time()
        print("Fitting model...")
        
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks=[saver])
        
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
    
    def predict_proba(self, X):
        st = time.time()
        print("Predicting...")
        
        self.preds = self.model.predict(X)
        
        self.predict_time = time.time() - st
        print(f"Predict time: {self.predict_time}")
        return self.preds
    
    def write_metrics(self, y_test):
        file_name = f'transformer_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

        file_path = f'./src/transformer/metrics/{file_name}'

        with open(file_path, 'w') as f:
            # f.write(f'Loaded from {self.model_name}\n')
            # f.write(f'params: {self.params}\n')
            # self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f'Predict time: {self.predict_time}\n')
            f.write(f'Accuracy: {accuracy_score(y_test, self.preds)}\n')
            f.write(f'Hamming Score: {1 - hamming_loss(y_test, self.preds)}\n')
            f.write(f'Jaccard Score: {jaccard_score(y_test, self.preds, average="micro")}\n')
            f.write(f'Hit Rate: {hit_rate(y_test, self.preds)}\n')
            f.write(f'F1 Score: {f1_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Precision Score: {precision_score(y_test, self.preds, average="samples", zero_division=True)}\n')
            f.write(f'Recall Score: {recall_score(y_test, self.preds, average="samples", zero_division=True)}\n')

    def save_model(self):
        self.model.save(f'./src/gru/pretrained/transformer.keras', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding})

class TransformerRunner:
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
        self.model = Transformer(load_models=self.load_models)
    
    def run_training(self):
        self.load_data()
        
        self.model.fit(self.X_train, self.y_train)
        self.model.save_model()
    
    def run_inference(self):
        self.load_data()
        
        self.model.predict(self.X_test)
        self.model.write_metrics(self.y_test)