import os
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import multiprocessing

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file = 'data/preprocessed_data.csv'

data = pd.read_csv(data_file, index_col=0)

X = data['plot'].values
X = X[:100]

def read_corpus(X, tokens_only=False):
    for i, line in enumerate(X):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield TaggedDocument(tokens, [i])
            
train_corpus = list(read_corpus(X))
X_tokens = list(read_corpus(X, tokens_only=True))

model = Doc2Vec(vector_size=300, min_count=2, epochs=40, workers=multiprocessing.cpu_count() - 1)
model.build_vocab(train_corpus)

print(f"Word 'life' appeared {model.wv.get_vecattr('life', 'count')} times in the training corpus.")

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

embeddings = []

for i in X_tokens:
    embeddings.append(model.infer_vector(i))
embeddings = np.array(embeddings)
print(embeddings.shape)