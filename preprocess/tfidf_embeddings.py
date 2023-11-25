import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
from time import time
from tqdm import tqdm
import multiprocessing
from nltk.tokenize import word_tokenize
import pickle

df = pd.read_csv('data/preprocessed_data.csv')

X = df['plot']

st = time()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# save vectorizer
with open('vectorizers/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Time taken by vectorizer: ", time() - st)

print(X.shape)

# # save as numpy arrays
scipy.sparse.save_npz('vectorised_data/X_tfidf.npz', X)