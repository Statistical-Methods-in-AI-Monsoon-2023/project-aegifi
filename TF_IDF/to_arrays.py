import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordsegment import load, segment
import json
from time import time
import multiprocessing
import re
from tqdm import tqdm

df = pd.read_csv('data/filtered_plots_and_genres.csv')

mlb = MultiLabelBinarizer()

# split the data into X and y
X = df['plot']
y = df['genres']

print(X.shape, y.shape)

y = y.str.strip('][').str.split(', ')
# remove quotes from each item in the list
y = y.apply(lambda row: [item.strip().strip("'") for item in row])

# transform the y data
y = mlb.fit_transform(y)

print("Preprocessing the data...")

load()

def preprocess(text):
        """
        Preprocesses the text
        """    
        text = text.lower()
        text = re.sub(r"[^a-zA-Z;.]+", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([;.])', r' \1 ', text)
        tokens = word_tokenize(text)
        
        filtered_tokens = []
        for word in tokens:
            sep = segment(word)
            if len(sep) > 1:
                filtered_tokens.extend(sep)
            else:
                filtered_tokens.append(word)
        tokens = filtered_tokens
        tokens = [word for word in tokens if word not in (stopwords.words('english') + ['.', ';'])]
        tokens = ['<EOS>' if token in ['.', ';'] else token for token in tokens]
        
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text, tokens

def preprocess_parallel(args):
    text = args[0]
    return preprocess(text)

def preprocess_data(data):

    num_processes = multiprocessing.cpu_count() -1
    print(f"Running {num_processes} processes in parallel")

    pool = multiprocessing.Pool(processes=num_processes)

    result_list = list(tqdm(pool.imap(preprocess_parallel, [(text,) for text in data]), total=len(data)))

    pool.close()
    pool.join()

    return zip(*result_list)

st = time()
X_text, X_tokens = preprocess_data(X)

print("Time taken by preprocessing: ", time() - st)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
vectorizer.get_feature_names_out()

print(X.shape, y.shape)

# save as numpy arrays
scipy.sparse.save_npz('X.npz', X)
np.save('y.npy', y)
