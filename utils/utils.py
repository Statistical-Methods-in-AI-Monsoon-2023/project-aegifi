# utils file
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
import multiprocessing

def preprocess(text):
        tokens = word_tokenize(text)
        return text,tokens

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


def get_avg_embeddings(data, embedding_matrix, word2idx, name=None):
    print('Getting average embeddings')
    average_embeddings = []
    for plot in tqdm(data):
        plot_embeddings = []
        for word in plot:
            if word in word2idx:
                plot_embeddings.append(embedding_matrix[word2idx[word]])
            else:
                plot_embeddings.append(embedding_matrix[word2idx['<UNK>']])
        plot_embeddings = np.array(plot_embeddings)
        
        average_embeddings.append(np.mean(plot_embeddings, axis=0))
        
    average_embeddings = np.array(average_embeddings)
    print(average_embeddings.shape)
    print(average_embeddings[0])
    
    # np.save(f'embeddings/{name}.npy', average_embeddings)
    return average_embeddings

def load_w2v():
    # embed_matrix = np.load('embeddings/embedding_matrix.npy', allow_pickle=True)
    # word2idx = json.load(open('embeddings/word2idx.json', 'r'))
    
    # df = pd.read_csv('data/preprocessed_data.csv')
    # X = df['plot']
    # tokenize the data using word_tokenize
    # _,X_tokens = preprocess_data(X)
    # get the average embeddings for the data
    # X_embed = get_avg_embeddings(X_tokens, embed_matrix, word2idx, 'embeddings/embed.npy')
    
    X_train = np.load('embeddings/train_embed.npy')
    X_test = np.load('embeddings/test_embed.npy')
    
    y = np.load('vectorised_data/y.npy')
    
    # split data into train and test
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    print('Loaded data')
    
    return X_train, X_test, y_train, y_test

def load_data(w2v=False):
    if w2v:
        return load_w2v()
    
    X = scipy.sparse.load_npz('vectorised_data/X.npz')
    y = np.load('vectorised_data/y.npy')
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print('Loaded data')

    return X_train, X_test, y_train, y_test

def hit_rate(predicted, true):
    # Use element-wise logical AND to check if at least one class is predicted correctly
    correct_predictions = np.any(np.logical_and(predicted, true), axis=1)
    
    # Calculate the hit rate as the percentage of correct samples
    hit_rate = (np.sum(correct_predictions) / len(correct_predictions))
    
    return hit_rate