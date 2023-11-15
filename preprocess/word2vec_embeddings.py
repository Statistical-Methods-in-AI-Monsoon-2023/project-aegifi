import gensim.downloader as api
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordsegment import load, segment
from collections import Counter
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import json
from time import time
import multiprocessing

df = pd.read_csv('data/filtered_plots_and_genres.csv')

# split the data into X and y
X = df['plot']

print(X.shape)
print("Preprocessing the data...")

load()

def preprocess(text):
        tokens = word_tokenize(text)

        return text, tokens

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
print("Creating the vocabulary...")

st = time()

def create_vocab(train_sentences):
    vocab_counter = Counter([word for sentence in train_sentences for word in sentence])

    # remove the words that appear only once
    # vocab_counter = Counter({word: freq for word, freq in vocab_counter.items() if freq > 1})

    vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, freq in vocab_counter.items()]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    print("Vocabulary size: ", len(vocab))
    print("Most common words: ", vocab_counter.most_common(10))
    print("Least common words: ", vocab_counter.most_common()[-10:])

    # save the vocab as npy file
    # np.save('vectorised_data/vocab.npy', vocab)
    # with open('vectorised_data/word2idx.json', 'w') as f:
    #     json.dump(word2idx, f, indent=4)

    # with open('vectorised_data/idx2word.json', 'w') as f:
    #     json.dump(idx2word, f, indent=4)
    
    return vocab, word2idx, idx2word

vocab, word2idx, idx2word = create_vocab(X_tokens)

print("Time taken by vocab creation: ", time() - st)

embedding_dim = 300

print(list(api.info()['models'].keys()))
word_vectors = api.load("word2vec-google-news-300")

# vocab = np.load('vectorised_data/vocab.npy', allow_pickle=True)

print("Creating Embedding Matrix...")
st = time()
embedding_matrix = []
num_unknown_words = 0
for word in tqdm(vocab):
    if word in word_vectors:
        # embedding_matrix.append(word_vectors[word])
        # print("Word {} found in w2v dict".format(word))
        if len(word_vectors[word]) != embedding_dim:
            print("Word {} size {} does not match embedding dim {}".format(
                word, len(word_vectors[word]), embedding_dim))
        else:
            embedding_matrix.append(word_vectors[word])
    else:
        # print("Word {} not in w2v dict".format(word))
        num_unknown_words += 1
        embedding_matrix.append([0]*embedding_dim)

print("Time taken: ", time() - st)
print("Number of unknown words: ", num_unknown_words)

print("Embedding matrix size: ", len(embedding_matrix))
# print("Embedding size: ", len(embedding_matrix[0]))
# print("Embedding matrix first row: ", embedding_matrix[56])

embedding_matrix = np.array(embedding_matrix) # since converting a list directly to tensor is very slow
print("Embedding matrix shape: ", embedding_matrix.shape)

# save the embedding matrix locally as npy file
np.save('embeddings/embedding_matrix.npy', embedding_matrix)