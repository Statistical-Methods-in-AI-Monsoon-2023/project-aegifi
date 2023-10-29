import gensim.downloader as api
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

print(list(api.info()['models'].keys()))
# if not 'word_vectors' in locals():  # so that it doesn't load again if it was already loaded
word_vectors = api.load("word2vec-google-news-300")

vocab = np.load('vectorised_data/vocab.npy', allow_pickle=True)

embedding_matrix = []
num_unknown_words = 0
for word in vocab:
    if word in word_vectors:
        # embedding_matrix.append(word_vectors[word])
        # print("Word {} found in w2v dict".format(word))
        if len(word_vectors[word]) != config["embedding_dim"]:
            print("Word {} size {} does not match embedding dim {}".format(
                word, len(word_vectors[word]), config["embedding_dim"]))
        else:
            embedding_matrix.append(word_vectors[word])
    else:
        # print("Word {} not in w2v dict".format(word))
        num_unknown_words += 1
        embedding_matrix.append([0]*config["embedding_dim"])

print("Number of unknown words: ", num_unknown_words)

print("Embedding matrix size: ", len(embedding_matrix))
print("Embedding size: ", len(embedding_matrix[0]))
print("Embedding matrix first row: ", embedding_matrix[56])

embedding_matrix = np.array(embedding_matrix) # since converting a list directly to tensor is very slow
print("Embedding matrix shape: ", embedding_matrix.shape)

# save the embedding matrix locally as npy file
np.save('embeddings/embedding_matrix.npy', embedding_matrix)