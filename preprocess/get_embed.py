import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize

df = pd.read_csv('data/filtered_plots_and_genres.csv')

X = df['plot']

print(X.shape)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

embed_matrix = np.load('embeddings/embedding_matrix.npy', allow_pickle=True)
word2idx = json.load(open('vectorised_data/word2idx.json', 'r'))
idx2word = json.load(open('vectorised_data/idx2word.json', 'r'))

print("Preprocessing the data...")

def preprocess(text):
    """
    Preprocesses the text
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z;.]+", ' ', text)
    text = text.replace('  ', ' ')
    text = text.replace(';', ' ; ')
    text = text.replace('.', ' . ')
    text = text.replace('  ', ' ')
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] == '.' or tokens[i] == ';':
            tokens[i] = '<EOS>'            
    return text, tokens

X_train_tokens = []
X_train_text = []
for text in tqdm(X_train):
    text, tokens = preprocess(text)
    X_train_tokens.append(tokens)
    X_train_text.append(text)

# print(X_train_tokens[0], X_train_text[0])
average_embeddings = []
for plot in X_train_tokens:
    plot_embeddings = []
    for word in plot:
        if word in word2idx:
            plot_embeddings.append(embed_matrix[word2idx[word]])
        else:
            plot_embeddings.append(embed_matrix[word2idx['<UNK>']])
    plot_embeddings = np.array(plot_embeddings)
    
    average_embeddings.append(np.mean(plot_embeddings, axis=0))
    
average_embeddings = np.array(average_embeddings)
print(average_embeddings.shape)
print(average_embeddings[0])
np.save('embeddings/train_embeddings.npy', average_embeddings)
