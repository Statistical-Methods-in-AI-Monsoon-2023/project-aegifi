import gensim.downloader as api
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

df = pd.read_csv('data/filtered_plots_and_genres.csv')

# split the data into X and y
X = df['plot']

print(X.shape)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# print(X_train[:5])

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
    return text, tokens

X_train_tokens = []
X_train_text = []
for text in tqdm(X_train):
    text, tokens = preprocess(text)
    X_train_tokens.append(tokens)
    X_train_text.append(text)
# print(X_train_tokens[0], X_train_text[0])
# exit(0)

print("Creating the vocabulary...")

def create_vocab(train_sentences):
    vocab_counter = Counter([word for sentence in train_sentences for word in sentence])

    # remove the words that appear only once
    # vocab_counter = Counter({word: freq for word, freq in vocab_counter.items() if freq > 1})

    vocab = [word for word, freq in vocab_counter.items()]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    print("Vocabulary size: ", len(vocab))
    print("Most common words: ", vocab_counter.most_common(10))
    print("Least common words: ", vocab_counter.most_common()[-10:])

    # save the vocab as npy file
    np.save('vectorised_data/vocab.npy', vocab)

    return vocab, word2idx, idx2word

vocab, word2idx, idx2word = create_vocab(X_train_tokens)

