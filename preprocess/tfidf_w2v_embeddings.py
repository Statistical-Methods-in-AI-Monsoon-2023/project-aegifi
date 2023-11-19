import gensim.downloader as api
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from time import time
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/preprocessed_data.csv')

# split the data into X and y
X = df['plot'].values
print(X.shape)

#########################################################################

print("Fitting TFIDF Vectorizer...")

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

tfidf_features = vectorizer.get_feature_names_out()

#########################################################################

print("\nPreprocessing the data...")

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

st = time()
X_text, X_tokens = preprocess_data(X)

print("Time taken by preprocessing: ", time() - st, end='\n\n')

#########################################################################

print("Creating the vocabulary...")

st = time()

def create_vocab(train_sentences):
    vocab_counter = Counter([word for sentence in train_sentences for word in sentence])

    vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, freq in vocab_counter.items()]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    print("Vocabulary size: ", len(vocab))
    print("Most common words: ", vocab_counter.most_common(10))
    print("Least common words: ", vocab_counter.most_common()[-10:])
    
    return vocab, word2idx, idx2word

vocab, word2idx, idx2word = create_vocab(X_tokens)

print("Time taken by vocab creation: ", time() - st, end='\n\n')

#########################################################################

print("Creating Embedding Matrix...")

embedding_dim = 300

print(list(api.info()['models'].keys()))
word_vectors = api.load("word2vec-google-news-300")

st = time()
embedding_matrix = []
num_unknown_words = 0
for word in tqdm(vocab):
    if word in word_vectors:
        if len(word_vectors[word]) != embedding_dim:
            print("Word {} size {} does not match embedding dim {}".format(
                word, len(word_vectors[word]), embedding_dim))
        else:
            embedding_matrix.append(word_vectors[word])
    else:
        num_unknown_words += 1
        embedding_matrix.append([0]*embedding_dim)

print("Time taken: ", time() - st)
print("Number of unknown words: ", num_unknown_words)

print("Embedding matrix size: ", len(embedding_matrix))

embedding_matrix = np.array(embedding_matrix)
print("Embedding matrix shape: ", embedding_matrix.shape, end='\n\n')

#########################################################################

print("Getting average embeddings...")

def get_weighted_embeddings(data, embedding_matrix, word2idx, tfidf, tfidf_features):
    average_embeddings = []
    for index, plot in enumerate(tqdm(data)):
        plot_embeddings = []
        weight_sum = 0
        feature_index = tfidf[index,:].nonzero()[1]
        tfidf_words = tfidf_features[feature_index]
        tfidf_scores = [tfidf[index, x] for x in feature_index]
        for word in plot:
            weight = 0
            for feature, score in zip(tfidf_words, tfidf_scores):
                if feature == word:
                    weight = score
                    break
            if word in word2idx:
                plot_embeddings.append(embedding_matrix[word2idx[word]] * weight)
            else:
                plot_embeddings.append(embedding_matrix[word2idx['<UNK>']] * weight)
            weight_sum += weight
        plot_embeddings = np.array(plot_embeddings)
        
        average_embeddings.append(np.mean(plot_embeddings, axis=0) / weight_sum)
        
    average_embeddings = np.array(average_embeddings)
    
    return average_embeddings

X_embed = get_weighted_embeddings(X_tokens, embedding_matrix, word2idx, X_tfidf, tfidf_features)
# print(X_embed)

print("Shape of TFIDF weighted embeddings: ", X_embed.shape)

np.save(f'vectorised_data/X_tfidf_w2v.npy', X_embed)