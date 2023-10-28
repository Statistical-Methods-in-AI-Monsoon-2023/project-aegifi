from gensim.models import Word2Vec
import numpy as np

def sentences_to_word2vec_embeddings(sentences, word2vec_model):
    """
    Convert a list of sentences into Word2Vec embeddings.

    Args:
    - sentences: A list of sentences, where each sentence is a list of words.
    - word2vec_model: A pre-trained Word2Vec model.

    Returns:
    - A 2D numpy array where each row represents the Word2Vec embedding for a sentence.
    """
    embeddings = []
    for sentence in sentences:
        # Filter out words that are not in the vocabulary of the Word2Vec model.
        words_in_vocab = [word for word in sentence if word in word2vec_model.wv]
        
        # If there are no words in the vocabulary, skip this sentence.
        if not words_in_vocab:
            continue
        
        # Calculate the mean of Word2Vec vectors for words in the sentence.
        sentence_embedding = np.mean(word2vec_model[words_in_vocab], axis=0)
        embeddings.append(sentence_embedding)

    return np.array(embeddings)

df = pd.read_csv('data/filtered_plots_and_genres.csv')
X = df['plot']

# convert each plot into a list of words
X = X.apply(lambda x: x.split())

#  convert X into a list of lists, where each inner list is a sentence
X = X.tolist()



# Load the pre-trained Word2Vec model.
word2vec_model = Word2Vec.load('models/word2vec.model')