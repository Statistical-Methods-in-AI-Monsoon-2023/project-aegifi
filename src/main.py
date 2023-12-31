import pickle
import argparse
import sys
import numpy as np
import json
from nltk.tokenize import word_tokenize
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 60000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100


sys.path.append('./src/naive_bayes')
sys.path.append('./src/xgboost')
sys.path.append('./src/gru')
sys.path.append('./src/transformer')

from xgb_model import XGBRunner
from binary_nb import BinaryNBRunner
from multinomial_nb import MultinomialNBRunner
from binary_gru import BinaryGRURunner
from rank_gru import RankGRURunner
from multinomial_gru import MultinomialGRURunner
from transformer import TransformerRunner

class Model:
    def __init__(self, model='xgb', word_embeddings='w2v', load_models=False):
        models = {
            'xgb': XGBRunner,
            'bnb': BinaryNBRunner,
            'mnb': MultinomialNBRunner,
            'bgru': BinaryGRURunner,
            'rgru': RankGRURunner,
            'mgru': MultinomialGRURunner,
            'trf': TransformerRunner
        }
        try:
            if model in ['xgb', 'bnb', 'mnb']:
                self.model = models[model](load_models=load_models, word_embeddings=word_embeddings)
            else:
                self.model = models[model](load_models=load_models)
        except KeyError:
            raise Exception('Invalid model name')

class Inferencer:
    def __init__(self, plot_sample, model, word_embeddings=None, genre=None):
        self.sample_X = [plot_sample]
        self.model_name = model
        self.embed_type = word_embeddings
        if genre:
            self.genre = genre
            
        self.md = Model(model=model, word_embeddings=word_embeddings, load_models=True)
        
        self.classes = [
            'Action',
            'Adventure',
            'Animation',
            'Biography',
            'Comedy',
            'Crime',
            'Drama',
            'Family',
            'Fantasy',
            'History',
            'Horror',
            'Music',
            'Musical',
            'Mystery',
            'Romance',
            'Sci-Fi',
            'Sport',
            'Thriller',
            'War',
            'Western'
        ]    

    class vector_pipeline:
        def init(self, model, vectorizer_path):
            # create a pipeline with vectorizer and model
            self.model = model
            vectorizer = None
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        def clean_text(self, text):
            """
                text: a string
                
                return: modified initial string
            """
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
            text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
            # text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
            return text
        
        def preprocess(self, X):
            cleaned_X = []
            for plot in X:
                cleaned_X.append(self.clean_text(plot))
            return cleaned_X

        def predict_proba(self, X):
            X = self.preprocess(X)
            
            # vectorize sample
            transformed = self.vectorizer.transform(X)
            
            # run inference
            output = self.model.predict_proba(transformed)
            return output
        
    class word2vec_pipeline:
        def __init__(self, model, embedding_matrix, vectorizer_path=None, word2idx_path='./vectorizers/word2idx.json'):
            self.model = model
            self.embedding_matrix = embedding_matrix
            # load word2idx
            self.word2idx = None
            with open(word2idx_path, 'rb') as f:
                self.word2idx = json.load(f)
            
            # load vectorizer if needed
            self.vectorizer = None
            if vectorizer_path:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
        
        def clean_text(self, text):
            """
                text: a string
                
                return: modified initial string
            """
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
            text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
            # text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
            return text
        
        def preprocess(self, X):
            cleaned_X = []
            for plot in X:
                cleaned_X.append(self.clean_text(plot))
            return cleaned_X
        
        def get_avg_embeddings(self, data):
            average_embeddings = []
            for plot in data:
                plot_embeddings = []
                for word in plot:
                    if word in self.word2idx:
                        plot_embeddings.append(self.embedding_matrix[self.word2idx[word]])
                    else:
                        plot_embeddings.append(self.embedding_matrix[self.word2idx['<UNK>']])
                plot_embeddings = np.array(plot_embeddings)
                
                average_embeddings.append(np.mean(plot_embeddings, axis=0))
                
            average_embeddings = np.array(average_embeddings)
            
            return average_embeddings
        
        def get_weighted_embeddings(self, data, tfidf, tfidf_features):
            average_embeddings = []
            for index, plot in enumerate(data):
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
                    # if weight == 0:
                    #     print(f'Word {word} not found in tfidf features')
                    if word in self.word2idx:
                        plot_embeddings.append(self.embedding_matrix[self.word2idx[word]] * weight)
                    else:
                        plot_embeddings.append(self.embedding_matrix[self.word2idx['<UNK>']] * weight)
                    weight_sum += weight
                plot_embeddings = np.array(plot_embeddings)
                
                if weight_sum == 0:
                    weight_sum = 1
                
                average_embeddings.append(np.mean(plot_embeddings, axis=0) / weight_sum)
                
            average_embeddings = np.array(average_embeddings)
            
            return average_embeddings

        def predict_proba(self, X):
            X = self.preprocess(X)
            if self.vectorizer:
                # print(X)
                # vectorize sample
                transformed = self.vectorizer.transform(X)
                
                sample_X = [word_tokenize(plot) for plot in X]
                
                # get weighted embeddings
                embeds = self.get_weighted_embeddings(sample_X, transformed, self.vectorizer.get_feature_names_out())
                
            else:
                sample_X = [word_tokenize(plot) for plot in X]
                # get average embeddings
                embeds = self.get_avg_embeddings(X)
                
            output = self.model.predict_proba(embeds)
            
            return output
        
    class doc2vec_pipeline:
        def __init__(self, model, model_path='./vectorizers/doc2vec.model'):
            self.embed = gensim.models.Doc2Vec.load(model_path)
            self.model = model
        
        def clean_text(self, text):
            """
                text: a string
                
                return: modified initial string
            """
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
            text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
            # text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
            return text
        
        def preprocess(self, X):
            cleaned_X = []
            for plot in X:
                cleaned_X.append(self.clean_text(plot))
            return cleaned_X
        
        def predict_proba(self, X):
            X = self.preprocess(X)
            embeddings = []
            for row in X:
                embeddings.append(self.embed.infer_vector(gensim.utils.simple_preprocess(row)))
            embeddings = np.array(embeddings)
            print(embeddings.shape)
            
            output = self.model.predict_proba(embeddings)
            
            return output
    
    class nn_pipeline:
        def __init__(self, model):
            self.model = model
        
        def clean_text(self, text):
            """
                text: a string
                
                return: modified initial string
            """
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
            text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
            # text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
            return text
        
        def tokenize(self, X):
            tokenizer = None
            with open('./vectorizers/gru_tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            word_index = tokenizer.word_index
            # print('Found %s unique tokens.' % len(word_index))
            # print("Before token", X)
            X = tokenizer.texts_to_sequences([X])
            # print("After token", X)
            X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
            # print("After pad", X.shape)
            return X
        
        def preprocess(self, X):
            X = self.clean_text(X)
            X = self.tokenize(X)
            return X

        def predict_proba(self, X):
            # print(X[0])
            preprocessed_X = []
            for plot in X:
                plot_temp = self.preprocess(plot)
                # print(plot_temp.shape)
                preprocessed_X.append(plot_temp.flatten())
            # print(preprocessed_X)
            preprocessed_X = np.array(preprocessed_X)
            output = self.model.predict_proba(preprocessed_X)

            return output
    
    def lime_explain(self, pipeline, genre, sample_X, is_bow=False):
        # get index of genre 
        genre_index = self.classes.index(genre)
        
        explainer = LimeTextExplainer(class_names=self.classes, bow=is_bow)
        if self.embed_type is None:
            explanation = explainer.explain_instance(sample_X[0], pipeline.predict_proba, num_features=10, labels=(genre_index,), num_samples=100)
        else:
            explanation = explainer.explain_instance(sample_X[0], pipeline.predict_proba, num_features=10, labels=(genre_index,), num_samples=1000)
        
        return explanation.as_list(label=genre_index)
    
    def run_lime(self):
        if self.embed_type == 'w2v':
            model = self.md.model.model
            embedding_matrix = np.load(f'vectorizers/embedding_matrix_w2v.npy')
            pipeline = self.word2vec_pipeline(model, embedding_matrix)
            return self.lime_explain(pipeline, self.genre, self.sample_X)
        elif self.embed_type == 'tf_w2v':
            model = self.md.model.model
            embedding_matrix = np.load(f'vectorizers/embedding_matrix_w2v.npy')
            pipeline = self.word2vec_pipeline(model, embedding_matrix, vectorizer_path=f'vectorizers/tfidf_vectorizer.pkl')
            return self.lime_explain(pipeline, self.genre, self.sample_X)
        elif self.embed_type == 'tfidf' or self.embed_type == 'bow':
            pipeline = self.vector_pipeline(self.md.model.model, f'vectorizers/{self.embed_type}_vectorizer.pkl')
            return self.lime_explain(pipeline, self.genre, self.sample_X, is_bow=True)
        elif self.embed_type == 'd2v':
            model = self.md.model.model
            pipeline = self.doc2vec_pipeline(model)
            return self.lime_explain(pipeline, self.genre, self.sample_X)
        else:
            model = self.md.model.model
            pipeline = self.nn_pipeline(model)
            return self.lime_explain(pipeline, self.genre, self.sample_X)
    
    def vectorizer_inf(self, vectorizer_path):
        #load vectorizer
        vectorizer = None
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Fit vectorizer on sample
        sample_X = vectorizer.transform(self.sample_X)

        # Run inference
        output = self.md.model.model.predict(sample_X)[0]

        # Output classes where the index of output is 1
        output_classes = [self.classes[i] for i in range(len(output)) if output[i] == 1]

        return output_classes
    
    def get_avg_embeddings(self, data, embedding_matrix, word2idx):
        average_embeddings = []
        for plot in data:
            plot_embeddings = []
            for word in plot:
                if word in word2idx:
                    plot_embeddings.append(embedding_matrix[word2idx[word]])
                else:
                    plot_embeddings.append(embedding_matrix[word2idx['<UNK>']])
            plot_embeddings = np.array(plot_embeddings)
            
            average_embeddings.append(np.mean(plot_embeddings, axis=0))
            
        average_embeddings = np.array(average_embeddings)
        
        return average_embeddings

    def get_weighted_embeddings(self, data, embedding_matrix, word2idx, tfidf, tfidf_features):
        average_embeddings = []
        for index, plot in enumerate(data):
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
                if weight == 0:
                    print(f'Word {word} not found in tfidf features')
                if word in word2idx:
                    plot_embeddings.append(embedding_matrix[word2idx[word]] * weight)
                else:
                    plot_embeddings.append(embedding_matrix[word2idx['<UNK>']] * weight)
                weight_sum += weight
            plot_embeddings = np.array(plot_embeddings)
            
            average_embeddings.append(np.mean(plot_embeddings, axis=0) / weight_sum)
            
        average_embeddings = np.array(average_embeddings)
        
        return average_embeddings

    def clean_text(self, text):
        """
            text: a string
            
            return: modified initial string
        """
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
        # text = re.sub(r'\W+', '', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        return text

    def tokenize(self, X):
        
        # import tokenizer
        tokenizer = None
        with open('./vectorizers/gru_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        word_index = tokenizer.word_index
        # print('Found %s unique tokens.' % len(word_index))

        X = tokenizer.texts_to_sequences([X])
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        return X

    def preprocess(self, X):
        X = self.clean_text(X)
        X = self.tokenize(X)
        return X
    
    def gru_inference(self):
        X = []
        for text in self.sample_X:
            X.append(self.preprocess(text))
            
        output = self.md.model.model.predict(X)[0]
    
        print(output)
        
        output_classes = [self.classes[i] for i in range(len(output)) if output[i] == 1]
        
        return output_classes

    def w2v_like_inf(self, embedding_path, vectorizer_path=None, w2idx_path='./vectorizers/word2idx.json'):
        embedding_matrix = np.load(embedding_path)
        # load word2idx
        word2idx = None
        with open(w2idx_path, 'rb') as f:
            word2idx = json.load(f)
        
        # load vectorizer if needed
        vectorizer = None
        if vectorizer_path:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            # vectorize sample
            transformed = vectorizer.transform(self.sample_X)
            
            self.sample_X = [word_tokenize(plot) for plot in self.sample_X]
            
            embeds = self.get_weighted_embeddings(self.sample_X, embedding_matrix, word2idx, transformed, vectorizer.get_feature_names_out())
            print(embeds)
            
        else:
            self.sample_X = [word_tokenize(plot) for plot in self.sample_X]
            
            embeds = self.get_avg_embeddings(self.sample_X, embedding_matrix, word2idx)
            
        output = self.md.model.model.predict(embeds)[0]
        output_classes = [self.classes[i] for i in range(len(output)) if output[i] == 1]
        
        return output_classes
    
    def d2v(self, model_path):
        model = gensim.models.Doc2Vec.load(model_path)
        
        X = gensim.utils.simple_preprocess(self.sample_X[0])
        embedding = model.infer_vector(X)
        
        output = self.md.model.model.predict(embedding.reshape(1, -1))[0]
        output_classes = [self.classes[i] for i in range(len(output)) if output[i] == 1]
        
        return output_classes
    
    def inference(self, vectorizer_type):
        # Determine the path based on vectorizer type
        if vectorizer_type == 'tfidf' or vectorizer_type == 'bow':
            vectorizer_path = f'vectorizers/{vectorizer_type}_vectorizer.pkl'
            return self.vectorizer_inf(vectorizer_path)
        elif vectorizer_type == 'w2v' or vectorizer_type == 'tf_w2v':
            embedding_path = f'vectorizers/embedding_matrix_w2v.npy'
            vectorizer_path = None
            if vectorizer_type == 'tf_w2v':
                vectorizer_path = f'vectorizers/tfidf_vectorizer.pkl'
            
            return self.w2v_like_inf(embedding_path, vectorizer_path)
        elif vectorizer_type == 'd2v':
            model_path = f'vectorizers/doc2vec.model'
            return self.d2v(model_path)
        else:
            raise ValueError("Invalid vectorizer_type. Supported types are 'tfidf' and 'bow'.")

def lime_run(model, word_embeddings='w2v', load_models=True, plot_sample=None, genre=None):
    
    if 'gru' in model or 'trf' in model:
        inf = Inferencer(plot_sample=plot_sample, model=model, genre=genre)
        return inf.run_lime()
    
    inf = Inferencer(plot_sample=plot_sample, model=model, word_embeddings=word_embeddings, genre=genre)
    return inf.run_lime()
    

def streamlit_run(model, word_embeddings='w2v', load_models=True, plot_sample=None,):
    
    if plot_sample:
        # run inference on a single movie plot
        
        if 'gru' in model or 'trf' in model:
            inf = Inferencer(plot_sample=plot_sample, model=model)
            return inf.gru_inference()
        
        inf = Inferencer(plot_sample=plot_sample, model=model, word_embeddings=word_embeddings)
        return inf.inference(vectorizer_type=word_embeddings)
    
    md = Model(model=model, word_embeddings=word_embeddings, load_models=load_models)
    if load_models:
        md.model.run_inference()
    else:
        md.model.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='xgb', help='Model to run')
    parser.add_argument('-w', '--word_embeddings', type=str, default='w2v', help='Word embeddings to use')
    parser.add_argument('-l', '--load_models', action='store_true', help='Whether to load pretrained models')
    parser.add_argument('--save-preds', action='store_true', help='Whether to save the predictions')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--infer', action='store_true', help='Whether to run inference on the model')

    args = parser.parse_args()
    print(args)
    
    md = Model(model=args.model, word_embeddings=args.word_embeddings, load_models=args.load_models)
    if args.train:
        md.model.run_training()
    if args.infer:
        md.model.run_inference(save_preds=args.save_preds)