import pickle
import argparse
import sys
sys.path.append('./src/naive_bayes')
sys.path.append('./src/xgboost')
sys.path.append('./src/gru')

from xgb_model import XGBRunner
from binary_nb import BinaryNBRunner
from multinomial_nb import MultinomialNBRunner
from binary_gru import BinaryGRURunner
from rank_gru import RankGRURunner
from multinomial_gru import MultinomialGRURunner

class Model:
    def __init__(self, model='xgb', word_embeddings='w2v', load_models=False):
        models = {
            'xgb': XGBRunner,
            'bnb': BinaryNBRunner,
            'mnb': MultinomialNBRunner,
            'bgru': BinaryGRURunner,
            'rgru': RankGRURunner,
            'mgru': MultinomialGRURunner
        }
        try:
            if model in ['xgb', 'bnb', 'mnb']:
                self.model = models[model](load_models=load_models, word_embeddings=word_embeddings)
            else:
                self.model = models[model](load_models=load_models)
        except KeyError:
            raise Exception('Invalid model name')

class Inferencer:
    def __init__(self, plot_sample, model, word_embeddings='w2v'):
        self.sample_X = plot_sample
        self.model_name = model
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
    
    def tfidf_inference(self):
        # load vectorizer
        vectorizer = None
        with open('vectorizer/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # fit vectorizer on sample
        sample_X = vectorizer.transform([self.sample_X])
        
        # run inference
        output = self.md.model.model.predict(sample_X)
        
        # output classes where index of output is 1
        output_classes = []
        for i in range(len(output)):
            if output[i] == 1:
                output_classes.append(self.classes[i])
        
        return output_classes

    def bow_inference(self):
        # load vectorizer
        vectorizer = None
        with open('vectorizer/bow_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # fit vectorizer on sample
        sample_X = vectorizer.transform([self.sample_X])
        
        # run inference
        output = self.md.model.model.predict(sample_X)
        
        # output classes where index of output is 1
        output_classes = []
        for i in range(len(output)):
            if output[i] == 1:
                output_classes.append(self.classes[i])
        
        return output_classes

def streamlit_run(model, word_embeddings='w2v', load_models=True, plot_sample=None):
    
    if plot_sample:
        # run inference on a single movie plot
        inf = Inferencer(plot_sample=plot_sample, model=model, word_embeddings=word_embeddings)
        return inf.bow_inference()
    
    md = Model(model=model, word_embeddings=word_embeddings, load_models=load_models)
    if load_models:
        md.model.run_inference()
    else:
        md.model.run_training()

if __name__ == '__main__':
    
    # take in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='xgb', help='Model to run')
    parser.add_argument('-w', '--word_embeddings', type=str, default='w2v', help='Word embeddings to use')
    parser.add_argument('-l', '--load_models', action='store_true', help='Whether to load pretrained models')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--infer', action='store_true', help='Whether to run inference on the model')

    args = parser.parse_args()
    print(args)
    
    md = Model(model=args.model, word_embeddings=args.word_embeddings, load_models=args.load_models)
    if args.train:
        md.model.run_training()
    if args.infer:
        md.model.run_inference()