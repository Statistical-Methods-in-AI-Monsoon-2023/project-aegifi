import argparse
import sys
sys.path.append('./src/naive_bayes')
sys.path.append('./src/xgboost')
sys.path.append('./src/gru')

from xgb_model import XGBRunner
from binary_nb import BinaryNBRunner
from multinomial_nb import MultinomailNBRunner
from binary_gru import BinaryGRURunner
from rank_gru import RankGRURunner
from multinomial_gru import MultinomialGRURunner

class Model:
    def __init__(self, model='xgb', word_embeddings='w2v', load_models=False):
        models = {
            'xgb': XGBRunner(word_embeddings=word_embeddings, load_models=load_models),
            'bnb': BinaryNBRunner(load_models=load_models),
            'mnb': MultinomailNBRunner(load_models=load_models),
            'bgru': BinaryGRURunner(load_models=load_models),
            'rgru': RankGRURunner(load_models=load_models),
            'mgru': MultinomialGRURunner(load_models=load_models)
        }
        try:
            self.model = models[model]
        except KeyError:
            raise Exception('Invalid model name')

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