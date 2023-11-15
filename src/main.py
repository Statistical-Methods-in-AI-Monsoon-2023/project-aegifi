import argparse
import sys
sys.path.append('./src/naive_bayes')
sys.path.append('./src/xgboost')

from xgb_model import XGBRunner
from binary_nb import BinaryNBRunner
from multinomial_nb import MultinomailNBRunner

class Model:
    def __init__(self, model='xgb', word_embeddings='w2v', load_models=False):
        if model == 'xgb':
            self.model = XGBRunner(word_embeddings=word_embeddings, load_models=load_models)
        elif model == 'bnb':
            self.model = BinaryNBRunner(load_models=load_models)
        elif model == 'mnb':
            self.model = MultinomailNBRunner(load_models=load_models)
        else:
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